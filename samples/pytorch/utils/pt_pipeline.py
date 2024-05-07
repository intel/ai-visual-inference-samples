import torch
import os
import intel_extension_for_pytorch as ipex
from samples.common.logger import init_and_get_logger, init_and_get_metrics_logger
from samples.common.get_dirs import labels_dir
from samples.common.frame_counter_stream import FrameCounterStream
from samples.common.utils import get_stop_condition
from samples.common.utils import text_file_to_list
from samples.common.stop_condition import (
    EosStopCondition,
    FramesProcessedStopCondition,
)

LABELS_PATH = labels_dir / "imagenet_2012.txt"

try:
    import intel_visual_ai
except (ImportError, ModuleNotFoundError):
    raise ModuleNotFoundError(
        f"Cannot import intel_visual_ai from syspath. Please install it or add path to src/python/intel_visual_ai to PYTHONPATH"
    )

from intel_visual_ai.metrics import MetricsTimer
from intel_visual_ai.stream import Stream
from intel_visual_ai.multi_stream_videoreader import MultiStreamVideoReader


def is_fp16_model(model):
    model_params_fp16 = [
        True if param.dtype == torch.float16 else False for param in model.parameters()
    ]
    return True if model_params_fp16 and all(model_params_fp16) else False


class PtPipeline:
    def __init__(
        self,
        model,
        args,
        model_width,
        model_height,
        logger=None,
    ):
        self.media_path = str(args.input)
        self.model_name = args.sample_name

        self.logger = (
            logger
            if logger is not None
            else init_and_get_logger(self.model_name, log_dir=args.output_dir)
        )
        decode_device = args.decode_device if args.decode_device is not None else args.device
        self._device = args.device
        self.logger.info(
            f"""{self.model_name} Pipeline Configuration
            media_path:       {self.media_path}
            inference_device: {args.device}
            decode_device:    {decode_device}
            extra_arguments:  {args}
            """
        )

        self._resize_with_torch = True if args.resize_method == "torch" else False
        if "cpu" in [decode_device, args.device] or args.inference_only:
            self.read_video = self.read_video_cpu
            self._resize_with_torch = True

        self._fp16_model = False
        if is_fp16_model(model):
            self._fp16_model = True

        self.model = model
        self._model_width = model_width
        self._model_height = model_height
        self.args = args
        self.batch_size = int(args.batch_size)
        self.transforms = None
        self.set_tranform()
        self.labels = []

        if args.labels_path:
            self.labels = text_file_to_list(args.labels_path)
        if self.args.watermark:
            os.makedirs(args.output_dir, exist_ok=True)
        self.metrics_logger = init_and_get_metrics_logger(
            sample_name=self.model_name,
            log_dir=self.args.output_dir,
            device=self.args.device,
        )
        self.__stop_condition = get_stop_condition(args)
        self.__play_in_loop = False if isinstance(self.__stop_condition, EosStopCondition) else True
        self._warmup()
        self.video = MultiStreamVideoReader()
        for _ in range(self.args.num_streams):
            self.video.add_stream(self.read_video())

    def set_tranform(self):
        pass

    def decode(self, input):
        return next(input).to(self._device)

    def preprocess(self, tensor):
        tensor = tensor.float() / 255.0
        if self._fp16_model:
            tensor = tensor.half()
        if self.transforms is not None:
            tensor = self.transforms(tensor)

        return tensor

    def inference(self, model_input_tensor):
        #########################
        # Inference
        # ----------------------------------
        # Applies :func `torch.no_grad` context-manager to disable gradient calculation for inferencing and lower memory consumption
        # The model output is a [N, 1000] tensor, here [1, 1000] for batch = 1 and the second dimension corresponds to the classes of the ImageNet 1K dataset
        # :func `torch.argmax` is used to return the indices of the maximum value of all elements in the input tensor.
        with torch.no_grad():
            outputs = self.model(model_input_tensor)
        return outputs

    def watermark(self, decoded_tensor, text, frame_id):
        pass

    def process_outputs(self, frame_counter, decoded_tensors, outputs):
        pass

    def store_output_to_csv(self, outputs):
        pass

    def read_video(self):
        video_reader = intel_visual_ai.VideoReader(self.media_path)
        # The default memory format for videoReader is set as torch_contiguous_format
        # It can be changed to torch_channels_last by setting
        # video_reader._c.set_memory_format(intel_visual_ai.XpuMemoryFormat.torch_channels_last)
        if not self._resize_with_torch:
            video_reader._c.set_output_resolution(self._model_width, self._model_height)
        video_reader._c.set_loop_mode(self.__play_in_loop)

        video_reader._c.set_async_depth(self.args.async_decoding_depth * self.batch_size)
        video_reader._c.set_frame_pool_params(
            (self.args.async_decoding_depth + 2) * self.batch_size
        )

        video_reader._c.set_batch_size(self.batch_size)
        return Stream(video_reader, backend_type="pytorch")

    def read_video_cpu(self):
        from samples.pytorch.utils.torchvision_videoreader import TorchvisionVideoReaderWithLoopMode

        return Stream(
            TorchvisionVideoReaderWithLoopMode(
                str(self.media_path), "video", loop_mode=self.__play_in_loop
            )
        )

    def _synchronize(self):
        if "xpu" in self.args.device:
            torch.xpu.synchronize()

    def _warmup(self):
        #########################
        # Warm Up reading few frames from video
        self.logger.info(
            f"Opening Warmup stream and running warmup for iterations {self.args.warmup_iterations}"
        )
        warmup_video = MultiStreamVideoReader()
        warmup_video.add_stream(self.read_video())
        warmup_frame = self.decode(warmup_video)
        tensors = []
        tensor = self.preprocess(warmup_frame)
        for _ in range(self.batch_size):
            tensors.append(tensor)
        batched_tensor = torch.stack(tensors)
        for _ in range(self.args.warmup_iterations):
            self.inference(batched_tensor)
            self._synchronize()
        self.logger.info("End of warm up phase")

    def run(self):
        if self.args.inference_only:
            self._run_inference_only_pipeline()
        else:
            self._run_full_pipeline()

    def _run_full_pipeline(self):
        if self.args.output_csv:
            iteration_outputs = []
        if isinstance(self.__stop_condition, (EosStopCondition, FramesProcessedStopCondition)):
            self.__stop_condition.set_stream(self.video)
        with MetricsTimer(
            self.video,
            batch_size=self.args.batch_size,
            logger=self.metrics_logger,
            print_fps_to_stdout=self.args.live_fps,
            output_dir=self.args.output_dir,
        ):
            while not self.__stop_condition.stopped:
                decoded_tensors = []
                while len(decoded_tensors) != self.batch_size:
                    decoded_tensor = None
                    try:
                        decoded_tensor = self.decode(self.video)
                    except StopIteration as e:
                        break
                    decoded_tensors.append(decoded_tensor)
                if not decoded_tensors:
                    break
                batched_tensor = torch.stack(decoded_tensors)
                batched_tensor = self.preprocess(batched_tensor)
                outputs = self.inference(batched_tensor)
                if self.args.output_csv:
                    iteration_outputs.append(outputs)
                if self.args.watermark or self.args.log_predictions:
                    self.process_outputs(self.video.frames_processed, decoded_tensors, outputs)
            self._synchronize()
            if self.args.output_csv:
                self.store_output_to_csv(iteration_outputs)

    def _run_inference_only_pipeline(self):
        processed_tensors = []
        frame = self.decode(self.video)
        for _ in range(self.batch_size):
            processed_tensor = self.preprocess(frame)
            processed_tensors.append(processed_tensor)
        batched_tensor = torch.stack(processed_tensors)
        self._synchronize()

        frame_counter = FrameCounterStream(self.batch_size)
        if isinstance(self.__stop_condition, FramesProcessedStopCondition):
            self.__stop_condition.set_stream(frame_counter)
        with MetricsTimer(
            frame_counter,
            batch_size=self.args.batch_size,
            logger=self.metrics_logger,
            print_fps_to_stdout=False,
            output_dir=self.args.output_dir,
        ):
            while not self.__stop_condition.stopped:
                outputs = self.inference(batched_tensor)
            self._synchronize()


def optimize_model(model, args, convert_to_fp16=False):

    if convert_to_fp16:
        model = model.half()
    model = model.to(args.device)

    if "xpu" in args.device:
        model = ipex.optimize(model)
    return model


def jit_optimize_model(model, args, width=1, height=1, trace=False):
    if args.disable_jit:
        return model
    if trace:
        random_tensor = torch.randn((1, 3, width, height), device=args.device)
        if is_fp16_model(model):
            random_tensor = random_tensor.half()
        model = torch.jit.trace(model, random_tensor)
    else:
        print("Running Jit Optimization on model")
        model = torch.jit.script(model)

    return model
