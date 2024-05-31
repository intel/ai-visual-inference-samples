import argparse
import sys
import warnings
from pathlib import Path

import torchvision

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print("No Ipex available")

try:
    import intel_visual_ai
except ImportError:
    print("No intel_visual_ai available")

import torch

warnings.filterwarnings("ignore")

ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_PATH))

from samples.pytorch.utils.metrics import MetricsTimer


class Precision:
    FP16 = "fp16"
    FP32 = "fp32"


class Device:
    CPU = "cpu"
    XPU = "xpu"
    CUDA = "cuda"


class InferenceArgs:
    # These args are pretty much common to all inferencer runs
    def __init__(
        self,
        num_frames,
        iterations,
        batch_size,
        inference_only=False,
        process_outputs=True,
        warmup_iterations=4,
        cpu_decode=False,
    ):
        self.num_frames = num_frames
        self.iterations = iterations
        self.batch_size = batch_size
        self.inference_only = inference_only
        self.process_outputs = process_outputs
        self.warmup_iterations = warmup_iterations
        self.cpu_decode = cpu_decode
        if self.process_outputs and self.inference_only:
            raise ValueError("Shouldn't try inference-only AND process-outputs...")


class _AbstractModelInferencer:
    @property
    def MODEL_NAME(self):
        raise NotImplementedError("need to set MODEL_NAME on your Inferencer class")

    def __init__(self, media_path, inference_args):

        if not Path(media_path).is_file():
            raise ValueError(f"Cannot find input media {media_path}")

        self.media_path = str(media_path)
        self.inference_args = inference_args
        self.device = None  # set in _model_init()
        self.model = self._model_init()
        print(f"Using device: {self.device}")
        # None to ensure we initialize metrics properly
        self.metrics = None
        self._avg_latency = None
        self._total_fps = None

        # set override functions based on device
        if self.inference_args.cpu_decode:
            print("Running decoding on CPU")
            self.read_video = self._read_video_cpu
            self._decode = self._decode_cpu
        elif self.device == Device.CUDA:
            print("CUDA requires cpu-decode")
            self.read_video = self._read_video_cpu
            self._decode = self._decode_cpu
        else:
            print(f"Running decoding on GPU ({self.device})")
            self.read_video = self._read_video_xpu
            self._decode = self._decode_gpu

    ############################################################
    # override these _* methods for your specific model, so interface can remain consistent
    def _model_readin(self):
        raise NotImplementedError("need to implement _model_readin()")

    def _model_specific_init(self, model):
        return model

    def _preprocess(self, batched_tensor):
        raise NotImplementedError("need to implement _preprocess()")

    def _inference(self, model_input_tensor):
        raise NotImplementedError("need to implement _inference()")

    def _process_outputs(self, **kwargs):
        raise NotImplementedError("need to implement _process_outputs()")

    def _warmup(self):
        print(f"running {self.inference_args.warmup_iterations} warmup iterations")
        video = self.read_video()
        for _ in range(self.inference_args.warmup_iterations):
            batched_tensor, _ = self._get_batched_tensor(video)
            self.inference(batched_tensor)
        print("completed warmup")

    ############################################################
    # deferred methods to keep common loops easier
    def decode(self, video):
        return self._decode(video)

    def preprocess(self, tensor):
        return self._preprocess(tensor)

    def inference(self, model_input_tensor):
        return self._inference(model_input_tensor)

    def process_outputs(self, **kwargs):
        return self._process_outputs(**kwargs)

    def warmup(self):
        return self._warmup()

    ############################################################

    ############################################################
    # we don't expect to override these but may be necessary in specific cases
    def _model_init(self):
        model = self._model_readin()
        model.eval()
        self.device = (
            Device.CUDA
            if hasattr(torch, Device.CUDA) and getattr(torch, Device.CUDA).is_available()
            else (
                Device.XPU
                if hasattr(torch, Device.XPU) and getattr(torch, Device.XPU).is_available()
                else Device.CPU
            )
        )
        model = self._model_specific_init(model)
        if self.device == Device.XPU:
            model = ipex.optimize(model)
        return model

    def _decode_default(self, video):
        return next(video)

    ############################################################
    # TODO: no read-video-cuda?
    def _read_video_xpu(self):
        intel_visual_ai.set_video_backend(self.device)
        intel_visual_ai.set_video_backend_params(loop_mode=True)
        video_reader = intel_visual_ai.VideoReader(self.media_path)
        return video_reader

    def _decode_gpu(self, video):
        return self._decode_default(video)

    ############################################################
    def _read_video_cpu(self):
        from samples.pytorch.utils.torchvision_videoreader import TorchvisionVideoReaderWithLoopMode

        return TorchvisionVideoReaderWithLoopMode(self.media_path, "video", loop_mode=True)

    def _decode_cpu(self, video):
        return self._decode_default(video)["data"]

    ############################################################
    def _reset_metrics(self):
        self.metrics = MetricsTimer(self.MODEL_NAME, ["file"], device=self.device)
        self._avg_latency = 0
        self._total_fps = 0

    def report_metrics(self, n_frames, iteration):
        self.metrics.end_timer()
        _, latency, fps = self.metrics.cal_current_itr_metrics(
            n_frames, iteration, self.inference_args.batch_size
        )

        self._total_fps += fps
        self._avg_latency += latency
        self.metrics.cal_all_itr_metrics(iteration, self._total_fps, self._avg_latency)

    ############################################################
    # TODO: move read_video() inside this func
    def _get_batched_tensor(self, video):
        decoded_tensors = None
        if self.inference_args.inference_only:
            # for inference-only, decode single frame, preprocess it, and duplicate that to batch_size, so it's right shape
            decoded_tensor = self.decode(video)
            processed_tensors = [
                self.preprocess(decoded_tensor) for _ in range(self.inference_args.batch_size)
            ]
        else:
            decoded_tensors = []
            processed_tensors = []
            for _ in range(self.inference_args.batch_size):
                decoded_tensor = self.decode(video)
                processed_tensor = self.preprocess(decoded_tensor)
                decoded_tensors.append(decoded_tensor)
                processed_tensors.append(processed_tensor)
        batched_tensor = torch.stack(processed_tensors)
        return batched_tensor, decoded_tensors

    def _run_inference_only_pipeline(self):
        video = self.read_video()
        batched_tensor, _ = self._get_batched_tensor(video)
        for itr in range(self.inference_args.iterations):
            self.metrics.start_timer()
            n_frames = 0
            while n_frames < self.inference_args.num_frames:
                n_frames += self.inference_args.batch_size
                _ = self.inference(batched_tensor)
                # doing xpu.synchronize() help get more accurate timings
                # sometimes we've done to("cpu") but some models send too much data
                if self.device != Device.CPU:
                    getattr(torch, self.device).synchronize()
            self.report_metrics(n_frames, itr + 1)

    # TODO: could combine full and inference-only with like 2 if-statements?
    def _run_full_pipeline(self):
        video = self.read_video()
        for itr in range(self.inference_args.iterations):
            self.metrics.start_timer()
            n_frames = 0
            while n_frames < self.inference_args.num_frames:
                batched_tensor, decoded_tensors = self._get_batched_tensor(video)
                n_frames += self.inference_args.batch_size
                outputs = self.inference(batched_tensor)
                if self.inference_args.process_outputs:
                    # NOTE: not every process_outputs will use all arguments
                    self.process_outputs(
                        outputs=outputs, decoded_tensors=decoded_tensors, n_frames=n_frames
                    )
            self.report_metrics(n_frames, itr + 1)

    def run(self):
        self._reset_metrics()
        self.warmup()
        self._reset_metrics()
        if self.inference_args.inference_only:
            self._run_inference_only_pipeline()
        else:
            self._run_full_pipeline()


def add_inference_args(parser, default_media_path=None):
    parser.add_argument(
        "--input", type=Path, default=default_media_path, help="Path to input media"
    )
    parser.add_argument("--batch-size", default=1, type=int, help="Set batch size")
    parser.add_argument(
        "--num-frames",
        default=100,
        type=int,
        help="Number of frames to process for each iteration",
    )
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations")
    parser.add_argument(
        "--inference-only",
        action=argparse.BooleanOptionalAction,
        help="Does not include decode/preprocess in timing loop; no output processing",
    )
    parser.add_argument(
        "--process-outputs",
        action=argparse.BooleanOptionalAction,
        help="does additional processing like watermarks or image-generation; mutex with --inference-only",
    )
    parser.add_argument(
        "--warmup-iterations",
        default=10,
        type=int,
        help="number of FRAMES to run inference on before recording metrics",
    )
    parser.add_argument(
        "--cpu-decode",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Force CPU decode, even on XPU/CUDA",
    )


def get_inference_args(args):
    return InferenceArgs(
        num_frames=args.num_frames,
        iterations=args.iterations,
        batch_size=args.batch_size,
        inference_only=args.inference_only,
        process_outputs=args.process_outputs,
        warmup_iterations=args.warmup_iterations,
        cpu_decode=args.cpu_decode,
    )
