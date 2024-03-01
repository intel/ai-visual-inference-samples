import argparse
from PIL import Image, ImageDraw
from samples.pytorch.utils.metrics import MetricsTimer
import torch
from torchvision.transforms.v2 import Resize, CenterCrop
import intel_visual_ai
import torchvision
import os
import json

from samples.pytorch.utils.get_dirs import media_dir, data_dir

default_media_file = os.path.abspath(
    os.path.join(media_dir, "20230104_dog_bark_1920x1080_3mbps_30fps_ld_h265.mp4")
)
labels_file_path = os.path.abspath(os.path.join(data_dir, "imagenet_class_index.json"))


def add_arguments(
    parser, default_media_path=default_media_file, default_batch_size=4, default_num_frames=40000
):
    parser.add_argument("--input", default=default_media_path, help="Path to input media")
    parser.add_argument(
        "--watermark",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Performs watermark and includes it for metrics calculation",
    )
    parser.add_argument("--batch-size", default=default_batch_size, type=int, help="Set batch size")
    parser.add_argument(
        "--frames-per-iteration",
        dest="frames",
        default=default_num_frames,
        type=int,
        help="Number of frames to process for each iteration",
    )
    parser.add_argument("--iterations", default=1, type=int, help="Number of iterations")
    parser.add_argument(
        "--warmup-iterations", default=4, type=int, help="Number of iterations to run warmup"
    )
    parser.add_argument(
        "--log-labels",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Logs Detected output labels or label ids",
    )
    parser.add_argument(
        "--disable-jit-optimization",
        dest="disable_jit",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Disables model Jit optimization",
    )
    parser.add_argument(
        "--inference-only",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Runs inference only pipeline and measures performance",
    )
    parser.add_argument(
        "--resize-in-decoder",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Combine resize and decoding steps",
    )
    parser.add_argument(
        "--only-download-models",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Option to Download models and exit",
    )
    parser.add_argument("--output-dir", default="output", help="Path to watermarked results")
    parser.add_argument("--device", default="xpu", help="What device to use for processing")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to dataset folder. If folder is empty will download default dataset",
    )
    return parser


class ImageNet2012Util:
    def __init__(
        self, model, media_path, args, model_name="", decode_device="", model_precision=""
    ):
        if decode_device == "cpu":
            self.decode = self.decode_cpu
            self.read_video = self.read_video_cpu
            self.preprocess = self.preprocess_cpu
        if "xpu" in args.device:
            intel_visual_ai.set_video_backend(args.device)
        if model_precision == "FP32":
            self.preprocess = self.preprocess_fp32
        self.model = model
        self.media_path = media_path
        self.model_name = model_name
        self.args = args
        self.batch_size = int(args.batch_size)
        transform = torch.nn.Sequential(
            Resize(256, antialias=None),
            CenterCrop(224),
        )
        self.transforms = torch.jit.script(transform)
        with open(labels_file_path) as labels_file:
            self.labels = json.load(labels_file)
        if self.args.watermark:
            os.makedirs(args.output_dir, exist_ok=True)

    def decode_cpu(self, input):
        #########################
        # 7 Decode Frames
        # ----------------------------------
        # Returns the next decode frame
        cpu_tensor = next(input)["data"]
        return cpu_tensor

    def decode(self, input):
        #########################
        # 7 Decode Frames
        # ----------------------------------
        # Returns the next decode frame
        return next(input)

    def preprocess_cpu(self, decoded_tensor):
        processed_tensor = decoded_tensor.float() / 255.0
        processed_tensor = self.transforms(processed_tensor).to("xpu")
        return processed_tensor

    def preprocess(self, decoded_tensor):
        processed_tensor = decoded_tensor.float().half() / 255.0
        if not self.args.resize_in_decoder:
            processed_tensor = self.transforms(processed_tensor)
        return processed_tensor

    def preprocess_fp32(self, decoded_tensor):
        processed_tensor = decoded_tensor.float() / 255.0
        if not self.args.resize_in_decoder:
            processed_tensor = self.transforms(processed_tensor)
        return processed_tensor

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
        #########################
        # Watermark
        # ----------------------------------
        # The decoded_tensor and the label provided by inference are sufficient to generate an output image with watermark
        # The operations are similar to those done for preprocessing, except we keep the tensor at 3 dimensions
        # Next, the operation of converting the tensor back to (H, W, C) is done on CPU in this example.
        # :func `torch.tensor.byte` is used to return tensor in UINT8 format.
        # Next a PIL image object is initialized and used for drawing.
        # A label text is provided for select label ids as a sample.
        # The generated PIL image is saved to disk with `frame_id` to create a unique file per frame.
        tensor = (decoded_tensor.cpu().permute(1, 2, 0)).byte()
        image_pil = Image.fromarray(tensor.numpy())
        draw = ImageDraw.Draw(image_pil)
        draw.text((10, 10), text, fill="red")
        image_name = os.path.join(self.args.output_dir, f"{self.model_name}_output_{frame_id}.png")
        image_pil.save(image_name)

    def process_outputs(self, frame_counter, decoded_tensors, outputs):
        for i, decoded_tensor in enumerate(decoded_tensors):
            label_id = torch.argmax(outputs[i])
            frame_id = frame_counter - self.batch_size + i
            label = " ".join(self.labels[f"{label_id}"])
            text = f"Predicted Index: {label_id}, Class: {label}"
            if self.args.log_labels:
                print(f"Frame {frame_id} {text}")
            if self.args.watermark:
                self.watermark(decoded_tensor, text, frame_id)

    def read_video(self):
        video_reader = intel_visual_ai.VideoReader(self.media_path)
        # The default memory format for videoReader is set as torch_contiguous_format
        # It can be changed to torch_channels_last by setting
        # video_reader._c.set_memory_format(intel_visual_ai.XpuMemoryFormat.torch_channels_last)
        if self.args.resize_in_decoder:
            video_reader._c.set_output_resolution(224, 224)
        video_reader._c.set_loop_mode(True)
        video_reader._c.set_frame_pool_params(self.batch_size * 2)
        video_reader._c.set_batch_size(self.batch_size)
        return video_reader

    def read_video_cpu(self):
        return intel_visual_ai.TorchvisionVideoReaderWithLoopMode(
            self.media_path, "video", loop_mode=True
        )

    def warmup(self):
        #########################
        # Warm Up reading few frames from video
        print("Warming up model")
        video = self.read_video()
        try:
            for _ in range(self.args.warmup_iterations):
                decoded_tensors = []
                for _ in range(self.batch_size):
                    decoded_tensor = self.decode(video)
                    decoded_tensors.append(decoded_tensor)
                batched_tensor = torch.stack(decoded_tensors)
                batched_tensor = self.preprocess(batched_tensor)
                self.inference(batched_tensor)
            torch.xpu.synchronize()
        except StopIteration:
            raise Exception(f"Not enough frames for warm up")
        print("End of warm up phase")

    def process_frames(self):
        if self.args.inference_only:
            self._run_inference_only_pipeline()
        else:
            self._run_full_pipeline()

    def _run_full_pipeline(self):
        m = MetricsTimer(self.model_name, log_dir=self.args.output_dir, device=self.args.device)
        total_fps = 0
        total_avg_latency = 0
        itr = 0
        for itr in range(self.args.iterations):
            print(f"Pipeline iteration #{itr+1}")
            frame_counter = 0
            video = self.read_video()
            m.start_timer()
            while frame_counter < self.args.frames:
                decoded_tensors = []
                for _ in range(self.batch_size):
                    decoded_tensor = None
                    try:
                        decoded_tensor = self.decode(video)
                    except StopIteration:
                        exception = f"Not enough frames to process {self.args.frames} frames"
                        print(f"\n{exception}")
                        raise Exception(exception)
                    decoded_tensors.append(decoded_tensor)
                frame_counter += self.batch_size
                batched_tensor = torch.stack(decoded_tensors)
                batched_tensor = self.preprocess(batched_tensor)
                outputs = self.inference(batched_tensor)
                if self.args.watermark or self.args.log_labels:
                    self.process_outputs(frame_counter, decoded_tensors, outputs)
            torch.xpu.synchronize()
            m.end_timer()
            _, curr_avg_latency, curr_fps = m.cal_current_itr_metrics(
                frame_counter, itr + 1, self.batch_size
            )
            total_fps += curr_fps
            total_avg_latency += curr_avg_latency
            m.reset_timer()
            m.cal_all_itr_metrics(itr + 1, total_fps, total_avg_latency)

    def _run_inference_only_pipeline(self):
        m = MetricsTimer(self.model_name, log_dir=self.args.output_dir, device=self.args.device)
        total_fps = 0
        total_avg_latency = 0
        video = self.read_video()
        processed_tensors = []
        decoded_tensor = self.decode(video)
        for _ in range(self.batch_size):
            processed_tensor = self.preprocess(decoded_tensor)
            processed_tensors.append(processed_tensor)
        batched_tensor = torch.stack(processed_tensors)
        for itr in range(self.args.iterations):
            print(f"Pipeline iteration #{itr+1}")
            frame_counter = 0
            m.start_timer()
            while frame_counter < self.args.frames:
                frame_counter += self.batch_size
                outputs = self.inference(batched_tensor)
                outputs = outputs.to("cpu")
            torch.xpu.synchronize()
            m.end_timer()
            _, curr_avg_latency, curr_fps = m.cal_current_itr_metrics(
                frame_counter, itr + 1, self.batch_size
            )
            total_fps += curr_fps
            total_avg_latency += curr_avg_latency
            m.reset_timer()
            m.cal_all_itr_metrics(itr + 1, total_fps, total_avg_latency)
