from multiprocessing import Process, Queue
import logging
import importlib

# Flag for PyTorch availability
torch_available = True
try:
    import torch
    from PIL import Image
except ImportError:
    torch_available = False


class VideoWriter:
    def __init__(self, device, output_path, encode_fps, encode_bitrate, **kwargs):
        self.writer = self._create(device, output_path, encode_fps, encode_bitrate, **kwargs)

    def write(self, frame_tensor):
        self.writer.write(frame_tensor)

    def close(self):
        self.writer.close()

    @staticmethod
    def _create(device, output_path, encode_fps, encode_bitrate, **kwargs):
        if device in ["cpu"]:
            return CpuVideoWriter(output_path, encode_fps, encode_bitrate, **kwargs)
        elif device == "vaapi":
            return VaapiVideoWriter(output_path, encode_fps, encode_bitrate, **kwargs)
        else:
            raise ValueError(f"Unsupported device type: {device}")

    def tensor_to_image(self, tensor):
        """
        Convert a PyTorch tensor to a PIL Image.

        Args:
        tensor (torch.Tensor): The input tensor to convert.

        Returns:
        PIL.Image: The converted PIL Image.
        """
        if torch_available:
            # PyTorch-specific conversion
            # Move tensor to CPU and convert to uint8 if necessary
            tensor = tensor.cpu()
            if tensor.dtype == torch.float16:
                tensor = (tensor.to(torch.float32) * 255).type(torch.uint8)
            elif tensor.dtype == torch.float32:
                tensor = (tensor * 255).type(torch.uint8)

            # Convert to NumPy array and transpose to (H, W, C) format
            np_array = tensor.numpy().transpose(1, 2, 0)

            # Convert to PIL Image
            return Image.fromarray(np_array)

        else:
            # Placeholder for OV-specific conversion
            raise NotImplementedError("Conversion from OV tensor to PIL Image is not implemented")


class CpuVideoWriter(VideoWriter):
    codec_settings = {
        "libaom-av1": {"pix_fmt": "yuv420p"},
        "libx264rgb": {"pix_fmt": "rgb24"},
        "libx264": {"pix_fmt": "yuv420p"},
        "libx265": {"pix_fmt": "yuv420p"},
    }

    def __init__(self, output_path, encode_fps, encode_bitrate, **kwargs):
        import av

        self.av = av
        self.output_path = output_path
        self.encode_fps = encode_fps
        self.encode_bitrate = encode_bitrate
        self.codec = kwargs.get("codec", "libx264rgb")
        self.pix_fmt = self.codec_settings[self.codec]["pix_fmt"]
        self.queue = Queue()
        self.process = Process(target=self._write_frames)
        self.process.start()

    def _write_frames(self):
        video_writer = None
        try:
            video_writer = self.av.open(self.output_path, mode="w")
            stream = video_writer.add_stream(self.codec, rate=self.encode_fps)
            stream.bit_rate = self.encode_bitrate
            stream.pix_fmt = self.pix_fmt

            while True:
                image = self.queue.get()  # Blocking call until an item is available

                if image is None:  # Sentinel value for shutdown
                    break

                av_frame = self.av.VideoFrame.from_image(image)
                if self.pix_fmt != "rgb24":
                    av_frame = av_frame.reformat(format=self.pix_fmt)

                for packet in stream.encode(av_frame):
                    video_writer.mux(packet)

            for packet in stream.encode():  # Flush remaining frames
                video_writer.mux(packet)

        except Exception as e:
            logging.error(f"Error occurred in _write_frames: {e}")

        finally:
            if video_writer:
                video_writer.close()
            logging.info("Video writing process completed.")

    def write(self, image):
        """Write a PIL Image to the video stream."""
        self.queue.put(image)

    def close(self):
        self.queue.put(None)
        self.process.join()


class VaapiVideoWriter(VideoWriter):
    def __init__(self, output_path, encode_fps, encode_bitrate, **kwargs):
        from . import get_video_backend

        self.backend = get_video_backend()
        self.width = kwargs.get("width")
        self.height = kwargs.get("height")
        self.codec = kwargs.get("codec", "h264_vaapi")

        # Import the necessary internal modules
        self.libvideowriter = importlib.import_module(".libvisual_ai", package=__package__)

        # Setup the video info
        video_info = self.libvideowriter.videowriter.VideoInfo()
        video_info.codec_name = self.codec
        video_info.width = self.width
        video_info.height = self.height
        video_info.bitrate = encode_bitrate
        if isinstance(encode_fps, tuple):
            video_info.framerate.num = encode_fps[0]
            video_info.framerate.den = encode_fps[1]
        else:
            video_info.framerate.num = encode_fps
            video_info.framerate.den = 1

        # Create the encoder
        if "xpu" in self.backend:
            # FIXME: XpuEncoder and XpuDecoder if we are using the same device the string MUST BE the same!
            # Tentatively it is fixed using the self.backend which equates to "xpu" but if there are more
            # Devices this mechanism will create two different context! This affects ContextManager!
            self.ovideo = self.libvideowriter.videowriter.XpuEncoder(
                output_path, self.backend, video_info
            )
        else:
            raise RuntimeError("Unknown video backend: {}".format(self.backend))

    def write(self, vaFrame):
        # Check if vaFrame is a list
        if isinstance(vaFrame, list):
            # Iterate over each item in the list
            for frame in vaFrame:
                nv12_surface_id = frame.va_surface_id
                self.ovideo.write(nv12_surface_id)
        else:
            # Existing logic for a single vaFrame
            nv12_surface_id = vaFrame.va_surface_id
            self.ovideo.write(nv12_surface_id)

    def close(self):
        self.ovideo.close()
