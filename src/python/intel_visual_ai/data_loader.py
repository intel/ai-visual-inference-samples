import intel_visual_ai
import logging

torch_available = True
try:
    import torch
except ImportError:
    torch_available = False


class MediaDataLoader:
    def __init__(
        self,
        video_path,
        batch_size=1,
        device="xpu",
        output_resolution=None,
        loop_mode=False,
        preprocess_func=None,
        transform=None,
        output_original_nv12=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.video_path = video_path
        self.output_resolution = output_resolution
        self.loop_mode = loop_mode
        self.output_original_nv12 = output_original_nv12
        self.video_reader = self._create_video_reader()
        self.preprocess_func = preprocess_func
        self.transform = transform
        self.device = device
        self.logger.info(f"MediaDataLoader initialized for {video_path}")

    def _create_video_reader(self):
        intel_visual_ai.set_video_backend_params(
            out_img_size=self.output_resolution,
            loop_mode=self.loop_mode,
            batch_size=self.batch_size,
            pool_size=self.batch_size * 2,
            output_original_nv12=self.output_original_nv12,
        )
        video_reader = intel_visual_ai.VideoReader(self.video_path)
        return video_reader

    def __iter__(self):
        return self

    def __next__(self):
        if torch_available:
            return self._next_with_torch()
        else:
            return self._next_with_ov()

    def _next_with_torch(self):
        processed_tensors = []
        decoded_tensors = []  # List to store all intermediate tensors
        if self.output_original_nv12:
            va_frames = []

        try:
            for _ in range(self.batch_size):
                # TODO: If we change the behaviour of videoreader this must be updated
                # Right now we are returning two frames RGBP/RGBA and va_frame
                if self.output_original_nv12:
                    decoded_tensor, va_frame = next(self.video_reader)
                    va_frames.append(va_frame)
                else:
                    decoded_tensor = next(self.video_reader)
                decoded_tensors.append(decoded_tensor)  # Keep original decoded tensor

        except StopIteration:
            if not decoded_tensors:
                raise

        cloned_tensors = [decoded_tensor.clone().detach() for decoded_tensor in decoded_tensors]
        processed_tensor = torch.stack(cloned_tensors)

        if self.preprocess_func:
            processed_tensor = self.preprocess_func(processed_tensor)
            if self.check_tensor_for_zeros(processed_tensor):
                self.logger.warning("Processed tensor contains only zeros after preprocessing")

        if self.transform:
            processed_tensor = self.transform(processed_tensor)
        processed_tensor = processed_tensor.to(self.device)

        if self.output_original_nv12:
            return processed_tensor, va_frames
        else:
            return processed_tensor

    def _next_with_ov(self):
        # Placeholder for ov-specific implementation of __next__
        raise NotImplementedError("MediaDataLoader iteration for ov is not implemented")

    def reset(self):
        self.video_reader = self._create_video_reader()

    @staticmethod
    def check_tensor_for_zeros(tensor):
        if torch_available:
            # PyTorch-specific zero check
            return torch.all(tensor == 0).item()
        else:
            # Placeholder for ov-specific zero check
            raise NotImplementedError("Zero check for ov tensor is not implemented")
