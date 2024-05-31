from PIL import Image, ImageDraw
from pathlib import Path
import torch
from torchvision.transforms.v2 import Resize, CenterCrop, Normalize
import os
import csv
from datetime import datetime
from samples.common.get_dirs import labels_dir

from samples.pytorch.utils.pt_pipeline import PtPipeline
from samples.pytorch.utils.pt_arguments import PtArguments

LABELS_PATH = labels_dir / "imagenet_2012.txt"
MODEL_WIDTH = 224
MODEL_HEIGHT = 224


class ImageNetArguments(PtArguments):
    def __init__(
        self,
        sample_name,
        batch_size,
        output_dir,
        decode_device=None,
        labels_path=LABELS_PATH,
        num_frames=None,
    ):
        super().__init__(
            sample_name=sample_name,
            batch_size=batch_size,
            output_dir=output_dir,
            labels_path=labels_path,
            decode_device=decode_device,
            num_frames=num_frames,
        )


class ImageNetPipeline(PtPipeline):

    def __init__(
        self,
        model,
        args,
        logger=None,
    ):
        super().__init__(model, args, MODEL_WIDTH, MODEL_HEIGHT, logger)

    def set_tranform(self):
        transform_layers = []
        if self._resize_with_torch:
            transform_layers.append(Resize(256, antialias=None))
            transform_layers.append(CenterCrop(self._model_width))

        if self.args.normalize_inputs:
            transform_layers.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        if transform_layers:
            self.transforms = torch.nn.Sequential(*transform_layers)

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

    def process_outputs(self, frame_ids, decoded_tensors, outputs):
        for i, decoded_tensor in enumerate(decoded_tensors):
            label_id = torch.argmax(outputs[i])
            frame_id = frame_ids[i]
            label = " ".join(self.labels[label_id])
            text = f"Predicted Index: {label_id}, Class: {label}"
            if self.args.log_predictions:
                self.logger.info(f"Frame {frame_id} {text}")
            if self.args.watermark:
                self.watermark(decoded_tensor, text, frame_id)

    def store_output_to_csv(self, outputs):
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        csv_file_path = Path(self.args.output_dir) / f"inference_result_{timestr}.csv"
        print(f"sotring results to file: {csv_file_path}")
        results = []
        for i, output in enumerate(outputs):
            for j in range(self.batch_size):
                top3_indices = torch.topk(output[j], k=3).indices
                results.append(
                    {
                        f"predicted_class_id_{idx}": int(val.cpu())
                        for idx, val in enumerate(top3_indices)
                    }
                )
        with open(csv_file_path, "w") as f:
            writer = csv.DictWriter(f, results[0].keys())
            writer.writeheader()
            writer.writerows(results)
