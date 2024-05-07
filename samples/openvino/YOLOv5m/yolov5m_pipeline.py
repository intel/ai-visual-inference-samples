from typing import List
import numpy as np
import torch
from typing import Tuple
from torchvision.transforms import v2
from torchvision import tv_tensors
from ultralytics.utils import ops
from intel_visual_ai.frame import Frame
from intel_visual_ai.openvino_infer_backend import ov
from samples.openvino.utils.ov_pipeline import OvPipeline
import json
import os
import atexit


def scale_output_boxes(
    batch_output: List[torch.Tensor],
    scaled_tensor_size: torch.Size,
    original_tensor_size: torch.Size,
    format="XYXY",
) -> List[torch.Tensor]:
    """Scale ROI boxes received from yolov5 model to original frame size.
    As yolov5 detection was executed on rescaled frame.

    Args:
        batch_output (List[torch.Tensor]): list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        scaled_tensor_size (torch.Size): The size of scaled tensor that was passed to yolov5 model as input
        original_tensor_size (torch.Size): The size of original video frame tensor

    Returns:
        List[torch.Tensor]: _description_
    """
    transforms = v2.Resize(size=(original_tensor_size))
    scaled_output = []
    for output in batch_output:
        output[:, :4] = transforms(
            tv_tensors.BoundingBoxes(output[:, :4], format=format, canvas_size=scaled_tensor_size)
        )
        scaled_output.append(output.numpy())
    return scaled_output


class YOLOv5mPipeline(OvPipeline):

    def configure_completion_callback(self, iou_threshold=0.45, **kwargs):
        if self._print_predictions:
            self.iou_threshold = iou_threshold
            self.detections = []
            atexit.register(self.save_detections_to_json, self.detections)
        super().configure_completion_callback(**kwargs)

    def postprocess(
        self,
        pred_boxes: np.ndarray,
        input_hw: Tuple[int, int],
        orig_img_hw: Tuple[int, int],
        min_conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        agnosting_nms: bool = False,
        max_detections: int = 300,
    ):
        """
        YOLO model postprocessing function. Applies non maximum suppression algorithm (NMS) to detections and rescale boxes to original image size
        Parameters:
            pred_boxes (np.ndarray): model output prediction boxes
            input_hw (np.ndarray): preprocessed image
            orig_image (np.ndarray): image before preprocessing
            min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
            nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            agnostic_nms (bool, *optional*, False): apply class agnostic NMS approach or not
            max_detections (int, *optional*, 300):  maximum detections after NMS
        Returns:
        pred (List[np.ndarray]): list of detected boxes in format [x1, y1, x2, y2, score, label]
        """
        nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
        preds = ops.non_max_suppression(
            torch.from_numpy(pred_boxes), min_conf_threshold, nms_iou_threshold, nc=80, **nms_kwargs
        )
        results = scale_output_boxes(preds, input_hw, orig_img_hw)
        return results

    def completion_callback(self, infer_request: ov.InferRequest, frames: List[Frame]) -> None:
        predictions = tuple(infer_request.results.values())
        results = self.postprocess(
            pred_boxes=predictions[0],
            input_hw=(self._model_info.height, self._model_info.width),
            orig_img_hw=(self.video.original_size.height, self.video.original_size.width),
            min_conf_threshold=self._threshold,
            nms_iou_threshold=self.iou_threshold,
        )
        for i, result in enumerate(results):
            for *xyxy, conf, lbl in result:
                detection = {
                    "stream_id": frames[i].stream_id,
                    "frame_num": frames[i].frame_id,
                    "label_id": int(lbl),
                    "label": f"{self.labels[int(lbl)]}",
                    "confidence": f"{conf:.2f}",
                    "bbox": f"{xyxy}",
                }
                self.logger.info(json.dumps(detection))
                self.detections.append(detection)

    def save_detections_to_json(self, detections):
        """
        writes detections to file
        Parameters:
        - detections: Dict containing the detection to append.
        """
        filename = self._output_dir / "detections.json"
        # Write the updated list of detections back to the file
        with open(filename, "w") as file:
            json.dump(detections, file, indent=4)
