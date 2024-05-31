import numpy as np
import torch
import json
import atexit
from typing import Tuple, List
from ultralytics.utils import ops
from intel_visual_ai.frame import Frame
from intel_visual_ai.openvino_infer_backend import ov
from samples.openvino.utils.ov_pipeline import OvPipeline
from samples.models.yolo import scale_output_boxes


class YoloPipeline(OvPipeline):

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
        pred (List[torch.Tensor]): list of detected boxes in format [x1, y1, x2, y2, score, label]
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
            for *xyxy, conf, lbl in result.numpy():
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
