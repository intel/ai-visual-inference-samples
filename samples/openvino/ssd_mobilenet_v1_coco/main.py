from pathlib import Path
import cProfile
import sys
import json
import atexit
from typing import List
from intel_visual_ai.frame import Frame
from intel_visual_ai.openvino_infer_backend import ov

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.common.get_dirs import media_dir, models_dir, labels_dir
from samples.openvino.utils.ov_pipeline import OvPipeline
from samples.openvino.utils.ov_arguments import OvArguments
from samples.common.utils import get_stop_condition

MEDIA_PATH = media_dir / "20230104_dog_bark_1920x1080_3mbps_30fps_ld_h265.mp4"
MODEL_PRECISION = "int8"
LABELS_PATH = labels_dir / "coco_91cl_bkgr.txt"
NIREQ = 4
BATCH_SIZE = 64
DEVICE = "xpu"
INFERENCE_INTERVAL = 1
NUM_STREAMS = 2
THRESHOLD = 0.5
MODEL_NAME = "ssd_mobilenet_v1_coco"
SAMPLE_NAME = MODEL_NAME
NUM_FRAMES = 400_000


class SSDMobileNetPipeline(OvPipeline):
    def configure_completion_callback(self, **kwargs):
        if self._print_predictions:
            self.detections = []
            atexit.register(self.save_detections_to_json, self.detections)
        super().configure_completion_callback(**kwargs)

    def postprocess(self, predictions):
        # Filter by confidence
        pred_filter = predictions[..., 2] >= self._threshold
        predictions = predictions[pred_filter]

        target_w, target_h = self.video.original_size.width, self.video.original_size.height

        # Scale and clip:
        # x1, x2
        predictions[..., [3, 5]] = (predictions[..., [3, 5]] * target_w).clip(0, target_w)
        # y1, y2
        predictions[..., [4, 6]] = (predictions[..., [4, 6]] * target_h).clip(0, target_h)
        return predictions

    def completion_callback(self, infer_request: ov.InferRequest, frames: List[Frame]) -> None:
        predictions = list(infer_request.results.values())
        if len(predictions) != 1:
            raise "invalid output layout"

        predictions = self.postprocess(predictions[0])
        for img_idx, lbl, conf, *xyxy in predictions:
            img_idx = int(img_idx)
            lbl = int(lbl)
            detection = {
                "stream_id": frames[img_idx].stream_id,
                "frame_num": frames[img_idx].frame_id,
                "label_id": lbl,
                "label": f"{self.labels[lbl]}",
                "confidence": f"{conf:.2f}",
                "bbox": f"{xyxy}",
            }
            self.logger.info(detection)
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


def main():

    ov_args = OvArguments(
        sample_name=SAMPLE_NAME,
        media=MEDIA_PATH,
        model=MODEL_NAME,
        device=DEVICE,
        nireq=NIREQ,
        batch_size=BATCH_SIZE,
        inference_interval=INFERENCE_INTERVAL,
        streams=NUM_STREAMS,
        threshold=THRESHOLD,
        output_dir=Path(__file__).resolve().parent / "output",
        labels_path=LABELS_PATH,
        num_frames=NUM_FRAMES,
        precision=MODEL_PRECISION,
    )

    args = ov_args.parse_args()

    stop_condition = get_stop_condition(args)
    logger = ov_args.create_logger()

    pipeline = SSDMobileNetPipeline.create_from_args(args, stop_condition, logger)
    if args.only_download_models:
        logger.info("Model downloaded successfully, exiting")
        return
    if args.profile:
        cProfile.run("pipeline.run()", sort="cumtime")
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
