import csv
import openvino as ov
import numpy as np
import atexit
from datetime import datetime
from typing import List
from intel_visual_ai.frame import Frame
from samples.common.get_dirs import labels_dir
from samples.openvino.utils.ov_pipeline import OvPipeline

LABELS_PATH = labels_dir / "imagenet_2012.txt"


class ImageNetPipeline(OvPipeline):
    def configure_completion_callback(self, **kwargs):
        self.__results = {stream_id: {} for stream_id in range(self.video.num_streams)}
        atexit.register(self._write_results_to_file, self.__results)
        self._top_predictions = 3
        return super().configure_completion_callback(**kwargs)

    def _write_results_to_file(self, results):
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        with open(self._output_dir / f"inference_result_{timestr}.csv", "w") as fd:
            _inference_result_writer = csv.DictWriter(
                fd,
                [f"predicted_class_id_{i}" for i in range(self._top_predictions)],
            )
            _inference_result_writer.writeheader()
            for stream_id, stream_predictions in sorted(results.items()):
                for frame_id, frame_predictions in sorted(stream_predictions.items()):
                    self.__write_predictions(frame_predictions, _inference_result_writer)
            self.logger.info(f"Results written to {fd.name}")

    def __write_predictions(self, prediction, writer):
        writer.writerow(
            {
                f"predicted_class_id_{i}": val[0]
                for i, val in enumerate(prediction[: self._top_predictions])
            }
        )

    def completion_callback(self, infer_request: ov.InferRequest, frames: List[Frame]) -> None:
        predictions = list(infer_request.results.values())
        if len(predictions) != 1:
            raise f"Unexpected output layout: {len(predictions)}"
        predictions = predictions[0]
        for i, prediction in enumerate(predictions):
            # calculate softmax(predictions, axis=0)
            b = np.exp(prediction - np.max(prediction)).sum(axis=0)
            prediction = np.exp(prediction - np.max(prediction)) / np.expand_dims(b, axis=0)

            label_ids = list(reversed(sorted(enumerate(prediction), key=lambda x: x[1])))
            self.__results[frames[i].stream_id][frames[i].frame_id] = label_ids[
                : self._top_predictions
            ]
            label_id, confidence = label_ids[0]
            if confidence < self._threshold:
                continue
            self.logger.info(
                f"stream_id={frames[i].stream_id} "
                f"frame_num={frames[i].frame_id} "
                f"label_id={label_id} "
                f"label={self.labels[label_id]} "
                f"confidence={confidence:.4f}"
            )
