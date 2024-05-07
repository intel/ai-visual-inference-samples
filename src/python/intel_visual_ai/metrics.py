import os
import logging
import time
from intel_visual_ai.stream import Stream, StreamMeta, Resolution
from threading import Thread, Event
from pathlib import Path
from datetime import datetime, timedelta

MILLISECONDS = 1000


class MetricsTimer:
    def __init__(
        self,
        stream: Stream,
        batch_size=1,
        logger: logging.Logger | None = None,
        print_fps_to_stdout=True,
        output_dir: Path | None = None,
    ):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.__stream: Stream = stream
        self.__batch_size = batch_size
        self.__previous_frames_count = 0
        self.__previous_timestamp = 0
        self.__stopped = Event()
        self.__thread = None
        self.__stdout = print_fps_to_stdout
        self.__fps_output_file_handler = None
        self._start = None
        self._end = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fps_output_file_name = f"fps_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.csv"
            self.__fps_output_file_handler = open(Path(output_dir) / fps_output_file_name, "w")

    def __fps_printer_thread(self):
        while not self.__stopped.wait(1):
            curr_time = time.perf_counter()
            fps = (self.__stream.frames_processed - self.__previous_frames_count) / (
                curr_time - self.__previous_timestamp
            )
            self.__previous_frames_count = self.__stream.frames_processed
            self.__previous_timestamp = curr_time
            if self.__stdout:
                self.logger.info(f"FPS: {fps:.2f}")
            if self.__fps_output_file_handler:
                self.__now += timedelta(seconds=1)
                self.__fps_output_file_handler.write(
                    f"{self.__now.strftime('%H:%M:%S')},{fps:.2f}\n"
                )

    def __start_timer(self):
        self._start = time.perf_counter()
        self.__previous_timestamp = time.perf_counter()
        self.__now = datetime.now()

    def __end_timer(self):
        self._end = time.perf_counter()

    def start(self):
        if self.__stdout or self.__fps_output_file_handler is not None:
            self.__thread = Thread(target=self.__fps_printer_thread)
            self.__thread.start()
        self.__start_timer()

    def stop(self):
        self.__end_timer()
        self.__stopped.set()
        if self.__thread:
            self.__thread.join()
        self.cal_current_itr_metrics()
        if self.__fps_output_file_handler:
            self.__fps_output_file_handler.close()

    def duration(self):
        return time.perf_counter() - self._start

    def __enter__(self):
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def cal_current_itr_metrics(self):
        """Calculates total metrics given start end and number of frames"""
        frame_count = self.__stream.frames_processed
        duration_s = self._end - self._start
        duration_ms = duration_s * MILLISECONDS
        avg_latency = duration_ms / frame_count
        avg_fps = frame_count / duration_s
        self.logger.info(f"Batch_size: {self.__batch_size}")
        self.logger.info(f"Total latency : {duration_ms:.4f} ms")
        self.logger.info(f"Number of frames : {frame_count}")
        self.logger.info(f"Per frame Latency : {avg_latency:.4f} ms")
        self.logger.info(f"Throughput : {avg_fps:.4f} fps")

        return duration_ms, avg_latency, avg_fps
