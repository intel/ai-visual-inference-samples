import functools
import time
import logging
import collections
from datetime import datetime
import os

MILLISECONDS = 1000

# Only one instance maintains latency values for all functions in the process


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Chose a class to maintain information about logger and latency across frames
class Metrics(metaclass=Singleton):
    def __init__(self):
        self.raw_data = collections.defaultdict(list)

    def set_logger(self, name, logger_type=["console", "file"], log_level=logging.INFO):
        """Sample logging function"""

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter("%(message)s")

        # Avoid duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if "file" in logger_type:
            # Note: This file needs to be in append mode or we overwrite it between function calls
            fh = logging.FileHandler(f"latency.log")
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if "console" in logger_type:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)

    def calculate_metrics(self, name="all", frame_count=0):
        """Calculates total latency and also reports per frame optionally"""

        # Returns total_latency,avg_latency,fps for all functions
        if name == "all":
            total_latency = sum([sum(latency_list) for latency_list in self.raw_data.values()])
            self.logger.info(f"Total latency : {total_latency:.4f} ms")
            if frame_count == 0:
                # When number of frames are not uniform across all functions
                avg_latency = sum(
                    [
                        (sum(latency_list) / len(latency_list))
                        for latency_list in self.raw_data.values()
                    ]
                )
                avg_fps = 1 / (avg_latency / MILLISECONDS)
            else:
                avg_latency = total_latency / frame_count
                total_latency_s = total_latency / MILLISECONDS
                avg_fps = frame_count / total_latency_s
                self.logger.info(f"Number of frames : {frame_count}")
            self.logger.info(f"Average latency per frame : {avg_latency:.4f} ms")
            self.logger.info(f"Throughput : {avg_fps:.4f} fps")
            return total_latency, avg_latency, avg_fps

        function_latency = 0
        avg_function_latency = 0
        function_fps = 0
        # Returns Returns total_latency,avg_latency,fps for specific function
        for func_name, latency_list in self.raw_data.items():
            if func_name == name:
                if frame_count == 0:
                    frame_count = len(latency_list)
                function_latency += sum(latency_list)
                self.logger.info(f"{func_name!r} latency : {function_latency:.4f} ms")
                avg_function_latency = function_latency / frame_count
                function_latency_s = function_latency / MILLISECONDS
                function_fps = frame_count / function_latency_s
                self.logger.info(f"{func_name!r} Number of frames : {frame_count}")
                self.logger.info(f"{func_name!r} Per frame latency : {avg_function_latency:.4f} ms")
                self.logger.info(f"{func_name!r} Throughput : {function_fps:.4f} fps")
                break
        return function_latency, avg_function_latency, function_fps

    def calculate_total_metrics(self, start, end, frame_count):
        """Calculates total metrics given start end and number of frames"""
        duration_s = end - start
        duration_ms = duration_s * MILLISECONDS
        avg_latency = duration_ms / frame_count
        avg_fps = frame_count / duration_s
        self.logger.info(f"Total latency : {duration_ms:.4f} ms")
        self.logger.info(f"Number of frames : {frame_count}")
        self.logger.info(f"Average latency per frame : {avg_latency:.4f} ms")
        self.logger.info(f"Throughput : {avg_fps:.4f} fps")
        return duration_ms, avg_latency, avg_fps

    def reset_timer(self):
        """Clears all latency data"""

        self.raw_data = collections.defaultdict(list)


def latency_timer(func):
    """Logs the runtime of the decorated function"""

    m = Metrics()
    m.set_logger(name="latency", logger_type=["file"])
    if hasattr(func, "__wrapper__"):
        return func

    @functools.wraps(func)
    def wrapper_latency_timer(*args, **kwargs):
        start_time = time.perf_counter() * MILLISECONDS
        retval = func(*args, **kwargs)
        end_time = time.perf_counter() * MILLISECONDS
        run_time = end_time - start_time
        # Can enable logging to see per frame/function values
        m.logger.debug(f" {func.__name__!r} latency : {run_time:.4f} ms")
        m.raw_data[func.__name__].append(run_time)
        return retval

    return wrapper_latency_timer


class MetricsTimer:
    def __init__(self, name, logger_type=["file"], log_level=logging.INFO, log_dir=None, device=""):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter("%(message)s")

        # Avoid duplicate logging
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        if "file" in logger_type:
            # Note: This file needs to be in append mode or we overwrite it between function calls
            timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            file_path = f"{name}_latency_{device}_{timestr}.log"
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                file_path = os.path.join(log_dir, file_path)
            fh = logging.FileHandler(file_path)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if "console" in logger_type:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)
        self.reset_timer()

    def start_timer(self):
        self._start = time.perf_counter()

    def end_timer(self):
        self._end = time.perf_counter()

    def cal_current_itr_metrics(self, frame_count, itr, batch_size=None):
        """Calculates total metrics given start end and number of frames"""
        duration_s = self._end - self._start
        duration_ms = duration_s * MILLISECONDS
        avg_latency = duration_ms / frame_count
        avg_fps = frame_count / duration_s
        self.logger.info(f"Iteration: {itr}")
        if batch_size:
            self.logger.info(f"Batch_size: {batch_size}")
        self.logger.info(f"Total latency : {duration_ms:.4f} ms")
        self.logger.info(f"Number of frames : {frame_count}")
        self.logger.info(f"Per frame Latency : {avg_latency:.4f} ms")
        self.logger.info(f"Throughput : {avg_fps:.4f} fps")
        return duration_ms, avg_latency, avg_fps

    def cal_all_itr_metrics(self, itr, total_fps, avg_latency):
        avg_fps = total_fps / itr
        avg_latency_per_frame = avg_latency / itr
        self.logger.info(f"Average throughput: {avg_fps:.4f} fps")
        self.logger.info(f"Average Per frame Latency: {avg_latency_per_frame:.4f} ms")

    def reset_timer(self):
        self._start = None
        self._end = None
