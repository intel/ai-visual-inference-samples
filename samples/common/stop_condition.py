import time
import sys
import pathlib
import logging as log

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / "library" / "python"))
from intel_visual_ai.multi_stream_videoreader import MultiStreamVideoReader


class StopCondition:
    def __init__(self) -> None:
        self._stopped = False

    @property
    def stopped(self):
        return self._stopped


class DurationStopCondition(StopCondition):
    def __init__(self, duration) -> None:
        super().__init__()
        self.__duration = duration
        self.__start_time = None

    @property
    def stopped(self):
        if self.__start_time is None:
            self.__start_time = time.time()
        if time.time() - self.__start_time > self.__duration:
            self._stopped = True
        return self._stopped


class FramesProcessedStopCondition(StopCondition):
    def __init__(self, frames_to_process) -> None:
        super().__init__()
        self.__frames_to_process = frames_to_process
        self.__stream = None

    def set_stream(self, stream: MultiStreamVideoReader):
        self.__stream = stream

    @property
    def stopped(self):
        return self.__stream.frames_processed >= self.__frames_to_process


class EosStopCondition(StopCondition):
    def __init__(self) -> None:
        super().__init__()
        self.__stream = None

    def set_stream(self, stream: MultiStreamVideoReader):
        self.__stream = stream

    @property
    def stopped(self):
        if self.__stream.finished:
            self._stopped = True
        return self._stopped
