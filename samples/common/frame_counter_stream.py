from intel_visual_ai.stream import StreamMeta


class FrameCounterStream(StreamMeta):
    def __init__(self, batch_size=1):
        self._frames_processed = 0
        self.__batch_size = batch_size

    @property
    def frames_processed(self):
        self._frames_processed += self.__batch_size
        return self._frames_processed

    def __next__(self):
        pass

    @property
    def finished(self):
        pass
