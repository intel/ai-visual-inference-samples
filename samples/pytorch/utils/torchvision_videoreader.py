import torchvision


class TorchvisionVideoReaderWithLoopMode(torchvision.io.VideoReader):
    def __init__(self, *args, loop_mode=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_mode = loop_mode

    def __next__(self):
        try:
            return super().__next__()["data"]
        except StopIteration as e:
            if self.loop_mode:
                self.seek(0)
                return super().__next__()["data"]
            else:
                raise e
