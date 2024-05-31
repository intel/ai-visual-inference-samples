import logging
import traceback
from intel_visual_ai.libvisual_ai import IttTask as cIttTask


class IttTask:
    def __init__(self, task_name, domain_name=None, logger=None) -> None:
        self.domain = cIttTask() if domain_name is None else cIttTask(domain_name)
        self.task_name = task_name
        self.logger = logger

    def start(self):
        self.domain.start(self.task_name)

    def stop(self):
        self.domain.end()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            if self.logger is None:
                self.logger = logging.getLogger("visual_ai")
            self.logger.error(
                f"Exception occurred in {self}: {traceback.print_exception(exc_type, exc_value, tb)}"
            )
        self.stop()


def itt_task(method):
    def wrapper(*args, **kwargs):
        # Create IttTask instance with the name of the decorated method
        with IttTask(task_name=method.__name__):
            result = method(*args, **kwargs)
            return result

    return wrapper
