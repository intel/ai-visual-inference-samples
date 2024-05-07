import os
from datetime import datetime
import logging
from pathlib import Path

DEFAULT_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")
BASE_LOGGER_NAME = "visual_ai"


def _add_file_handler(logger, file_path: Path, log_level, formatter):
    if not file_path.parent.exists():
        os.makedirs(file_path.parent, exist_ok=True)
    fh = logging.FileHandler(str(file_path))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def init_and_get_metrics_logger(sample_name, log_dir, device="", log_level=logging.INFO):
    logger = logging.getLogger(f"{BASE_LOGGER_NAME}.{sample_name}.metrics")
    logger.setLevel(log_level)
    timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    _add_file_handler(
        logger,
        Path(log_dir) / f"{sample_name}_latency_{device}_{timestr}.log",
        log_level=log_level,
        formatter=logging.Formatter("%(message)s"),
    )
    return logger


def init_and_get_logger(
    sample_name, logger_type=["file", "console"], log_level=logging.INFO, log_dir=None, device=""
):
    logger = logging.getLogger(f"{BASE_LOGGER_NAME}")
    # Disabling propagating to root logger because root logger might be configured elsewhere
    logger.propagate = False
    logger.setLevel(log_level)

    # Avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    if "file" in logger_type:
        # Note: This file needs to be in append mode or we overwrite it between function calls
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        _add_file_handler(
            logger,
            Path(log_dir) / f"{sample_name}_{device}_{timestr}_{os.getgid()}_full.log",
            log_level=log_level,
            formatter=DEFAULT_FORMATTER,
        )
    if "console" in logger_type:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(DEFAULT_FORMATTER)
        logger.addHandler(ch)
    sample_logger = logging.getLogger(f"{BASE_LOGGER_NAME}.{sample_name}")
    return sample_logger
