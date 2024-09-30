import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: str = "train.log"):
    log_file = Path("logs") / log_file
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Log to console
    logger.add(log_file, rotation="10 MB", level="DEBUG")  # Log to file


def task_wrapper(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper
