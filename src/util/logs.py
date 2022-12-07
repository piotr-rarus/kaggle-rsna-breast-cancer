import logging
import sys
from logging import Formatter, StreamHandler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = _get_console_handler(level)
    logger.addHandler(console_handler)

    return logger


def _get_console_handler(level: int) -> logging.Handler:
    formatter = Formatter(
        fmt="""
        %(asctime)s loglevel=%(levelname)-6s %(funcName)s()
        %(message)s
        call_trace=%(pathname)s:%(lineno)-4d
        """
    )

    handler = StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler
