"""Structured JSON logging configuration."""

import logging
import sys

from pythonjsonlogger import jsonlogger


def setup_logger():
    """Configure structured JSON logging."""
    logger = logging.getLogger("ollama_deep_researcher")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()
