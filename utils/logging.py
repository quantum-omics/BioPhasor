"""
biophasor.utils.logging — Library logging configuration.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

import logging


def get_logger(name: str = "biophasor", level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
