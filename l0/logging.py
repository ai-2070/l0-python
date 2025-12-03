import logging

logger = logging.getLogger("l0")
logger.addHandler(logging.NullHandler())


def enable_debug() -> None:
    """Enable debug logging for L0."""
    logger.setLevel(logging.DEBUG)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[l0] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
