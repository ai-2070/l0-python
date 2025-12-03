import logging

logger = logging.getLogger("l0")
logger.addHandler(logging.NullHandler())


def enable_debug() -> None:
    """Enable debug logging for L0."""
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[l0] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
