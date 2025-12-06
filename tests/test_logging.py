"""Tests for l0.logging module."""

import logging

from l0.logging import enable_debug, logger


class TestEnableDebug:
    def test_enable_debug_sets_level(self):
        """Test that enable_debug sets DEBUG level."""
        # Store original state
        original_level = logger.level
        original_handlers = logger.handlers[:]

        try:
            enable_debug()
            assert logger.level == logging.DEBUG
        finally:
            # Restore original state
            logger.setLevel(original_level)
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            for h in original_handlers:
                logger.addHandler(h)

    def test_logger_name(self):
        """Test that logger has correct name."""
        assert logger.name == "l0"

    def test_enable_debug_does_not_duplicate_handlers(self):
        """Test that repeated enable_debug calls don't add duplicate handlers."""
        # Store original state
        original_level = logger.level
        original_handlers = logger.handlers[:]

        # Remove any existing StreamHandlers
        for h in logger.handlers[:]:
            if isinstance(h, logging.StreamHandler):
                logger.removeHandler(h)

        try:
            enable_debug()
            enable_debug()
            enable_debug()

            stream_handlers = [
                h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            ]
            assert len(stream_handlers) == 1
        finally:
            # Restore original level
            logger.setLevel(original_level)
            # Restore original handlers
            for h in logger.handlers[:]:
                if isinstance(h, logging.StreamHandler):
                    logger.removeHandler(h)
            for h in original_handlers:
                if h not in logger.handlers:
                    logger.addHandler(h)
