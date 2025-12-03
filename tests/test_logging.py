"""Tests for l0.logging module."""

import logging

from l0.logging import enable_debug, logger


class TestEnableDebug:
    def test_enable_debug_sets_level(self):
        """Test that enable_debug sets DEBUG level."""
        # Store original level
        original_level = logger.level

        try:
            enable_debug()
            assert logger.level == logging.DEBUG
        finally:
            # Restore original level
            logger.setLevel(original_level)

    def test_logger_name(self):
        """Test that logger has correct name."""
        assert logger.name == "l0"
