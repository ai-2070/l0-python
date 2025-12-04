"""Tests for l0.retry module."""

import pytest

from src.l0.retry import RetryManager
from src.l0.types import BackoffStrategy, ErrorCategory, Retry


class TestRetryManager:
    def test_default_config(self):
        mgr = RetryManager()
        assert mgr.config.attempts == 3
        assert mgr.model_retry_count == 0
        assert mgr.network_retry_count == 0

    def test_should_retry_network_error(self):
        """Network errors should always retry."""
        mgr = RetryManager()
        error = Exception("Connection reset")

        assert mgr.should_retry(error) is True

        # Even after many attempts
        for _ in range(5):
            mgr.record_attempt(error)
        assert mgr.should_retry(error) is True

    def test_should_retry_model_error_limited(self):
        """Model errors should respect attempt limit."""
        mgr = RetryManager(Retry(attempts=2))
        error = Exception("Model error")

        assert mgr.should_retry(error) is True
        mgr.record_attempt(error)
        assert mgr.should_retry(error) is True
        mgr.record_attempt(error)
        assert mgr.should_retry(error) is False

    def test_should_not_retry_fatal(self):
        """Fatal errors should never retry."""
        mgr = RetryManager()

        class FatalError(Exception):
            status_code = 401

        assert mgr.should_retry(FatalError()) is False

    def test_max_retries_absolute_limit(self):
        """Total retries should not exceed max_retries."""
        mgr = RetryManager(Retry(max_retries=3))
        error = Exception("Connection reset")

        for _ in range(3):
            mgr.record_attempt(error)

        assert mgr.should_retry(error) is False

    def test_record_attempt_increments_counters(self):
        mgr = RetryManager()

        # Network error
        mgr.record_attempt(Exception("Connection reset"))
        assert mgr.network_retry_count == 1
        assert mgr.model_retry_count == 0

        # Model error
        mgr.record_attempt(Exception("Model failed"))
        assert mgr.network_retry_count == 1
        assert mgr.model_retry_count == 1

    def test_get_delay_exponential(self):
        mgr = RetryManager(
            Retry(
                base_delay=1.0,  # seconds
                max_delay=10.0,
                strategy=BackoffStrategy.EXPONENTIAL,
            )
        )
        error = Exception("Error")

        # First attempt: 1.0s
        delay1 = mgr.get_delay(error)
        assert delay1 == 1.0

        mgr.record_attempt(error)
        # Second attempt: 2.0s
        delay2 = mgr.get_delay(error)
        assert delay2 == 2.0

    def test_get_delay_linear(self):
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                max_delay=10.0,
                strategy=BackoffStrategy.LINEAR,
            )
        )
        error = Exception("Error")

        delay1 = mgr.get_delay(error)
        assert delay1 == 1.0

        mgr.record_attempt(error)
        delay2 = mgr.get_delay(error)
        assert delay2 == 2.0

    def test_get_delay_fixed(self):
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                strategy=BackoffStrategy.FIXED,
            )
        )
        error = Exception("Error")

        delay1 = mgr.get_delay(error)
        mgr.record_attempt(error)
        delay2 = mgr.get_delay(error)

        assert delay1 == 1.0
        assert delay2 == 1.0

    def test_get_delay_capped_at_max(self):
        mgr = RetryManager(
            Retry(
                base_delay=5.0,
                max_delay=8.0,
                strategy=BackoffStrategy.EXPONENTIAL,
            )
        )
        error = Exception("Error")

        # After many retries, should cap at max
        for _ in range(10):
            mgr.record_attempt(error)

        delay = mgr.get_delay(error)
        assert delay <= 8.0

    def test_reset(self):
        mgr = RetryManager()
        mgr.record_attempt(Exception("Error"))
        mgr.record_attempt(Exception("Connection reset"))

        mgr.reset()

        assert mgr.model_retry_count == 0
        assert mgr.network_retry_count == 0
        assert mgr.total_retries == 0

    def test_get_state(self):
        mgr = RetryManager()
        mgr.record_attempt(Exception("Error"))

        state = mgr.get_state()

        assert state["model_retry_count"] == 1
        assert state["network_retry_count"] == 0
        assert state["total_retries"] == 1

    def test_transient_error_backoff_increases(self):
        """Transient errors (e.g., 429) should use network_retry_count for backoff."""
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                max_delay=10.0,
                strategy=BackoffStrategy.EXPONENTIAL,
            )
        )

        class RateLimitError(Exception):
            status_code = 429

        error = RateLimitError("rate limited")

        # First attempt: 1.0s (attempt=0)
        delay1 = mgr.get_delay(error)
        assert delay1 == 1.0

        mgr.record_attempt(error)
        # Second attempt: 2.0s (attempt=1)
        delay2 = mgr.get_delay(error)
        assert delay2 == 2.0

        mgr.record_attempt(error)
        # Third attempt: 4.0s (attempt=2)
        delay3 = mgr.get_delay(error)
        assert delay3 == 4.0

        # Verify it's using network_retry_count, not model_retry_count
        assert mgr.network_retry_count == 2
        assert mgr.model_retry_count == 0
