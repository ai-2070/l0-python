"""Tests for l0.retry module."""

from typing import Any

import pytest

from l0.retry import RetryContext, RetryManager, _error_to_retryable_type
from l0.types import (
    BackoffStrategy,
    ErrorCategory,
    ErrorTypeDelays,
    Retry,
    RetryableErrorType,
    State,
)


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

    def test_get_delay_fixed_jitter(self):
        """Fixed jitter should add randomness to exponential base."""
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                max_delay=10.0,
                strategy=BackoffStrategy.FIXED_JITTER,
            )
        )
        error = Exception("Error")

        # Fixed jitter is: temp/2 + random * (temp/2)
        # So delay should be between 0.5 and 1.0 for first attempt
        delays = [mgr.get_delay(error) for _ in range(10)]
        assert all(0.5 <= d <= 1.0 for d in delays)

    def test_get_delay_full_jitter(self):
        """Full jitter should add more randomness."""
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                max_delay=10.0,
                strategy=BackoffStrategy.FULL_JITTER,
            )
        )
        error = Exception("Error")

        # Full jitter is: random * min(base * 2^attempt, cap)
        # So delay should be between 0 and 1.0 for first attempt
        delays = [mgr.get_delay(error) for _ in range(10)]
        assert all(0 <= d <= 1.0 for d in delays)

    def test_get_error_history(self):
        """Error history should track all recorded errors."""
        mgr = RetryManager()
        e1 = Exception("Error 1")
        e2 = Exception("Error 2")

        mgr.record_attempt(e1)
        mgr.record_attempt(e2)

        history = mgr.get_error_history()
        assert len(history) == 2
        assert e1 in history
        assert e2 in history
        # Should be a copy
        history.clear()
        assert len(mgr.get_error_history()) == 2

    def test_retry_on_filter(self):
        """retry_on should filter which error types can retry."""
        # Only allow network errors and timeouts
        mgr = RetryManager(
            Retry(
                retry_on=[
                    RetryableErrorType.NETWORK_ERROR,
                    RetryableErrorType.TIMEOUT,
                ]
            )
        )

        # Network error should be allowed
        network_error = Exception("Connection reset")
        assert mgr.should_retry(network_error) is True

        # Guardrail violation should be blocked
        class ContentError(Exception):
            pass

        # Create an error that gets categorized as CONTENT
        content_err = ContentError("guardrail violation detected")
        # Force error to be categorized as content
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.CONTENT
            assert mgr.should_retry(content_err) is False

    def test_retry_on_empty_list_blocks_all(self):
        """Empty retry_on list should block all retries."""
        mgr = RetryManager(Retry(retry_on=[]))
        error = Exception("Any error")
        assert mgr.should_retry(error) is False

    @pytest.mark.asyncio
    async def test_should_retry_async_with_callback(self):
        """Async should_retry should respect custom callback."""
        callback_calls = []

        def custom_should_retry(
            error: Exception, state: State, attempt: int, category: ErrorCategory
        ) -> bool:
            callback_calls.append((error, attempt, category))
            return attempt < 2  # Only retry first 2 attempts

        mgr = RetryManager(Retry(should_retry=custom_should_retry))
        error = Exception("Error")

        # First call should allow retry
        assert await mgr.should_retry_async(error) is True
        mgr.record_attempt(error)

        # Second call should still allow
        assert await mgr.should_retry_async(error) is True
        mgr.record_attempt(error)

        # Third call should deny (attempt >= 2)
        assert await mgr.should_retry_async(error) is False

        assert len(callback_calls) == 3

    @pytest.mark.asyncio
    async def test_should_retry_async_with_async_callback(self):
        """Async callbacks should be properly awaited."""

        async def async_should_retry(
            error: Exception, state: State, attempt: int, category: ErrorCategory
        ) -> bool:
            return attempt < 1

        mgr = RetryManager(Retry(should_retry=async_should_retry))
        error = Exception("Error")

        assert await mgr.should_retry_async(error) is True
        mgr.record_attempt(error)
        assert await mgr.should_retry_async(error) is False

    def test_custom_calculate_delay(self):
        """Custom calculate_delay callback should be used."""

        def custom_delay(ctx: RetryContext) -> float:
            # Return attempt * 5 seconds
            return ctx.attempt * 5.0

        mgr = RetryManager(Retry(calculate_delay=custom_delay))
        error = Exception("Error")

        assert mgr.get_delay(error) == 0.0  # attempt=0
        mgr.record_attempt(error)
        assert mgr.get_delay(error) == 5.0  # attempt=1
        mgr.record_attempt(error)
        assert mgr.get_delay(error) == 10.0  # attempt=2

    def test_error_type_delays_for_network_errors(self):
        """Per-error-type delays should be used for network errors."""
        mgr = RetryManager(
            Retry(
                base_delay=1.0,
                strategy=BackoffStrategy.FIXED,
                error_type_delays=ErrorTypeDelays(
                    connection_dropped=5.0,
                    timeout=3.0,
                    dns_error=2.0,
                ),
            )
        )

        # Create a connection dropped error
        # Mock the error analysis
        from unittest.mock import patch

        from l0.errors import NetworkError, NetworkErrorType

        with patch.object(NetworkError, "analyze") as mock_analyze:
            mock_analyze.return_value = type(
                "Analysis", (), {"type": NetworkErrorType.CONNECTION_DROPPED}
            )()

            # Use categorize_error to return NETWORK
            with patch("l0.retry.categorize_error") as mock_cat:
                mock_cat.return_value = ErrorCategory.NETWORK
                error = Exception("Connection dropped")
                delay = mgr.get_delay(error)
                assert delay == 5.0

    @pytest.mark.asyncio
    async def test_wait_sleeps_for_delay(self):
        """wait() should sleep for the calculated delay."""
        import asyncio
        from unittest.mock import patch

        mgr = RetryManager(Retry(base_delay=0.1, strategy=BackoffStrategy.FIXED))
        error = Exception("Error")

        with patch.object(asyncio, "sleep") as mock_sleep:
            mock_sleep.return_value = None
            await mgr.wait(error)
            mock_sleep.assert_called_once_with(0.1)


class TestErrorToRetryableType:
    """Tests for _error_to_retryable_type helper function."""

    def test_timeout_error(self):
        """TimeoutError should map to TIMEOUT."""
        from l0.runtime import TimeoutError as L0TimeoutError

        error = L0TimeoutError("Timed out", "initial_token", 5.0)
        assert _error_to_retryable_type(error) == RetryableErrorType.TIMEOUT

    def test_network_error(self):
        """Network errors should map to NETWORK_ERROR."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.NETWORK
            error = Exception("Connection failed")
            assert _error_to_retryable_type(error) == RetryableErrorType.NETWORK_ERROR

    def test_rate_limit_error(self):
        """Rate limit errors should map to RATE_LIMIT."""
        from unittest.mock import patch

        class RateLimitError(Exception):
            status_code = 429

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.TRANSIENT
            error = RateLimitError("rate limit exceeded")
            assert _error_to_retryable_type(error) == RetryableErrorType.RATE_LIMIT

    def test_server_error(self):
        """Server errors (5xx) should map to SERVER_ERROR."""
        from unittest.mock import patch

        class ServerError(Exception):
            status_code = 503

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.TRANSIENT
            error = ServerError("Service unavailable")
            assert _error_to_retryable_type(error) == RetryableErrorType.SERVER_ERROR

    def test_zero_output_content_error(self):
        """Zero output content errors should map to ZERO_OUTPUT."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.CONTENT
            error = Exception("zero output detected")
            assert _error_to_retryable_type(error) == RetryableErrorType.ZERO_OUTPUT

    def test_guardrail_violation_error(self):
        """Guardrail violation errors should map to GUARDRAIL_VIOLATION."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.CONTENT
            error = Exception("guardrail violation")
            assert (
                _error_to_retryable_type(error)
                == RetryableErrorType.GUARDRAIL_VIOLATION
            )

    def test_drift_error(self):
        """Drift errors should map to DRIFT."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.CONTENT
            error = Exception("drift detected")
            assert _error_to_retryable_type(error) == RetryableErrorType.DRIFT

    def test_incomplete_error(self):
        """Incomplete errors should map to INCOMPLETE."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.CONTENT
            error = Exception("incomplete response")
            assert _error_to_retryable_type(error) == RetryableErrorType.INCOMPLETE

    def test_model_error_zero_output(self):
        """Model errors with zero/empty should map to ZERO_OUTPUT."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.MODEL
            error = Exception("zero tokens generated")
            assert _error_to_retryable_type(error) == RetryableErrorType.ZERO_OUTPUT

    def test_fatal_error_returns_none(self):
        """Fatal errors should return None (not retryable)."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.FATAL
            error = Exception("Fatal error")
            assert _error_to_retryable_type(error) is None

    def test_internal_error_returns_none(self):
        """Internal errors should return None (not retryable)."""
        from unittest.mock import patch

        with patch("l0.retry.categorize_error") as mock_cat:
            mock_cat.return_value = ErrorCategory.INTERNAL
            error = Exception("Internal error")
            assert _error_to_retryable_type(error) is None


class TestRetryContext:
    """Tests for RetryContext dataclass."""

    def test_retry_context_fields(self):
        """RetryContext should have all expected fields."""
        error = Exception("Test error")
        ctx = RetryContext(
            attempt=2,
            error=error,
            category=ErrorCategory.NETWORK,
            is_network=True,
            model_retry_count=0,
            network_retry_count=2,
            total_retries=2,
            base_delay=1.0,
            max_delay=10.0,
        )

        assert ctx.attempt == 2
        assert ctx.error == error
        assert ctx.category == ErrorCategory.NETWORK
        assert ctx.is_network is True
        assert ctx.model_retry_count == 0
        assert ctx.network_retry_count == 2
        assert ctx.total_retries == 2
        assert ctx.base_delay == 1.0
        assert ctx.max_delay == 10.0


class TestRetryPresets:
    """Tests for Retry preset methods."""

    def test_recommended_preset(self):
        """Retry.recommended() should return sensible defaults."""
        config = Retry.recommended()
        assert config.attempts == 3
        assert config.max_retries == 6
        assert config.strategy == BackoffStrategy.FIXED_JITTER
        assert config.error_type_delays is not None

    def test_strict_preset(self):
        """Retry.strict() should use full jitter."""
        config = Retry.strict()
        assert config.attempts == 3
        assert config.max_retries == 6
        assert config.strategy == BackoffStrategy.FULL_JITTER

    def test_exponential_preset(self):
        """Retry.exponential() should use exponential backoff."""
        config = Retry.exponential()
        assert config.attempts == 4
        assert config.max_retries == 8
        assert config.strategy == BackoffStrategy.EXPONENTIAL

    def test_mobile_preset(self):
        """Retry.mobile() should have higher delays."""
        config = Retry.mobile()
        assert config.max_delay == 15.0
        assert config.error_type_delays is not None
        assert config.error_type_delays.background_throttle == 15.0

    def test_edge_preset(self):
        """Retry.edge() should have shorter delays."""
        config = Retry.edge()
        assert config.base_delay == 0.5
        assert config.max_delay == 5.0
