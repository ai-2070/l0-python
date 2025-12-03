"""Tests for l0.errors module."""

import pytest

from l0.errors import (
    Error,
    ErrorCode,
    ErrorContext,
    FailureType,
    NetworkError,
    NetworkErrorType,
    RecoveryPolicy,
    RecoveryStrategy,
)
from l0.types import ErrorCategory, ErrorTypeDelays, Retry

# ─────────────────────────────────────────────────────────────────────────────
# Error Class Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestError:
    """Tests for Error class."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = Error("Something went wrong", code=ErrorCode.ZERO_OUTPUT)
        assert str(error) == "Something went wrong"
        assert error.code == ErrorCode.ZERO_OUTPUT
        assert error.timestamp > 0

    def test_error_with_context(self):
        """Test error with full context."""
        context = ErrorContext(
            code=ErrorCode.GUARDRAIL_VIOLATION,
            checkpoint="Some partial content",
            token_count=50,
            content_length=200,
            model_retry_count=2,
            network_retry_count=1,
            fallback_index=0,
            metadata={"rule": "json_rule"},
        )
        error = Error(
            "Guardrail violation", code=ErrorCode.GUARDRAIL_VIOLATION, context=context
        )

        assert error.code == ErrorCode.GUARDRAIL_VIOLATION
        assert error.context.checkpoint == "Some partial content"
        assert error.context.token_count == 50
        assert error.context.metadata["rule"] == "json_rule"

    def test_has_checkpoint(self):
        """Test has_checkpoint property."""
        # No checkpoint
        error = Error("Error", code=ErrorCode.ZERO_OUTPUT)
        assert not error.has_checkpoint

        # With checkpoint
        context = ErrorContext(code=ErrorCode.ZERO_OUTPUT, checkpoint="partial")
        error = Error("Error", code=ErrorCode.ZERO_OUTPUT, context=context)
        assert error.has_checkpoint

    def test_get_checkpoint(self):
        """Test get_checkpoint method."""
        context = ErrorContext(code=ErrorCode.ZERO_OUTPUT, checkpoint="partial content")
        error = Error("Error", code=ErrorCode.ZERO_OUTPUT, context=context)
        assert error.get_checkpoint() == "partial content"

    def test_to_detailed_string(self):
        """Test to_detailed_string method."""
        context = ErrorContext(
            code=ErrorCode.ZERO_OUTPUT,
            checkpoint="Some content here",
            token_count=10,
            model_retry_count=2,
        )
        error = Error("Zero output", code=ErrorCode.ZERO_OUTPUT, context=context)
        detailed = error.to_detailed_string()

        assert "ZERO_OUTPUT" in detailed
        assert "Token count: 10" in detailed
        assert "Model retries: 2" in detailed
        assert "Has checkpoint: True" in detailed

    def test_repr(self):
        """Test error repr."""
        error = Error("Test error", code=ErrorCode.STREAM_ABORTED)
        assert "STREAM_ABORTED" in repr(error)
        assert "Test error" in repr(error)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_all_error_codes_exist(self):
        """Test all expected error codes exist."""
        assert ErrorCode.STREAM_ABORTED == "STREAM_ABORTED"
        assert ErrorCode.INITIAL_TOKEN_TIMEOUT == "INITIAL_TOKEN_TIMEOUT"
        assert ErrorCode.INTER_TOKEN_TIMEOUT == "INTER_TOKEN_TIMEOUT"
        assert ErrorCode.ZERO_OUTPUT == "ZERO_OUTPUT"
        assert ErrorCode.GUARDRAIL_VIOLATION == "GUARDRAIL_VIOLATION"
        assert ErrorCode.FATAL_GUARDRAIL_VIOLATION == "FATAL_GUARDRAIL_VIOLATION"
        assert ErrorCode.DRIFT_DETECTED == "DRIFT_DETECTED"
        assert ErrorCode.INVALID_STREAM == "INVALID_STREAM"
        assert ErrorCode.ALL_STREAMS_EXHAUSTED == "ALL_STREAMS_EXHAUSTED"
        assert ErrorCode.NETWORK_ERROR == "NETWORK_ERROR"

    def test_error_code_is_string_enum(self):
        """Test ErrorCode is a string enum."""
        assert isinstance(ErrorCode.ZERO_OUTPUT, str)


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        context = ErrorContext(code=ErrorCode.ZERO_OUTPUT)
        assert context.checkpoint is None
        assert context.token_count == 0
        assert context.content_length == 0
        assert context.model_retry_count == 0
        assert context.network_retry_count == 0
        assert context.fallback_index == 0
        assert context.metadata is None

    def test_full_context(self):
        """Test context with all fields."""
        context = ErrorContext(
            code=ErrorCode.GUARDRAIL_VIOLATION,
            checkpoint="checkpoint data",
            token_count=100,
            content_length=500,
            model_retry_count=3,
            network_retry_count=2,
            fallback_index=1,
            metadata={"key": "value"},
        )
        assert context.code == ErrorCode.GUARDRAIL_VIOLATION
        assert context.checkpoint == "checkpoint data"
        assert context.token_count == 100
        assert context.metadata == {"key": "value"}


class TestFailureType:
    """Tests for FailureType enum."""

    def test_failure_types(self):
        """Test all failure types exist."""
        assert FailureType.NETWORK == "network"
        assert FailureType.MODEL == "model"
        assert FailureType.TOOL == "tool"
        assert FailureType.TIMEOUT == "timeout"
        assert FailureType.ABORT == "abort"
        assert FailureType.ZERO_OUTPUT == "zero_output"
        assert FailureType.UNKNOWN == "unknown"


class TestRecoveryStrategy:
    """Tests for RecoveryStrategy enum."""

    def test_recovery_strategies(self):
        """Test all recovery strategies exist."""
        assert RecoveryStrategy.RETRY == "retry"
        assert RecoveryStrategy.FALLBACK == "fallback"
        assert RecoveryStrategy.CONTINUE == "continue"
        assert RecoveryStrategy.HALT == "halt"


class TestRecoveryPolicy:
    """Tests for RecoveryPolicy dataclass."""

    def test_default_values(self):
        """Test default policy values."""
        policy = RecoveryPolicy()
        assert policy.retry_enabled is True
        assert policy.fallback_enabled is False
        assert policy.max_retries == 3
        assert policy.max_fallbacks == 0
        assert policy.attempt == 1
        assert policy.fallback_index == 0

    def test_custom_policy(self):
        """Test custom policy values."""
        policy = RecoveryPolicy(
            retry_enabled=True,
            fallback_enabled=True,
            max_retries=5,
            max_fallbacks=2,
            attempt=3,
            fallback_index=1,
        )
        assert policy.max_retries == 5
        assert policy.max_fallbacks == 2
        assert policy.attempt == 3
        assert policy.fallback_index == 1


class TestErrorIsinstance:
    """Tests for isinstance() with Error."""

    def test_isinstance_error_true(self):
        """Test isinstance returns True for Error."""
        error = Error("Test", code=ErrorCode.ZERO_OUTPUT)
        assert isinstance(error, Error)

    def test_isinstance_error_false_for_exception(self):
        """Test isinstance returns False for regular Exception."""
        error = Exception("Test")
        assert not isinstance(error, Error)

    def test_isinstance_error_false_for_value_error(self):
        """Test isinstance returns False for ValueError."""
        error = ValueError("Test")
        assert not isinstance(error, Error)

    def test_error_is_exception(self):
        """Test Error is a subclass of Exception."""
        error = Error("Test", code=ErrorCode.ZERO_OUTPUT)
        assert isinstance(error, Exception)


class TestErrorIsRetryable:
    """Tests for Error.is_retryable() static method."""

    def test_network_errors_retryable(self):
        """Test network errors are retryable."""
        assert Error.is_retryable(Exception("Connection reset"))
        assert Error.is_retryable(Exception("Timeout"))

    def test_fatal_errors_not_retryable(self):
        """Test fatal errors are not retryable."""

        class AuthError(Exception):
            status_code = 401

        assert not Error.is_retryable(AuthError())

    def test_transient_errors_retryable(self):
        """Test transient errors are retryable."""

        class RateLimitError(Exception):
            status_code = 429

        assert Error.is_retryable(RateLimitError())


# ─────────────────────────────────────────────────────────────────────────────
# Categorize Error Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorCategorize:
    """Tests for Error.categorize() static method."""

    def test_network_errors(self):
        """Test network error patterns are detected."""
        network_errors = [
            Exception("Connection reset by peer"),
            Exception("Connection refused"),
            Exception("Connection timeout"),
            Exception("Request timed out"),
            Exception("DNS failed to resolve"),
            Exception("Name resolution failed"),
            Exception("Socket error occurred"),
            Exception("SSL error during handshake"),
            Exception("EOF occurred unexpectedly"),
            Exception("Broken pipe"),
            Exception("Network unreachable"),
            Exception("Host unreachable"),
        ]
        for error in network_errors:
            assert Error.categorize(error) == ErrorCategory.NETWORK, (
                f"Failed for: {error}"
            )

    def test_rate_limit_transient(self):
        """Test rate limit errors are transient."""
        error = Exception("Rate limit exceeded")
        assert Error.categorize(error) == ErrorCategory.TRANSIENT

    def test_http_429_transient(self):
        """Test HTTP 429 is transient."""

        class HTTPError(Exception):
            status_code = 429

        assert Error.categorize(HTTPError()) == ErrorCategory.TRANSIENT

    def test_http_503_transient(self):
        """Test HTTP 503 is transient."""

        class HTTPError(Exception):
            status_code = 503

        assert Error.categorize(HTTPError()) == ErrorCategory.TRANSIENT

    def test_http_401_fatal(self):
        """Test HTTP 401 is fatal."""

        class HTTPError(Exception):
            status_code = 401

        assert Error.categorize(HTTPError()) == ErrorCategory.FATAL

    def test_http_403_fatal(self):
        """Test HTTP 403 is fatal."""

        class HTTPError(Exception):
            status_code = 403

        assert Error.categorize(HTTPError()) == ErrorCategory.FATAL

    def test_unknown_error_is_model(self):
        """Test unknown errors default to MODEL category."""
        error = Exception("Some unknown error")
        assert Error.categorize(error) == ErrorCategory.MODEL


class TestNetworkErrorDetection:
    """Test NetworkError specific detection methods."""

    def test_is_connection_dropped(self):
        """Test connection dropped detection."""
        assert NetworkError.is_connection_dropped(Exception("Connection reset by peer"))
        assert NetworkError.is_connection_dropped(Exception("connection closed"))
        assert NetworkError.is_connection_dropped(Exception("Broken pipe"))
        assert not NetworkError.is_connection_dropped(Exception("Some other error"))

    def test_is_econnreset(self):
        """Test ECONNRESET detection."""
        assert NetworkError.is_econnreset(Exception("ECONNRESET"))
        assert NetworkError.is_econnreset(Exception("Connection reset by peer"))
        assert not NetworkError.is_econnreset(Exception("Connection refused"))

    def test_is_econnrefused(self):
        """Test ECONNREFUSED detection."""
        assert NetworkError.is_econnrefused(Exception("ECONNREFUSED"))
        assert NetworkError.is_econnrefused(Exception("Connection refused"))
        assert not NetworkError.is_econnrefused(Exception("Connection reset"))

    def test_is_timeout(self):
        """Test timeout detection."""
        assert NetworkError.is_timeout(TimeoutError("Request timed out"))
        assert NetworkError.is_timeout(Exception("Connection timeout"))
        assert NetworkError.is_timeout(Exception("deadline exceeded"))
        assert not NetworkError.is_timeout(Exception("Connection refused"))

    def test_is_dns(self):
        """Test DNS error detection."""
        assert NetworkError.is_dns(Exception("DNS lookup failed"))
        assert NetworkError.is_dns(Exception("getaddrinfo failed"))
        assert NetworkError.is_dns(Exception("Name resolution failed"))
        assert not NetworkError.is_dns(Exception("Connection refused"))

    def test_is_ssl(self):
        """Test SSL error detection."""
        assert NetworkError.is_ssl(Exception("SSL handshake failed"))
        assert NetworkError.is_ssl(Exception("Certificate verify failed"))
        assert NetworkError.is_ssl(Exception("TLS error"))
        assert not NetworkError.is_ssl(Exception("Connection refused"))

    def test_is_sse_aborted(self):
        """Test SSE aborted detection."""
        assert NetworkError.is_sse_aborted(Exception("SSE connection aborted"))
        assert NetworkError.is_sse_aborted(Exception("stream aborted"))
        assert not NetworkError.is_sse_aborted(Exception("Connection refused"))

    def test_is_no_bytes(self):
        """Test no bytes detection."""
        assert NetworkError.is_no_bytes(Exception("No bytes received"))
        assert NetworkError.is_no_bytes(Exception("Empty response"))
        assert not NetworkError.is_no_bytes(Exception("Connection refused"))

    def test_is_partial_chunks(self):
        """Test partial chunks detection."""
        assert NetworkError.is_partial_chunks(Exception("Incomplete chunk"))
        assert NetworkError.is_partial_chunks(Exception("Truncated response"))
        assert NetworkError.is_partial_chunks(Exception("premature close"))
        assert not NetworkError.is_partial_chunks(Exception("Connection refused"))

    def test_is_runtime_killed(self):
        """Test runtime killed detection."""
        assert NetworkError.is_runtime_killed(Exception("Lambda timeout"))
        assert NetworkError.is_runtime_killed(Exception("Worker terminated"))
        assert NetworkError.is_runtime_killed(Exception("SIGTERM received"))
        assert not NetworkError.is_runtime_killed(Exception("Connection refused"))

    def test_is_background_throttle(self):
        """Test background throttle detection."""
        assert NetworkError.is_background_throttle(Exception("Tab suspended"))
        assert NetworkError.is_background_throttle(Exception("Background throttle"))
        assert not NetworkError.is_background_throttle(Exception("Connection refused"))

    def test_check_any_network_error(self):
        """Test NetworkError.check() detects any network error."""
        assert NetworkError.check(Exception("Connection reset"))
        assert NetworkError.check(Exception("DNS failed"))
        assert NetworkError.check(TimeoutError("Timed out"))
        assert not NetworkError.check(Exception("Invalid JSON"))


class TestNetworkErrorAnalysis:
    """Test NetworkError.analyze() method."""

    def test_analyze_connection_dropped(self):
        """Test analysis of connection dropped errors."""
        analysis = NetworkError.analyze(Exception("Connection reset by peer"))
        assert analysis.type == NetworkErrorType.CONNECTION_DROPPED
        assert analysis.retryable is True
        assert analysis.counts_toward_limit is False

    def test_analyze_ssl_not_retryable(self):
        """Test SSL errors are not retryable."""
        analysis = NetworkError.analyze(Exception("SSL certificate verify failed"))
        assert analysis.type == NetworkErrorType.SSL_ERROR
        assert analysis.retryable is False

    def test_analyze_timeout(self):
        """Test analysis of timeout errors."""
        analysis = NetworkError.analyze(TimeoutError("Request timed out"))
        assert analysis.type == NetworkErrorType.TIMEOUT
        assert analysis.retryable is True

    def test_analyze_unknown(self):
        """Test analysis of unknown errors."""
        analysis = NetworkError.analyze(Exception("Unknown network issue"))
        assert analysis.type == NetworkErrorType.UNKNOWN
        assert analysis.retryable is True


class TestNetworkErrorUtilities:
    """Test NetworkError utility methods."""

    def test_describe(self):
        """Test NetworkError.describe()."""
        desc = NetworkError.describe(Exception("Connection refused"))
        assert "econnrefused" in desc.lower()
        assert "network error" in desc.lower()

    def test_suggest_delay(self):
        """Test NetworkError.suggest_delay()."""
        # ECONNREFUSED has 2.0s base delay
        delay = NetworkError.suggest_delay(Exception("Connection refused"), attempt=0)
        assert delay == 2.0

        # Exponential backoff
        delay = NetworkError.suggest_delay(Exception("Connection refused"), attempt=2)
        assert delay == 8.0  # 2.0 * 2^2

    def test_suggest_delay_respects_max(self):
        """Test suggest_delay respects max_delay."""
        delay = NetworkError.suggest_delay(
            Exception("Connection refused"),
            attempt=10,
            max_delay=5.0,
        )
        assert delay == 5.0

    def test_is_stream_interrupted(self):
        """Test NetworkError.is_stream_interrupted()."""
        err = Exception("Connection reset")
        # With tokens received, it's interrupted
        assert NetworkError.is_stream_interrupted(err, token_count=50)
        # Without tokens, not interrupted
        assert not NetworkError.is_stream_interrupted(err, token_count=0)


class TestRetryPresets:
    """Test Retry class preset methods."""

    def test_recommended(self):
        """Test Retry.recommended() preset."""
        retry = Retry.recommended()
        assert retry.attempts == 3
        assert retry.max_retries == 6
        assert retry.error_type_delays is not None

    def test_mobile(self):
        """Test Retry.mobile() preset."""
        retry = Retry.mobile()
        assert retry.max_delay == 15.0
        assert retry.error_type_delays.background_throttle == 15.0
        assert retry.error_type_delays.timeout == 3.0

    def test_edge(self):
        """Test Retry.edge() preset."""
        retry = Retry.edge()
        assert retry.base_delay == 0.5
        assert retry.max_delay == 5.0


class TestErrorTypeDelays:
    """Test ErrorTypeDelays configuration."""

    def test_default_values(self):
        """Test default delay values."""
        delays = ErrorTypeDelays()
        assert delays.connection_dropped == 1.0
        assert delays.econnrefused == 2.0
        assert delays.background_throttle == 5.0

    def test_custom_values(self):
        """Test custom delay values."""
        delays = ErrorTypeDelays(
            timeout=5.0,
            connection_dropped=3.0,
        )
        assert delays.timeout == 5.0
        assert delays.connection_dropped == 3.0
        assert delays.econnreset == 1.0  # default
