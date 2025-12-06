"""Tests for l0.errors module."""

import pytest

from l0.errors import (
    Error,
    ErrorCode,
    ErrorContext,
    NetworkError,
    NetworkErrorAnalysis,
    NetworkErrorType,
)
from l0.types import ErrorCategory


class TestL0Error:
    """Tests for L0 Error class."""

    def test_create_error_with_context(self):
        """Test creating error with context."""
        error = Error(
            "Test error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                checkpoint="some content",
                token_count=10,
            ),
        )

        assert str(error) == "Test error"
        assert error.code == ErrorCode.NETWORK_ERROR
        assert error.context.checkpoint == "some content"
        assert error.context.token_count == 10
        assert error.timestamp is not None

    def test_get_correct_category(self):
        """Test that error returns correct category."""
        network_error = Error("Network", ErrorContext(code=ErrorCode.NETWORK_ERROR))
        assert network_error.category == ErrorCategory.NETWORK

        content_error = Error("Content", ErrorContext(code=ErrorCode.GUARDRAIL_VIOLATION))
        assert content_error.category == ErrorCategory.CONTENT

    def test_has_checkpoint(self):
        """Test has_checkpoint property."""
        with_checkpoint = Error(
            "Error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                checkpoint="content",
            ),
        )
        assert with_checkpoint.has_checkpoint is True

        no_checkpoint = Error("Error", ErrorContext(code=ErrorCode.NETWORK_ERROR))
        assert no_checkpoint.has_checkpoint is False

        empty_checkpoint = Error(
            "Error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                checkpoint="",
            ),
        )
        assert empty_checkpoint.has_checkpoint is False

    def test_get_checkpoint(self):
        """Test get_checkpoint method."""
        error = Error(
            "Error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                checkpoint="saved content",
            ),
        )
        assert error.get_checkpoint() == "saved content"

    def test_to_detailed_string(self):
        """Test to_detailed_string format."""
        error = Error(
            "Test error",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                token_count=10,
                model_retry_count=2,
                fallback_index=1,
                checkpoint="content",
            ),
        )

        detailed = error.to_detailed_string()
        assert "Test error" in detailed
        assert "Tokens: 10" in detailed
        assert "Retries: 2" in detailed
        assert "Fallback: 1" in detailed
        assert "chars" in detailed
        assert " | " in detailed

    def test_to_json(self):
        """Test JSON serialization."""
        error = Error(
            "Test",
            ErrorContext(
                code=ErrorCode.NETWORK_ERROR,
                token_count=5,
                checkpoint="test content",
            ),
        )

        json_data = error.to_json()
        assert json_data["name"] == "Error"
        assert json_data["code"] == "NETWORK_ERROR"
        assert json_data["message"] == "Test"
        assert json_data["tokenCount"] == 5
        assert json_data["checkpoint"] == "test content"


class TestIsL0Error:
    """Tests for Error.is_error / Error.is_l0_error methods."""

    def test_returns_true_for_l0_error(self):
        """Test returns True for L0 Error."""
        error = Error("Test", ErrorContext(code=ErrorCode.NETWORK_ERROR))
        assert Error.is_error(error) is True
        assert Error.is_l0_error(error) is True

    def test_returns_false_for_regular_error(self):
        """Test returns False for regular Error."""
        error = Exception("Test")
        assert Error.is_error(error) is False
        assert Error.is_l0_error(error) is False

    def test_returns_false_for_non_errors(self):
        """Test returns False for non-errors."""
        assert Error.is_error("not an error") is False
        assert Error.is_error(None) is False
        assert Error.is_error(42) is False

        assert Error.is_error(None) is False
        assert Error.is_error("error") is False
        assert Error.is_error(42) is False


class TestGetErrorCategory:
    """Tests for Error.get_category method."""

    def test_categorize_network_errors(self):
        """Test network error categorization."""
        assert Error.get_category(ErrorCode.NETWORK_ERROR) == ErrorCategory.NETWORK

    def test_categorize_transient_errors(self):
        """Test transient error categorization."""
        assert (
            Error.get_category(ErrorCode.INITIAL_TOKEN_TIMEOUT)
            == ErrorCategory.TRANSIENT
        )
        assert (
            Error.get_category(ErrorCode.INTER_TOKEN_TIMEOUT) == ErrorCategory.TRANSIENT
        )

    def test_categorize_content_errors(self):
        """Test content error categorization."""
        assert (
            Error.get_category(ErrorCode.GUARDRAIL_VIOLATION) == ErrorCategory.CONTENT
        )
        assert (
            Error.get_category(ErrorCode.FATAL_GUARDRAIL_VIOLATION)
            == ErrorCategory.CONTENT
        )
        assert Error.get_category(ErrorCode.DRIFT_DETECTED) == ErrorCategory.CONTENT
        assert Error.get_category(ErrorCode.ZERO_OUTPUT) == ErrorCategory.CONTENT

    def test_categorize_internal_errors(self):
        """Test internal error categorization."""
        assert Error.get_category(ErrorCode.INVALID_STREAM) == ErrorCategory.INTERNAL
        assert Error.get_category(ErrorCode.ADAPTER_NOT_FOUND) == ErrorCategory.INTERNAL
        assert (
            Error.get_category(ErrorCode.FEATURE_NOT_ENABLED) == ErrorCategory.INTERNAL
        )

    def test_categorize_provider_errors(self):
        """Test provider error categorization."""
        assert Error.get_category(ErrorCode.STREAM_ABORTED) == ErrorCategory.PROVIDER
        assert (
            Error.get_category(ErrorCode.ALL_STREAMS_EXHAUSTED)
            == ErrorCategory.PROVIDER
        )


class TestNetworkErrorDetection:
    """Tests for NetworkError detection methods."""

    class TestIsConnectionDropped:
        def test_detect_connection_dropped_errors(self):
            assert (
                NetworkError.is_connection_dropped(Exception("connection dropped"))
                is True
            )
            assert (
                NetworkError.is_connection_dropped(Exception("connection closed"))
                is True
            )
            assert (
                NetworkError.is_connection_dropped(Exception("connection reset"))
                is True
            )
            assert NetworkError.is_connection_dropped(Exception("ECONNRESET")) is True
            assert NetworkError.is_connection_dropped(Exception("broken pipe")) is True

        def test_returns_false_for_other_errors(self):
            assert NetworkError.is_connection_dropped(Exception("timeout")) is False

    class TestIsFetchTypeError:
        def test_detect_fetch_type_errors(self):
            error = TypeError("Failed to fetch")
            assert NetworkError.is_fetch_error(error) is True

        def test_detect_network_request_failed(self):
            error = TypeError("Network request failed")
            assert NetworkError.is_fetch_error(error) is True

        def test_returns_false_for_non_type_errors(self):
            assert NetworkError.is_fetch_error(Exception("Failed to fetch")) is False

    class TestIsECONNRESET:
        def test_detect_econnreset_errors(self):
            assert NetworkError.is_econnreset(Exception("ECONNRESET")) is True
            assert (
                NetworkError.is_econnreset(Exception("connection reset by peer"))
                is True
            )

    class TestIsECONNREFUSED:
        def test_detect_econnrefused_errors(self):
            assert NetworkError.is_econnrefused(Exception("ECONNREFUSED")) is True
            assert NetworkError.is_econnrefused(Exception("connection refused")) is True

    class TestIsSSEAborted:
        def test_detect_sse_aborted_errors(self):
            assert (
                NetworkError.is_sse_aborted(Exception("SSE connection failed")) is True
            )
            assert NetworkError.is_sse_aborted(Exception("stream aborted")) is True

    class TestIsNoBytes:
        def test_detect_no_bytes_errors(self):
            assert NetworkError.is_no_bytes(Exception("no bytes received")) is True
            assert NetworkError.is_no_bytes(Exception("empty response")) is True
            assert NetworkError.is_no_bytes(Exception("zero bytes")) is True

    class TestIsPartialChunks:
        def test_detect_partial_chunk_errors(self):
            assert (
                NetworkError.is_partial_chunks(Exception("partial chunk received"))
                is True
            )
            assert (
                NetworkError.is_partial_chunks(Exception("truncated response")) is True
            )
            assert NetworkError.is_partial_chunks(Exception("premature close")) is True

    class TestIsRuntimeKilled:
        def test_detect_runtime_killed_errors(self):
            assert (
                NetworkError.is_runtime_killed(Exception("worker terminated")) is True
            )
            assert NetworkError.is_runtime_killed(Exception("lambda timeout")) is True
            assert NetworkError.is_runtime_killed(Exception("SIGTERM")) is True

    class TestIsBackgroundThrottle:
        def test_detect_background_throttle_errors(self):
            assert (
                NetworkError.is_background_throttle(Exception("background suspend"))
                is True
            )
            assert (
                NetworkError.is_background_throttle(Exception("tab suspended")) is True
            )
            assert NetworkError.is_background_throttle(Exception("page hidden")) is True

    class TestIsDNSError:
        def test_detect_dns_errors(self):
            assert NetworkError.is_dns(Exception("DNS lookup failed")) is True
            assert NetworkError.is_dns(Exception("ENOTFOUND")) is True
            assert NetworkError.is_dns(Exception("getaddrinfo failed")) is True

    class TestIsSSLError:
        def test_detect_ssl_errors(self):
            assert NetworkError.is_ssl(Exception("SSL handshake failed")) is True
            assert NetworkError.is_ssl(Exception("certificate expired")) is True
            assert NetworkError.is_ssl(Exception("self signed certificate")) is True

    class TestIsTimeoutError:
        def test_detect_timeout_errors(self):
            assert NetworkError.is_timeout(Exception("timeout")) is True
            assert NetworkError.is_timeout(Exception("timed out")) is True
            assert NetworkError.is_timeout(Exception("deadline exceeded")) is True

        def test_detect_timeout_error_by_type(self):
            assert NetworkError.is_timeout(TimeoutError("Operation timed out")) is True


class TestAnalyzeNetworkError:
    """Tests for NetworkError.analyze method."""

    def test_analyze_connection_dropped_error(self):
        analysis = NetworkError.analyze(Exception("connection dropped"))
        assert analysis.type == NetworkErrorType.CONNECTION_DROPPED
        assert analysis.retryable is True
        assert analysis.counts_toward_limit is False

    def test_analyze_fetch_error(self):
        error = TypeError("Failed to fetch")
        analysis = NetworkError.analyze(error)
        assert analysis.type == NetworkErrorType.FETCH_ERROR
        assert analysis.retryable is True

    def test_analyze_ssl_error_as_non_retryable(self):
        analysis = NetworkError.analyze(Exception("SSL certificate error"))
        assert analysis.type == NetworkErrorType.SSL_ERROR
        assert analysis.retryable is False

    def test_return_unknown_for_unrecognized_errors(self):
        analysis = NetworkError.analyze(Exception("some random error"))
        assert analysis.type == NetworkErrorType.UNKNOWN
        assert analysis.retryable is True


class TestIsNetworkError:
    """Tests for NetworkError.check method."""

    def test_returns_true_for_network_errors(self):
        assert NetworkError.check(Exception("connection dropped")) is True
        assert NetworkError.check(Exception("ECONNRESET")) is True
        assert NetworkError.check(Exception("timeout")) is True

    def test_returns_false_for_non_network_errors(self):
        assert NetworkError.check(Exception("syntax error")) is False
        assert NetworkError.check(Exception("undefined is not a function")) is False


class TestDescribeNetworkError:
    """Tests for NetworkError.describe method."""

    def test_describe_network_error(self):
        description = NetworkError.describe(Exception("connection dropped"))
        assert "Network error" in description
        assert "connection_dropped" in description

    def test_include_possible_cause_if_available(self):
        description = NetworkError.describe(Exception("ECONNREFUSED"))
        assert "econnrefused" in description
        assert "Server may be down" in description


class TestCreateNetworkError:
    """Tests for NetworkError.create method."""

    def test_create_enhanced_error_with_analysis(self):
        original = Exception("connection dropped")
        analysis = NetworkError.analyze(original)
        enhanced = NetworkError.create(original, analysis)

        assert hasattr(enhanced, "analysis")
        assert getattr(enhanced, "analysis") == analysis
        assert "connection_dropped" in str(enhanced)


class TestIsStreamInterrupted:
    """Tests for NetworkError.is_stream_interrupted method."""

    def test_returns_true_for_network_error_with_tokens(self):
        assert (
            NetworkError.is_stream_interrupted(Exception("connection dropped"), 5)
            is True
        )

    def test_returns_false_for_network_error_with_no_tokens(self):
        assert (
            NetworkError.is_stream_interrupted(Exception("connection dropped"), 0)
            is False
        )

    def test_detect_explicit_stream_interrupted_messages(self):
        assert (
            NetworkError.is_stream_interrupted(Exception("stream interrupted"), 0)
            is True
        )
        assert (
            NetworkError.is_stream_interrupted(
                Exception("connection lost mid-stream"), 0
            )
            is True
        )


class TestSuggestRetryDelay:
    """Tests for NetworkError.suggest_delay method."""

    def test_suggest_delay_based_on_error_type(self):
        conn_error = Exception("connection dropped")
        delay = NetworkError.suggest_delay(conn_error, 0)
        assert delay > 0

    def test_apply_exponential_backoff(self):
        error = Exception("connection dropped")
        delay0 = NetworkError.suggest_delay(error, 0)
        delay1 = NetworkError.suggest_delay(error, 1)
        delay2 = NetworkError.suggest_delay(error, 2)

        assert delay1 == delay0 * 2
        assert delay2 == delay0 * 4

    def test_return_zero_for_ssl_errors(self):
        ssl_error = Exception("SSL certificate error")
        delay = NetworkError.suggest_delay(ssl_error, 0)
        assert delay == 0

    def test_respect_max_delay(self):
        error = Exception("connection dropped")
        delay = NetworkError.suggest_delay(error, 10, max_delay=1.0)
        assert delay <= 1.0

    def test_use_custom_delays_if_provided(self):
        error = Exception("connection dropped")
        custom_delays = {NetworkErrorType.CONNECTION_DROPPED: 5.0}
        delay = NetworkError.suggest_delay(error, 0, custom_delays=custom_delays)
        assert delay == 5.0


class TestNetworkErrorClass:
    """Tests for NetworkError scoped API."""

    def test_check_detects_network_errors(self):
        assert NetworkError.check(Exception("connection dropped")) is True
        assert NetworkError.check(Exception("syntax error")) is False

    def test_analyze_returns_analysis(self):
        analysis = NetworkError.analyze(Exception("connection dropped"))
        assert isinstance(analysis, NetworkErrorAnalysis)
        assert analysis.type == NetworkErrorType.CONNECTION_DROPPED

    def test_describe_returns_string(self):
        desc = NetworkError.describe(Exception("timeout"))
        assert isinstance(desc, str)
        assert "timeout" in desc.lower()

    def test_create_returns_enhanced_error(self):
        original = Exception("connection dropped")
        enhanced = NetworkError.create(original)
        assert hasattr(enhanced, "analysis")

    def test_suggest_delay_with_custom_delays(self):
        error = Exception("timeout")
        custom = {NetworkErrorType.TIMEOUT: 3.0}
        delay = NetworkError.suggest_delay(error, 0, custom_delays=custom)
        assert delay == 3.0
