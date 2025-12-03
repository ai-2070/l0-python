"""Tests for l0.errors module."""

import pytest

from l0.errors import NetworkError, NetworkErrorType, categorize_error
from l0.types import ErrorCategory, ErrorTypeDelays, Retry


class TestCategorizeError:
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
            assert categorize_error(error) == ErrorCategory.NETWORK, (
                f"Failed for: {error}"
            )

    def test_rate_limit_transient(self):
        """Test rate limit errors are transient."""
        error = Exception("Rate limit exceeded")
        assert categorize_error(error) == ErrorCategory.TRANSIENT

    def test_http_429_transient(self):
        """Test HTTP 429 is transient."""

        class HTTPError(Exception):
            status_code = 429

        assert categorize_error(HTTPError()) == ErrorCategory.TRANSIENT

    def test_http_503_transient(self):
        """Test HTTP 503 is transient."""

        class HTTPError(Exception):
            status_code = 503

        assert categorize_error(HTTPError()) == ErrorCategory.TRANSIENT

    def test_http_401_fatal(self):
        """Test HTTP 401 is fatal."""

        class HTTPError(Exception):
            status_code = 401

        assert categorize_error(HTTPError()) == ErrorCategory.FATAL

    def test_http_403_fatal(self):
        """Test HTTP 403 is fatal."""

        class HTTPError(Exception):
            status_code = 403

        assert categorize_error(HTTPError()) == ErrorCategory.FATAL

    def test_unknown_error_is_model(self):
        """Test unknown errors default to MODEL category."""
        error = Exception("Some unknown error")
        assert categorize_error(error) == ErrorCategory.MODEL


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
