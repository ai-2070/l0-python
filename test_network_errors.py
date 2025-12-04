"""Test network error handling implementation."""

from l0 import (
    ErrorTypeDelays,
    NetworkError,
    NetworkErrorAnalysis,
    NetworkErrorType,
    Retry,
)


def test_network_error_detection():
    """Test specific error detection methods."""
    # Test ECONNRESET
    err = Exception("Connection reset by peer")
    assert NetworkError.check(err), "Should detect as network error"
    assert NetworkError.is_econnreset(err), "Should detect ECONNRESET"
    assert NetworkError.is_connection_dropped(err), "Should detect connection dropped"

    # Test timeout
    timeout_err = TimeoutError("Request timed out")
    assert NetworkError.check(timeout_err), "Should detect timeout as network error"
    assert NetworkError.is_timeout(timeout_err), "Should detect timeout"

    # Test DNS
    dns_err = Exception("getaddrinfo failed: Name or service not known")
    assert NetworkError.is_dns(dns_err), "Should detect DNS error"

    # Test SSL (not retryable)
    ssl_err = Exception("SSL certificate verify failed")
    assert NetworkError.is_ssl(ssl_err), "Should detect SSL error"

    print("  Detection tests passed")


def test_network_error_analysis():
    """Test error analysis."""
    # "Connection reset by peer" matches connection_dropped first (both patterns match)
    err = Exception("Connection reset by peer")
    analysis = NetworkError.analyze(err)

    assert analysis.type == NetworkErrorType.CONNECTION_DROPPED
    assert analysis.retryable is True
    assert analysis.counts_toward_limit is False
    assert "backoff" in analysis.suggestion.lower()

    # SSL should not be retryable
    ssl_err = Exception("SSL handshake failed")
    ssl_analysis = NetworkError.analyze(ssl_err)
    assert ssl_analysis.type == NetworkErrorType.SSL_ERROR
    assert ssl_analysis.retryable is False

    print("  Analysis tests passed")


def test_utility_functions():
    """Test utility functions."""
    err = Exception("Connection refused")

    # describe
    desc = NetworkError.describe(err)
    assert "econnrefused" in desc.lower()

    # suggest_delay
    delay = NetworkError.suggest_delay(err, attempt=0)
    assert delay == 2.0, f"Expected 2.0s for ECONNREFUSED, got {delay}"

    delay_exp = NetworkError.suggest_delay(err, attempt=2)
    assert delay_exp == 8.0, f"Expected 8.0s (2.0 * 2^2), got {delay_exp}"

    # is_stream_interrupted
    assert NetworkError.is_stream_interrupted(err, token_count=50)
    assert not NetworkError.is_stream_interrupted(err, token_count=0)

    print("  Utility tests passed")


def test_retry_presets():
    """Test Retry class presets."""
    # recommended
    rec = Retry.recommended()
    assert rec.attempts == 3
    assert rec.max_retries == 6
    assert rec.error_type_delays is not None

    # mobile
    mob = Retry.mobile()
    assert mob.max_delay == 15.0
    assert mob.error_type_delays.background_throttle == 15.0

    # edge
    edge = Retry.edge()
    assert edge.base_delay == 0.5
    assert edge.max_delay == 5.0

    print("  Retry preset tests passed")


def test_error_type_delays():
    """Test ErrorTypeDelays configuration."""
    delays = ErrorTypeDelays(
        timeout=5.0,
        connection_dropped=3.0,
    )
    assert delays.timeout == 5.0
    assert delays.connection_dropped == 3.0
    assert delays.econnreset == 1.0  # default

    print("  ErrorTypeDelays tests passed")


if __name__ == "__main__":
    print("Testing network error implementation...")
    test_network_error_detection()
    test_network_error_analysis()
    test_utility_functions()
    test_retry_presets()
    test_error_type_delays()
    print("\nAll tests passed!")
