"""Tests for l0.errors module."""

import pytest

from l0.errors import categorize_error
from l0.types import ErrorCategory


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
