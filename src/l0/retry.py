"""Retry manager with error-aware backoff."""

from __future__ import annotations

import asyncio
import random

from .errors import NetworkError, NetworkErrorType, categorize_error
from .logging import logger
from .types import BackoffStrategy, ErrorCategory, Retry


class RetryManager:
    """Manages retry logic with error-aware backoff."""

    def __init__(self, config: Retry | None = None):
        self.config = config or Retry()
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0

    def should_retry(self, error: Exception) -> bool:
        category = categorize_error(error)
        logger.debug(
            f"Error category: {category}, model_retries: {self.model_retry_count}"
        )

        # Check absolute max
        if self.total_retries >= self.config.max_retries:
            return False

        if category in (ErrorCategory.FATAL, ErrorCategory.INTERNAL):
            return False
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            return True  # Always retry, doesn't count toward model limit

        # MODEL or CONTENT - counts toward limit
        return self.model_retry_count < self.config.attempts

    def record_attempt(self, error: Exception) -> None:
        category = categorize_error(error)
        self.total_retries += 1
        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            self.network_retry_count += 1
        else:
            self.model_retry_count += 1

    def get_delay(self, error: Exception) -> float:
        """Get delay in seconds (Pythonic).

        Uses per-error-type delays for network errors if configured,
        otherwise falls back to standard backoff calculation.
        """
        category = categorize_error(error)
        attempt = (
            self.network_retry_count
            if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT)
            else self.model_retry_count
        )

        # Use per-error-type delays for network errors
        if category == ErrorCategory.NETWORK and self.config.error_type_delays:
            analysis = NetworkError.analyze(error)
            base = self._get_error_type_delay(analysis.type)
        else:
            base = self.config.base_delay

        cap = self.config.max_delay

        match self.config.strategy:
            case BackoffStrategy.EXPONENTIAL:
                delay = min(base * (2**attempt), cap)
            case BackoffStrategy.LINEAR:
                delay = min(base * (attempt + 1), cap)
            case BackoffStrategy.FIXED:
                delay = base
            case BackoffStrategy.FIXED_JITTER:
                temp = min(base * (2**attempt), cap)
                delay = temp / 2 + random.random() * (temp / 2)
            case BackoffStrategy.FULL_JITTER:
                delay = random.random() * min(base * (2**attempt), cap)
            case _:
                delay = base

        logger.debug(f"Retry delay: {delay:.2f}s (strategy: {self.config.strategy})")
        return float(delay)

    def _get_error_type_delay(self, error_type: NetworkErrorType) -> float:
        """Get base delay for a specific network error type."""
        if not self.config.error_type_delays:
            return self.config.base_delay

        delays = self.config.error_type_delays
        mapping = {
            NetworkErrorType.CONNECTION_DROPPED: delays.connection_dropped,
            NetworkErrorType.FETCH_ERROR: delays.fetch_error,
            NetworkErrorType.ECONNRESET: delays.econnreset,
            NetworkErrorType.ECONNREFUSED: delays.econnrefused,
            NetworkErrorType.SSE_ABORTED: delays.sse_aborted,
            NetworkErrorType.NO_BYTES: delays.no_bytes,
            NetworkErrorType.PARTIAL_CHUNKS: delays.partial_chunks,
            NetworkErrorType.RUNTIME_KILLED: delays.runtime_killed,
            NetworkErrorType.BACKGROUND_THROTTLE: delays.background_throttle,
            NetworkErrorType.DNS_ERROR: delays.dns_error,
            NetworkErrorType.SSL_ERROR: delays.ssl_error,
            NetworkErrorType.TIMEOUT: delays.timeout,
            NetworkErrorType.UNKNOWN: delays.unknown,
        }
        return mapping.get(error_type, self.config.base_delay)

    async def wait(self, error: Exception) -> None:
        delay = self.get_delay(error)
        await asyncio.sleep(delay)

    def get_state(self) -> dict[str, int]:
        return {
            "model_retry_count": self.model_retry_count,
            "network_retry_count": self.network_retry_count,
            "total_retries": self.total_retries,
        }

    def reset(self) -> None:
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0
