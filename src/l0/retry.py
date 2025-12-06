"""Retry manager with error-aware backoff."""

from __future__ import annotations

import asyncio
import inspect
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine

from .errors import NetworkError, NetworkErrorType, categorize_error
from .logging import logger
from .types import BackoffStrategy, ErrorCategory, Retry, RetryableErrorType

if TYPE_CHECKING:
    from .types import State


@dataclass
class RetryContext:
    """Context passed to custom delay calculation callback.

    Attributes:
        attempt: Current attempt number (0-based for delay calculation)
        error: The exception that triggered the retry
        category: Error category (NETWORK, TRANSIENT, MODEL, etc.)
        is_network: Whether this is a network error
        model_retry_count: Number of model retries so far
        network_retry_count: Number of network retries so far
        total_retries: Total number of retries so far
        base_delay: Configured base delay
        max_delay: Configured max delay
    """

    attempt: int
    error: Exception
    category: ErrorCategory
    is_network: bool
    model_retry_count: int
    network_retry_count: int
    total_retries: int
    base_delay: float
    max_delay: float


def _error_to_retryable_type(error: Exception) -> RetryableErrorType | None:
    """Map an exception to a RetryableErrorType."""
    category = categorize_error(error)

    # Check for timeout
    from .runtime import TimeoutError as L0TimeoutError

    if isinstance(error, L0TimeoutError):
        return RetryableErrorType.TIMEOUT

    # Check for network errors
    if category == ErrorCategory.NETWORK:
        return RetryableErrorType.NETWORK_ERROR

    # Check for transient errors (rate limit, server error)
    if category == ErrorCategory.TRANSIENT:
        msg = str(error).lower()
        status = getattr(error, "status_code", None) or getattr(error, "status", None)

        if "rate" in msg and "limit" in msg or status == 429:
            return RetryableErrorType.RATE_LIMIT
        if status and 500 <= status < 600:
            return RetryableErrorType.SERVER_ERROR
        return RetryableErrorType.TIMEOUT  # Default transient to timeout

    # Check for content errors
    if category == ErrorCategory.CONTENT:
        msg = str(error).lower()
        if "zero" in msg and "output" in msg:
            return RetryableErrorType.ZERO_OUTPUT
        if "guardrail" in msg or "violation" in msg:
            return RetryableErrorType.GUARDRAIL_VIOLATION
        if "drift" in msg:
            return RetryableErrorType.DRIFT
        if "incomplete" in msg:
            return RetryableErrorType.INCOMPLETE
        return RetryableErrorType.GUARDRAIL_VIOLATION  # Default content to guardrail

    # Check for model errors - map to content errors
    if category == ErrorCategory.MODEL:
        msg = str(error).lower()
        if "zero" in msg or "empty" in msg:
            return RetryableErrorType.ZERO_OUTPUT
        return RetryableErrorType.GUARDRAIL_VIOLATION

    return None


class RetryManager:
    """Manages retry logic with error-aware backoff.

    Supports:
    - Per-error-type delays
    - Custom retry_on filter (which error types to retry)
    - Custom should_retry callback (veto retries)
    - Custom calculate_delay callback (custom delay logic)
    """

    def __init__(self, config: Retry | None = None):
        self.config = config or Retry()
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0
        self._error_history: list[Exception] = []

    def _is_error_type_allowed(self, error: Exception) -> bool:
        """Check if error type is in the retry_on list."""
        if self.config.retry_on is None:
            return True  # No filter, all retryable types allowed

        error_type = _error_to_retryable_type(error)
        if error_type is None:
            return False

        return error_type in self.config.retry_on

    async def _check_should_retry_callback(
        self,
        error: Exception,
        state: "State | None",
        attempt: int,
        category: ErrorCategory,
    ) -> bool:
        """Check custom should_retry callback if provided."""
        if self.config.should_retry is None:
            return True

        # Call the callback (may be sync or async)
        result: bool | Coroutine[Any, Any, bool] = self.config.should_retry(
            error, state, attempt, category
        )

        # Handle async callback
        if inspect.iscoroutine(result):
            return bool(await result)

        return bool(result)

    def should_retry(
        self,
        error: Exception,
        state: "State | None" = None,
    ) -> bool:
        """Check if error should trigger a retry (sync version).

        For async support with should_retry callback, use should_retry_async().
        """
        category = categorize_error(error)
        logger.debug(
            f"Error category: {category}, model_retries: {self.model_retry_count}"
        )

        # Check absolute max
        max_retries = (
            self.config.max_retries if self.config.max_retries is not None else 10
        )
        if self.total_retries >= max_retries:
            logger.debug(f"Max retries reached: {self.total_retries}")
            return False

        # Check if error category is retryable at all
        if category in (ErrorCategory.FATAL, ErrorCategory.INTERNAL):
            logger.debug(f"Non-retryable category: {category}")
            return False

        # Check retry_on filter
        if not self._is_error_type_allowed(error):
            logger.debug("Error type not in retry_on list")
            return False

        # Check model retry limit for non-network errors
        if category not in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            attempts = self.config.attempts if self.config.attempts is not None else 3
            if self.model_retry_count >= attempts:
                logger.debug(
                    f"Model retry limit reached: {self.model_retry_count} >= {attempts}"
                )
                return False

        return True

    async def should_retry_async(
        self,
        error: Exception,
        state: "State | None" = None,
    ) -> bool:
        """Check if error should trigger a retry (async version).

        Supports async should_retry callback.
        """
        # First check basic conditions
        if not self.should_retry(error, state):
            return False

        # Then check custom callback
        category = categorize_error(error)
        attempt = (
            self.network_retry_count
            if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT)
            else self.model_retry_count
        )

        return await self._check_should_retry_callback(error, state, attempt, category)

    def record_attempt(self, error: Exception) -> None:
        """Record a retry attempt."""
        category = categorize_error(error)
        self.total_retries += 1
        self._error_history.append(error)

        if category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT):
            self.network_retry_count += 1
        else:
            self.model_retry_count += 1

        logger.debug(
            f"Recorded retry: total={self.total_retries}, "
            f"model={self.model_retry_count}, network={self.network_retry_count}"
        )

    def get_delay(self, error: Exception) -> float:
        """Get delay in seconds.

        Uses custom calculate_delay callback if provided, otherwise
        uses per-error-type delays for network errors or standard backoff.
        """
        category = categorize_error(error)
        is_network = category in (ErrorCategory.NETWORK, ErrorCategory.TRANSIENT)
        attempt = self.network_retry_count if is_network else self.model_retry_count

        # Check for custom delay calculation
        # Default values for delays
        default_base_delay = 1.0
        default_max_delay = 30.0
        base_delay = (
            self.config.base_delay
            if self.config.base_delay is not None
            else default_base_delay
        )
        max_delay = (
            self.config.max_delay
            if self.config.max_delay is not None
            else default_max_delay
        )

        if self.config.calculate_delay is not None:
            context = RetryContext(
                attempt=attempt,
                error=error,
                category=category,
                is_network=is_network,
                model_retry_count=self.model_retry_count,
                network_retry_count=self.network_retry_count,
                total_retries=self.total_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            )
            delay = self.config.calculate_delay(context)
            logger.debug(f"Custom delay: {delay:.2f}s")
            return float(delay)

        # Use per-error-type delays for network errors
        if category == ErrorCategory.NETWORK and self.config.error_type_delays:
            analysis = NetworkError.analyze(error)
            base = self._get_error_type_delay(analysis.type)
        else:
            base = base_delay

        cap = max_delay

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
        default_base_delay = 1.0
        base_delay = (
            self.config.base_delay
            if self.config.base_delay is not None
            else default_base_delay
        )

        if not self.config.error_type_delays:
            return base_delay

        delays = self.config.error_type_delays
        mapping: dict[NetworkErrorType, float | None] = {
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
        delay = mapping.get(error_type)
        return delay if delay is not None else base_delay

    async def wait(self, error: Exception) -> None:
        """Wait for the calculated delay before retrying."""
        delay = self.get_delay(error)
        await asyncio.sleep(delay)

    def get_state(self) -> dict[str, Any]:
        """Get current retry state."""
        return {
            "model_retry_count": self.model_retry_count,
            "network_retry_count": self.network_retry_count,
            "total_retries": self.total_retries,
            "error_history_length": len(self._error_history),
        }

    def get_error_history(self) -> list[Exception]:
        """Get list of errors that triggered retries."""
        return self._error_history.copy()

    def reset(self) -> None:
        """Reset retry state."""
        self.model_retry_count = 0
        self.network_retry_count = 0
        self.total_retries = 0
        self._error_history.clear()
