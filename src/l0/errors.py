"""Error handling for L0.

Provides structured error types, error codes, and recovery information.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .types import ErrorCategory, ErrorTypeDelays

# ─────────────────────────────────────────────────────────────────────────────
# Error Codes
# ─────────────────────────────────────────────────────────────────────────────


class ErrorCode(str, Enum):
    """Error codes for programmatic handling.

    Usage:
        from l0 import Error, ErrorCode

        try:
            result = await l0.run(stream)
        except Error as e:
            if e.code == ErrorCode.ZERO_OUTPUT:
                # Model produced nothing - maybe adjust prompt
                pass
            elif e.code == ErrorCode.GUARDRAIL_VIOLATION:
                # Content failed validation
                pass
    """

    # Stream errors
    STREAM_ABORTED = "STREAM_ABORTED"
    INITIAL_TOKEN_TIMEOUT = "INITIAL_TOKEN_TIMEOUT"
    INTER_TOKEN_TIMEOUT = "INTER_TOKEN_TIMEOUT"

    # Content errors
    ZERO_OUTPUT = "ZERO_OUTPUT"
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"
    FATAL_GUARDRAIL_VIOLATION = "FATAL_GUARDRAIL_VIOLATION"
    DRIFT_DETECTED = "DRIFT_DETECTED"

    # Configuration errors
    INVALID_STREAM = "INVALID_STREAM"
    ADAPTER_NOT_FOUND = "ADAPTER_NOT_FOUND"
    FEATURE_NOT_ENABLED = "FEATURE_NOT_ENABLED"

    # Exhaustion errors
    ALL_STREAMS_EXHAUSTED = "ALL_STREAMS_EXHAUSTED"

    # Network errors
    NETWORK_ERROR = "NETWORK_ERROR"


def get_error_category(code: ErrorCode) -> ErrorCategory:
    """Map error code to category.

    Args:
        code: L0 error code

    Returns:
        ErrorCategory for the given code

    Usage:
        from l0 import get_error_category, ErrorCode, ErrorCategory

        category = get_error_category(ErrorCode.NETWORK_ERROR)
        if category == ErrorCategory.NETWORK:
            # Network error - retry forever
            pass
    """
    if code == ErrorCode.NETWORK_ERROR:
        return ErrorCategory.NETWORK

    if code in (ErrorCode.INITIAL_TOKEN_TIMEOUT, ErrorCode.INTER_TOKEN_TIMEOUT):
        return ErrorCategory.TRANSIENT

    if code in (
        ErrorCode.GUARDRAIL_VIOLATION,
        ErrorCode.FATAL_GUARDRAIL_VIOLATION,
        ErrorCode.DRIFT_DETECTED,
        ErrorCode.ZERO_OUTPUT,
    ):
        return ErrorCategory.CONTENT

    if code in (
        ErrorCode.INVALID_STREAM,
        ErrorCode.ADAPTER_NOT_FOUND,
        ErrorCode.FEATURE_NOT_ENABLED,
    ):
        return ErrorCategory.INTERNAL

    if code in (ErrorCode.STREAM_ABORTED, ErrorCode.ALL_STREAMS_EXHAUSTED):
        return ErrorCategory.PROVIDER

    return ErrorCategory.MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Failure Types (what went wrong)
# ─────────────────────────────────────────────────────────────────────────────


class FailureType(str, Enum):
    """What actually went wrong - the root cause of the failure.

    Used in error events to classify the failure type.
    """

    NETWORK = "network"  # Connection drops, DNS, SSL, fetch errors
    MODEL = "model"  # Model refused, content filter, guardrail violation
    TOOL = "tool"  # Tool execution failed
    TIMEOUT = "timeout"  # Initial token or inter-token timeout
    ABORT = "abort"  # User or signal abort
    ZERO_OUTPUT = "zero_output"  # Empty response from model
    UNKNOWN = "unknown"  # Unclassified error


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Strategy (what L0 decided to do)
# ─────────────────────────────────────────────────────────────────────────────


class RecoveryStrategy(str, Enum):
    """What L0 decided to do next after an error."""

    RETRY = "retry"  # Will retry the same stream
    FALLBACK = "fallback"  # Will try next fallback stream
    CONTINUE = "continue"  # Will continue despite error (non-fatal)
    HALT = "halt"  # Will stop, no recovery possible


# ─────────────────────────────────────────────────────────────────────────────
# Recovery Policy (why L0 chose that strategy)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RecoveryPolicy:
    """Why L0 chose a particular recovery strategy.

    Provides context about the retry/fallback configuration and current state.
    """

    retry_enabled: bool = True
    fallback_enabled: bool = False
    max_retries: int = 3
    max_fallbacks: int = 0
    attempt: int = 1  # Current retry attempt (1-based)
    fallback_index: int | None = None  # Current fallback index (0 = primary)


# ─────────────────────────────────────────────────────────────────────────────
# Error Context
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ErrorContext:
    """Rich context for L0 errors.

    Provides detailed information about the state when the error occurred.
    """

    code: ErrorCode
    checkpoint: str | None = None  # Last good content for continuation
    token_count: int | None = None  # Tokens before failure
    content_length: int | None = None  # Content length before failure
    model_retry_count: int | None = None  # Retry attempts made
    network_retry_count: int | None = None  # Network retries made
    fallback_index: int | None = None  # Which fallback was tried
    metadata: dict[str, Any] | None = None  # Internal metadata
    context: dict[str, Any] | None = None  # User-provided context


# ─────────────────────────────────────────────────────────────────────────────
# Error Class
# ─────────────────────────────────────────────────────────────────────────────


class Error(Exception):
    """L0 error with rich context for debugging and recovery.

    Usage:
        from l0 import Error, is_error

        try:
            result = await l0.run(stream)
        except Error as e:
            print(e.code)           # ErrorCode.ZERO_OUTPUT
            print(e.context)        # ErrorContext with details
            print(e.has_checkpoint) # True if checkpoint available

            if e.has_checkpoint:
                checkpoint = e.get_checkpoint()
                # Retry with checkpoint context

            # Detailed string for logging
            print(e.to_detailed_string())

    Attributes:
        code: The error code (ErrorCode enum)
        context: Rich context about the error (ErrorContext)
        timestamp: Unix timestamp when error occurred
    """

    def __init__(
        self,
        message: str,
        context: ErrorContext,
    ) -> None:
        super().__init__(message)
        self.context = context
        self.code = context.code
        self.timestamp = time.time()

    @property
    def category(self) -> ErrorCategory:
        """Get error category for routing decisions."""
        return get_error_category(self.code)

    @property
    def has_checkpoint(self) -> bool:
        """Check if error has a checkpoint for continuation."""
        return bool(self.context.checkpoint)

    @property
    def is_recoverable(self) -> bool:
        """Check if error has checkpoint for recovery.

        Deprecated: Use has_checkpoint instead.
        """
        return self.has_checkpoint

    def get_checkpoint(self) -> str | None:
        """Get checkpoint content if available."""
        return self.context.checkpoint

    def to_detailed_string(self) -> str:
        """Get detailed string representation for logging.

        Returns a pipe-separated string with key information,
        matching the TypeScript format.
        """
        parts = [str(self.args[0]) if self.args else ""]

        if self.context.token_count:
            parts.append(f"Tokens: {self.context.token_count}")

        if self.context.model_retry_count:
            parts.append(f"Retries: {self.context.model_retry_count}")

        if self.context.fallback_index and self.context.fallback_index > 0:
            parts.append(f"Fallback: {self.context.fallback_index}")

        if self.context.checkpoint:
            parts.append(f"Checkpoint: {len(self.context.checkpoint)} chars")

        return " | ".join(parts)

    def __repr__(self) -> str:
        return f"Error(code={self.code.value!r}, message={self.args[0]!r})"

    def to_json(self) -> dict[str, Any]:
        """Serialize error for logging/transport.

        Returns:
            Dictionary with error details suitable for JSON serialization.

        Example:
            ```python
            try:
                result = await l0.run(stream)
            except Error as e:
                # Log to monitoring service
                import json
                log_entry = json.dumps(e.to_json())
                send_to_monitoring(log_entry)
            ```
        """
        return {
            "name": self.__class__.__name__,
            "code": self.code.value,
            "category": self.category.value,
            "message": str(self.args[0]) if self.args else "",
            "timestamp": self.timestamp,
            "hasCheckpoint": self.has_checkpoint,
            "checkpoint": self.context.checkpoint,
            "tokenCount": self.context.token_count,
            "modelRetryCount": self.context.model_retry_count,
            "networkRetryCount": self.context.network_retry_count,
            "fallbackIndex": self.context.fallback_index,
            "metadata": self.context.metadata,
            "context": self.context.context,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Static Methods for Any Exception
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def categorize(error: Exception) -> ErrorCategory:
        """Categorize any exception for retry decisions.

        Args:
            error: Any exception

        Returns:
            ErrorCategory (NETWORK, TRANSIENT, MODEL, FATAL, etc.)

        Usage:
            from l0 import Error, ErrorCategory

            category = Error.categorize(error)
            if category == ErrorCategory.NETWORK:
                # Retry forever with backoff
                pass
            elif category == ErrorCategory.FATAL:
                # Don't retry
                pass
        """
        return _categorize_error(error)

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Check if any exception should trigger a retry.

        Args:
            error: Any exception

        Returns:
            True if error is retryable, False otherwise

        Usage:
            from l0 import Error

            if Error.is_retryable(error):
                # retry logic
                pass
        """
        return _is_retryable(error)

    @staticmethod
    def is_error(error: Any) -> bool:
        """Type guard for L0 Error.

        Args:
            error: Any value to check

        Returns:
            True if error is an L0 Error instance

        Usage:
            from l0 import Error

            try:
                result = await l0.run(stream)
            except Exception as e:
                if Error.is_error(e):
                    print(e.code)  # Access L0-specific properties
                    print(e.category)
        """
        return isinstance(error, Error)

    # Alias for TypeScript parity
    is_l0_error = is_error

    @staticmethod
    def get_category(code: ErrorCode) -> ErrorCategory:
        """Map error code to category.

        Args:
            code: L0 error code

        Returns:
            ErrorCategory for the given code

        Usage:
            from l0 import Error, ErrorCode, ErrorCategory

            category = Error.get_category(ErrorCode.NETWORK_ERROR)
            if category == ErrorCategory.NETWORK:
                # Network error - retry forever
                pass
        """
        return get_error_category(code)


# ─────────────────────────────────────────────────────────────────────────────
# Network Error Types
# ─────────────────────────────────────────────────────────────────────────────


class NetworkErrorType(str, Enum):
    """Network error types that L0 can detect."""

    CONNECTION_DROPPED = "connection_dropped"
    FETCH_ERROR = "fetch_error"
    ECONNRESET = "econnreset"
    ECONNREFUSED = "econnrefused"
    SSE_ABORTED = "sse_aborted"
    NO_BYTES = "no_bytes"
    PARTIAL_CHUNKS = "partial_chunks"
    RUNTIME_KILLED = "runtime_killed"
    BACKGROUND_THROTTLE = "background_throttle"
    DNS_ERROR = "dns_error"
    SSL_ERROR = "ssl_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Network Error Analysis
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NetworkErrorAnalysis:
    """Detailed network error analysis."""

    type: NetworkErrorType
    retryable: bool
    counts_toward_limit: bool
    suggestion: str
    context: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# NetworkError Class - Scoped API
# ─────────────────────────────────────────────────────────────────────────────


class NetworkError:
    """Network error detection and analysis utilities.

    Usage:
        from l0 import NetworkError

        # Check specific error types
        if NetworkError.is_timeout(error):
            ...

        # Analyze any error
        analysis = NetworkError.analyze(error)
        print(analysis.type)        # NetworkErrorType.TIMEOUT
        print(analysis.retryable)   # True
        print(analysis.suggestion)  # "Retry with longer timeout..."

        # Get human-readable description
        desc = NetworkError.describe(error)

        # Get suggested retry delay
        delay = NetworkError.suggest_delay(error, attempt=2)

        # Check if stream was interrupted mid-flight
        if NetworkError.is_stream_interrupted(error, token_count=50):
            print("Partial content in checkpoint")
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Specific Error Detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def is_connection_dropped(error: Exception) -> bool:
        """Detect if error is a connection drop."""
        msg = str(error).lower()
        return (
            "connection dropped" in msg
            or "connection closed" in msg
            or "connection lost" in msg
            or "connection reset" in msg
            or "econnreset" in msg
            or "pipe broken" in msg
            or "broken pipe" in msg
            or "socket error" in msg
            or "eof occurred" in msg
            or "network unreachable" in msg
            or "host unreachable" in msg
        )

    @staticmethod
    def is_fetch_error(error: Exception) -> bool:
        """Detect if error is a fetch/request TypeError."""
        if not isinstance(error, TypeError):
            return False
        msg = str(error).lower()
        return (
            "fetch" in msg
            or "failed to fetch" in msg
            or "network request failed" in msg
        )

    @staticmethod
    def is_econnreset(error: Exception) -> bool:
        """Detect if error is ECONNRESET."""
        msg = str(error).lower()
        code = NetworkError._get_error_code(error)
        return (
            "econnreset" in msg
            or "connection reset by peer" in msg
            or code == "104"  # ECONNRESET on Linux
        )

    @staticmethod
    def is_econnrefused(error: Exception) -> bool:
        """Detect if error is ECONNREFUSED."""
        msg = str(error).lower()
        code = NetworkError._get_error_code(error)
        return (
            "econnrefused" in msg
            or "connection refused" in msg
            or code == "111"  # ECONNREFUSED on Linux
        )

    @staticmethod
    def is_sse_aborted(error: Exception) -> bool:
        """Detect if error is SSE abortion."""
        msg = str(error).lower()
        return (
            "sse" in msg
            or "server-sent events" in msg
            or ("stream" in msg and "abort" in msg)
            or "stream aborted" in msg
            or "eventstream" in msg
            or type(error).__name__ == "AbortError"
        )

    @staticmethod
    def is_no_bytes(error: Exception) -> bool:
        """Detect if error is due to no bytes arriving."""
        msg = str(error).lower()
        return (
            "no bytes" in msg
            or "empty response" in msg
            or "zero bytes" in msg
            or "no data received" in msg
            or "content-length: 0" in msg
        )

    @staticmethod
    def is_partial_chunks(error: Exception) -> bool:
        """Detect if error is due to partial/incomplete chunks."""
        msg = str(error).lower()
        return (
            "partial chunk" in msg
            or "incomplete chunk" in msg
            or "truncated" in msg
            or "premature close" in msg
            or "unexpected end of data" in msg
            or "incomplete data" in msg
            or "incomplete read" in msg
        )

    @staticmethod
    def is_runtime_killed(error: Exception) -> bool:
        """Detect if error is due to runtime being killed (Lambda/Edge timeout)."""
        msg = str(error).lower()
        return (
            ("worker" in msg and "terminated" in msg)
            or ("runtime" in msg and "killed" in msg)
            or "edge runtime" in msg
            or "lambda timeout" in msg
            or "function timeout" in msg
            or "execution timeout" in msg
            or "worker died" in msg
            or "process exited" in msg
            or "sigterm" in msg
            or "sigkill" in msg
        )

    @staticmethod
    def is_background_throttle(error: Exception) -> bool:
        """Detect if error is due to mobile/browser background throttling."""
        msg = str(error).lower()
        return (
            ("background" in msg and "suspend" in msg)
            or "background throttle" in msg
            or "tab suspended" in msg
            or "page hidden" in msg
            or "visibility hidden" in msg
            or "inactive tab" in msg
            or "background tab" in msg
        )

    @staticmethod
    def is_dns(error: Exception) -> bool:
        """Detect DNS errors."""
        msg = str(error).lower()
        code = NetworkError._get_error_code(error)
        return (
            "dns" in msg
            or "enotfound" in msg
            or "name resolution" in msg
            or "host not found" in msg
            or "getaddrinfo" in msg
            or "nodename nor servname provided" in msg
            or code == "-2"  # EAI_NONAME
        )

    @staticmethod
    def is_ssl(error: Exception) -> bool:
        """Detect SSL/TLS errors."""
        msg = str(error).lower()
        return (
            "ssl" in msg
            or "tls" in msg
            or "certificate" in msg
            or "cert" in msg
            or "handshake" in msg
            or "self signed" in msg
            or "unable to verify" in msg
        )

    @staticmethod
    def is_timeout(error: Exception) -> bool:
        """Detect timeout errors."""
        msg = str(error).lower()
        return (
            type(error).__name__ == "TimeoutError"
            or isinstance(error, TimeoutError)
            or "timeout" in msg
            or "timed out" in msg
            or "time out" in msg
            or "deadline exceeded" in msg
            or "etimedout" in msg
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def check(error: Exception) -> bool:
        """Check if error is any type of network error."""
        return (
            NetworkError.is_connection_dropped(error)
            or NetworkError.is_fetch_error(error)
            or NetworkError.is_econnreset(error)
            or NetworkError.is_econnrefused(error)
            or NetworkError.is_sse_aborted(error)
            or NetworkError.is_no_bytes(error)
            or NetworkError.is_partial_chunks(error)
            or NetworkError.is_runtime_killed(error)
            or NetworkError.is_background_throttle(error)
            or NetworkError.is_dns(error)
            or NetworkError.is_ssl(error)
            or NetworkError.is_timeout(error)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def analyze(error: Exception) -> NetworkErrorAnalysis:
        """Analyze network error and provide detailed information."""
        if NetworkError.is_connection_dropped(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.CONNECTION_DROPPED,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with exponential backoff - connection was interrupted",
            )

        if NetworkError.is_fetch_error(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.FETCH_ERROR,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry immediately - fetch() failed to initiate",
            )

        if NetworkError.is_econnreset(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.ECONNRESET,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with backoff - connection was reset by peer",
            )

        if NetworkError.is_econnrefused(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.ECONNREFUSED,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with longer delay - server refused connection",
                context={
                    "possible_cause": "Server may be down or not accepting connections"
                },
            )

        if NetworkError.is_sse_aborted(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.SSE_ABORTED,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry immediately - SSE stream was aborted",
            )

        if NetworkError.is_no_bytes(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.NO_BYTES,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry immediately - server sent no data",
                context={
                    "possible_cause": "Empty response or connection closed before data sent"
                },
            )

        if NetworkError.is_partial_chunks(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.PARTIAL_CHUNKS,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry immediately - received incomplete data",
                context={"possible_cause": "Connection closed mid-stream"},
            )

        if NetworkError.is_runtime_killed(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.RUNTIME_KILLED,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with shorter timeout - runtime was terminated",
                context={
                    "possible_cause": "Edge/Lambda timeout - consider smaller requests",
                },
            )

        if NetworkError.is_background_throttle(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.BACKGROUND_THROTTLE,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry when page becomes visible - mobile/browser throttling",
                context={
                    "possible_cause": "Browser suspended network for background tab",
                    "resolution": "Wait for visibility change event",
                },
            )

        if NetworkError.is_dns(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.DNS_ERROR,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with longer delay - DNS lookup failed",
                context={
                    "possible_cause": "Network connectivity issue or invalid hostname"
                },
            )

        if NetworkError.is_ssl(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.SSL_ERROR,
                retryable=False,
                counts_toward_limit=False,
                suggestion="Don't retry - SSL/TLS error (configuration issue)",
                context={
                    "possible_cause": "Certificate validation failed or SSL handshake error",
                    "resolution": "Check server certificate or SSL configuration",
                },
            )

        if NetworkError.is_timeout(error):
            return NetworkErrorAnalysis(
                type=NetworkErrorType.TIMEOUT,
                retryable=True,
                counts_toward_limit=False,
                suggestion="Retry with longer timeout - request timed out",
            )

        # Unknown network error
        return NetworkErrorAnalysis(
            type=NetworkErrorType.UNKNOWN,
            retryable=True,
            counts_toward_limit=False,
            suggestion="Retry with caution - unknown network error",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def describe(error: Exception) -> str:
        """Get human-readable description of network error."""
        analysis = NetworkError.analyze(error)
        description = f"Network error: {analysis.type.value}"

        if "possible_cause" in analysis.context:
            description += f" ({analysis.context['possible_cause']})"

        return description

    @staticmethod
    def is_stream_interrupted(error: Exception, token_count: int) -> bool:
        """Check if error indicates stream was interrupted mid-flight."""
        # If we received some tokens but then got a network error, stream was interrupted
        if token_count > 0 and NetworkError.check(error):
            return True

        # Check for specific interrupted stream indicators
        msg = str(error).lower()
        return (
            "stream interrupted" in msg
            or "stream closed unexpectedly" in msg
            or "connection lost mid-stream" in msg
            or (NetworkError.is_partial_chunks(error) and token_count > 0)
        )

    @staticmethod
    def suggest_delay(
        error: Exception,
        attempt: int,
        custom_delays: dict[NetworkErrorType, float] | None = None,
        max_delay: float = 30.0,
    ) -> float:
        """Suggest retry delay based on network error type.

        Args:
            error: Error to analyze
            attempt: Retry attempt number (0-based)
            custom_delays: Optional custom delays per error type (in seconds)
            max_delay: Maximum delay cap (default: 30.0 seconds)

        Returns:
            Suggested delay in seconds

        Example:
            ```python
            from l0 import NetworkError, NetworkErrorType

            # With default delays
            delay = NetworkError.suggest_delay(error, attempt=0)

            # With custom delays
            custom = {
                NetworkErrorType.CONNECTION_DROPPED: 2.0,
                NetworkErrorType.TIMEOUT: 1.5,
            }
            delay = NetworkError.suggest_delay(error, attempt=0, custom_delays=custom)
            ```
        """
        from .types import ErrorTypeDelays

        analysis = NetworkError.analyze(error)

        # Use custom delay if provided
        if custom_delays and analysis.type in custom_delays:
            base_delay = custom_delays[analysis.type]
        else:
            delays = ErrorTypeDelays()
            base_delay = NetworkError._get_type_delay(analysis.type, delays)

        if base_delay == 0:
            return 0.0

        # Exponential backoff
        return float(min(base_delay * (2**attempt), max_delay))

    @staticmethod
    def create(
        original: Exception,
        analysis: NetworkErrorAnalysis | None = None,
    ) -> Exception:
        """Create enhanced network error with analysis attached.

        Args:
            original: Original exception
            analysis: Optional pre-computed analysis (will compute if not provided)

        Returns:
            Enhanced exception with .analysis attribute

        Example:
            ```python
            from l0 import NetworkError

            try:
                # some network operation
                pass
            except Exception as e:
                if NetworkError.check(e):
                    enhanced = NetworkError.create(e)
                    print(enhanced.analysis.type)
                    print(enhanced.analysis.suggestion)
            ```
        """
        if analysis is None:
            analysis = NetworkError.analyze(original)

        # Create a new exception with the analysis attached
        msg = f"{original} [{analysis.type.value}]"
        enhanced = type(original)(msg)
        enhanced.analysis = analysis  # type: ignore
        enhanced.__cause__ = original
        return enhanced

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_error_code(error: Exception) -> str | None:
        """Get error code if present (for OSError, etc.)."""
        errno = getattr(error, "errno", None)
        if errno is not None:
            return str(errno)
        code = getattr(error, "code", None)
        if code is not None:
            return str(code)
        return None

    @staticmethod
    def _get_type_delay(error_type: NetworkErrorType, delays: ErrorTypeDelays) -> float:
        """Get base delay for a specific network error type."""
        default_delay = 1.0
        mapping: dict[NetworkErrorType, float] = {
            NetworkErrorType.CONNECTION_DROPPED: delays.connection_dropped
            or default_delay,
            NetworkErrorType.FETCH_ERROR: delays.fetch_error or default_delay,
            NetworkErrorType.ECONNRESET: delays.econnreset or default_delay,
            NetworkErrorType.ECONNREFUSED: delays.econnrefused or default_delay,
            NetworkErrorType.SSE_ABORTED: delays.sse_aborted or default_delay,
            NetworkErrorType.NO_BYTES: delays.no_bytes or default_delay,
            NetworkErrorType.PARTIAL_CHUNKS: delays.partial_chunks or default_delay,
            NetworkErrorType.RUNTIME_KILLED: delays.runtime_killed or default_delay,
            NetworkErrorType.BACKGROUND_THROTTLE: delays.background_throttle
            or default_delay,
            NetworkErrorType.DNS_ERROR: delays.dns_error or default_delay,
            NetworkErrorType.SSL_ERROR: delays.ssl_error or 0.0,
            NetworkErrorType.TIMEOUT: delays.timeout or default_delay,
            NetworkErrorType.UNKNOWN: delays.unknown or default_delay,
        }
        return mapping.get(error_type, delays.unknown or default_delay)


# ─────────────────────────────────────────────────────────────────────────────
# Error Categorization (for retry decisions)
# ─────────────────────────────────────────────────────────────────────────────


def _categorize_error(error: Exception) -> ErrorCategory:
    """Categorize error for retry decisions (internal)."""
    msg = str(error).lower()

    # Check for L0 Error class with specific error codes
    if isinstance(error, Error):
        if error.code == ErrorCode.GUARDRAIL_VIOLATION:
            # Recoverable guardrail violation - should retry
            return ErrorCategory.CONTENT
        if error.code == ErrorCode.FATAL_GUARDRAIL_VIOLATION:
            # Non-recoverable guardrail violation - halt
            return ErrorCategory.FATAL
        if error.code == ErrorCode.DRIFT_DETECTED:
            return ErrorCategory.CONTENT
        if error.code == ErrorCode.ZERO_OUTPUT:
            return ErrorCategory.CONTENT
        if error.code in (
            ErrorCode.INITIAL_TOKEN_TIMEOUT,
            ErrorCode.INTER_TOKEN_TIMEOUT,
        ):
            return ErrorCategory.TRANSIENT
        if error.code == ErrorCode.STREAM_ABORTED:
            return ErrorCategory.FATAL
        if error.code == ErrorCode.NETWORK_ERROR:
            return ErrorCategory.NETWORK

    # Check for L0 TimeoutError (inter-token timeout is transient, can retry with continuation)
    # Import here to avoid circular import
    from .runtime import TimeoutError as L0TimeoutError

    if isinstance(error, L0TimeoutError):
        # Inter-token timeout is transient - good candidate for continuation
        if error.timeout_type == "inter_token":
            return ErrorCategory.TRANSIENT
        # Initial token timeout is also transient but less likely to benefit from continuation
        return ErrorCategory.TRANSIENT

    # Check network patterns first
    if NetworkError.check(error):
        # SSL/TLS errors are not retryable (configuration issue)
        if NetworkError.is_ssl(error):
            return ErrorCategory.FATAL
        return ErrorCategory.NETWORK

    # Check HTTP status if available
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status:
        if status == 429:
            return ErrorCategory.TRANSIENT
        if status in (401, 403):
            return ErrorCategory.FATAL
        if 500 <= status < 600:
            return ErrorCategory.TRANSIENT

    # Check for rate limit in message
    if "rate" in msg and "limit" in msg:
        return ErrorCategory.TRANSIENT

    return ErrorCategory.MODEL


def _is_retryable(error: Exception) -> bool:
    """Determine if error should trigger retry (internal)."""
    category = _categorize_error(error)
    return category not in (ErrorCategory.FATAL, ErrorCategory.INTERNAL)


def is_error(error: Any) -> bool:
    """Type guard for L0 Error.

    Args:
        error: Any value to check

    Returns:
        True if error is an L0 Error instance

    Usage:
        from l0 import is_error, Error

        try:
            result = await l0.run(stream)
        except Exception as e:
            if is_error(e):
                print(e.code)  # Access L0-specific properties
                print(e.category)
    """
    return isinstance(error, Error)


# Alias for TypeScript parity
is_l0_error = is_error


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Network Error Detection Functions
# ─────────────────────────────────────────────────────────────────────────────
# These mirror the TypeScript exports for convenience


def is_connection_dropped(error: Exception) -> bool:
    """Detect if error is a connection drop."""
    return NetworkError.is_connection_dropped(error)


def is_fetch_error(error: Exception) -> bool:
    """Detect if error is a fetch/request TypeError."""
    return NetworkError.is_fetch_error(error)


def is_econnreset(error: Exception) -> bool:
    """Detect if error is ECONNRESET."""
    return NetworkError.is_econnreset(error)


def is_econnrefused(error: Exception) -> bool:
    """Detect if error is ECONNREFUSED."""
    return NetworkError.is_econnrefused(error)


def is_sse_aborted(error: Exception) -> bool:
    """Detect if error is SSE abortion."""
    return NetworkError.is_sse_aborted(error)


def is_no_bytes(error: Exception) -> bool:
    """Detect if error is due to no bytes arriving."""
    return NetworkError.is_no_bytes(error)


def is_partial_chunks(error: Exception) -> bool:
    """Detect if error is due to partial/incomplete chunks."""
    return NetworkError.is_partial_chunks(error)


def is_runtime_killed(error: Exception) -> bool:
    """Detect if error is due to runtime being killed."""
    return NetworkError.is_runtime_killed(error)


def is_background_throttle(error: Exception) -> bool:
    """Detect if error is due to mobile/browser background throttling."""
    return NetworkError.is_background_throttle(error)


def is_dns_error(error: Exception) -> bool:
    """Detect DNS errors."""
    return NetworkError.is_dns(error)


def is_ssl_error(error: Exception) -> bool:
    """Detect SSL/TLS errors."""
    return NetworkError.is_ssl(error)


def is_timeout_error(error: Exception) -> bool:
    """Detect timeout errors."""
    return NetworkError.is_timeout(error)


def is_network_error(error: Exception) -> bool:
    """Check if error is any type of network error."""
    return NetworkError.check(error)


def analyze_network_error(error: Exception) -> NetworkErrorAnalysis:
    """Analyze network error and provide detailed information."""
    return NetworkError.analyze(error)


def describe_network_error(error: Exception) -> str:
    """Get human-readable description of network error."""
    return NetworkError.describe(error)


def create_network_error(
    original: Exception,
    analysis: NetworkErrorAnalysis | None = None,
) -> Exception:
    """Create enhanced network error with analysis attached."""
    return NetworkError.create(original, analysis)


def is_stream_interrupted(error: Exception, token_count: int) -> bool:
    """Check if error indicates stream was interrupted mid-flight."""
    return NetworkError.is_stream_interrupted(error, token_count)


def suggest_retry_delay(
    error: Exception,
    attempt: int,
    custom_delays: dict[NetworkErrorType, float] | None = None,
    max_delay: float = 30.0,
) -> float:
    """Suggest retry delay based on network error type."""
    return NetworkError.suggest_delay(error, attempt, custom_delays, max_delay)


# Legacy aliases for backwards compatibility
categorize_error = _categorize_error
is_retryable = _is_retryable
