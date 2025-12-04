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

    # Exhaustion errors
    ALL_STREAMS_EXHAUSTED = "ALL_STREAMS_EXHAUSTED"

    # Network errors
    NETWORK_ERROR = "NETWORK_ERROR"


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
    fallback_index: int = 0  # Current fallback index (0 = primary)


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
    token_count: int = 0  # Tokens before failure
    content_length: int = 0  # Content length before failure
    model_retry_count: int = 0  # Retry attempts made
    network_retry_count: int = 0  # Network retries made
    fallback_index: int = 0  # Which fallback was tried
    metadata: dict[str, Any] | None = None


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
        code: ErrorCode,
        context: ErrorContext | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or ErrorContext(code=code)
        self.timestamp = time.time()

    @property
    def has_checkpoint(self) -> bool:
        """Check if error has a checkpoint for continuation."""
        return bool(self.context.checkpoint)

    def get_checkpoint(self) -> str | None:
        """Get checkpoint content if available."""
        return self.context.checkpoint

    def to_detailed_string(self) -> str:
        """Get detailed string representation for logging."""
        lines = [
            f"Error [{self.code.value}]: {self.args[0]}",
            f"  Timestamp: {self.timestamp}",
            f"  Token count: {self.context.token_count}",
            f"  Content length: {self.context.content_length}",
            f"  Model retries: {self.context.model_retry_count}",
            f"  Network retries: {self.context.network_retry_count}",
            f"  Fallback index: {self.context.fallback_index}",
            f"  Has checkpoint: {self.has_checkpoint}",
        ]

        if self.context.metadata:
            lines.append(f"  Metadata: {self.context.metadata}")

        if self.has_checkpoint:
            checkpoint_preview = (self.context.checkpoint or "")[:100]
            if len(self.context.checkpoint or "") > 100:
                checkpoint_preview += "..."
            lines.append(f"  Checkpoint preview: {checkpoint_preview!r}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Error(code={self.code.value!r}, message={self.args[0]!r})"

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
        max_delay: float = 30.0,
    ) -> float:
        """Suggest retry delay based on network error type.

        Args:
            error: Error to analyze
            attempt: Retry attempt number (0-based)
            max_delay: Maximum delay cap (default: 30.0 seconds)

        Returns:
            Suggested delay in seconds
        """
        from .types import ErrorTypeDelays

        analysis = NetworkError.analyze(error)
        delays = ErrorTypeDelays()

        base_delay = NetworkError._get_type_delay(analysis.type, delays)
        if base_delay == 0:
            return 0.0

        # Exponential backoff
        return float(min(base_delay * (2**attempt), max_delay))

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_error_code(error: Exception) -> str | None:
        """Get error code if present (for OSError, etc.)."""
        if hasattr(error, "errno"):
            return str(error.errno)
        if hasattr(error, "code"):
            return str(error.code)
        return None

    @staticmethod
    def _get_type_delay(error_type: NetworkErrorType, delays: ErrorTypeDelays) -> float:
        """Get base delay for a specific network error type."""
        mapping: dict[NetworkErrorType, float] = {
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
            NetworkErrorType.SSL_ERROR: 0.0,
            NetworkErrorType.TIMEOUT: delays.timeout,
            NetworkErrorType.UNKNOWN: delays.unknown,
        }
        return mapping.get(error_type, delays.unknown)


# ─────────────────────────────────────────────────────────────────────────────
# Error Categorization (for retry decisions)
# ─────────────────────────────────────────────────────────────────────────────


def _categorize_error(error: Exception) -> ErrorCategory:
    """Categorize error for retry decisions (internal)."""
    msg = str(error).lower()

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


# Legacy aliases for backwards compatibility
categorize_error = _categorize_error
is_retryable = _is_retryable
