"""Sentry integration for L0 monitoring."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from ..events import ObservabilityEvent, ObservabilityEventType
from .telemetry import Telemetry

if TYPE_CHECKING:
    pass


# Sentry severity levels
SeverityLevel = Literal["debug", "info", "warning", "error", "fatal"]


class SentryClient(Protocol):
    """Sentry client interface (compatible with sentry_sdk)."""

    def capture_exception(
        self,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Capture an exception."""
        ...

    def capture_message(
        self,
        message: str,
        level: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Capture a message."""
        ...

    def add_breadcrumb(self, **kwargs: Any) -> None:
        """Add a breadcrumb."""
        ...

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        ...

    def set_extra(self, key: str, value: Any) -> None:
        """Set extra data."""
        ...

    def set_context(self, name: str, context: dict[str, Any]) -> None:
        """Set context."""
        ...


class SentryConfig(BaseModel):
    """Configuration for Sentry integration.

    Attributes:
        capture_network_errors: Whether to capture network errors
        capture_guardrail_violations: Whether to capture guardrail violations
        min_guardrail_severity: Minimum severity to capture for guardrails
        breadcrumbs_for_tokens: Whether to add breadcrumbs for tokens
        enable_tracing: Whether to enable performance monitoring (spans)
        tags: Custom tags to add to all events
        environment: Environment name
    """

    capture_network_errors: bool = True
    capture_guardrail_violations: bool = True
    min_guardrail_severity: SeverityLevel = "error"
    breadcrumbs_for_tokens: bool = False
    enable_tracing: bool = True
    tags: dict[str, str] = Field(default_factory=dict)
    environment: str | None = None


class Sentry:
    """L0 Sentry integration for error tracking and performance monitoring.

    Usage:
        ```python
        import sentry_sdk
        from l0.monitoring import Sentry, SentryConfig

        # Initialize Sentry
        sentry_sdk.init(dsn="...")

        # Create Sentry integration
        sentry_integration = Sentry(
            sentry=sentry_sdk,
            config=SentryConfig(
                capture_network_errors=True,
                breadcrumbs_for_tokens=False,
            ),
        )

        # Use in L0 execution
        sentry_integration.start_execution("chat-completion", {"model": "gpt-4"})
        sentry_integration.start_stream()

        for token in stream:
            sentry_integration.record_token(token)

        sentry_integration.complete_stream(token_count)
        ```
    """

    def __init__(
        self,
        sentry: SentryClient,
        config: SentryConfig | None = None,
    ) -> None:
        """Initialize Sentry integration.

        Args:
            sentry: Sentry client instance (import sentry_sdk)
            config: Configuration options
        """
        self._sentry = sentry
        self._config = config or SentryConfig()

        # Set default tags
        for key, value in self._config.tags.items():
            self._sentry.set_tag(key, value)

        if self._config.environment:
            self._sentry.set_tag("environment", self._config.environment)

    def start_execution(
        self,
        name: str = "l0.execution",
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[], None] | None:
        """Start tracking an L0 execution.

        Args:
            name: Span name
            metadata: Additional metadata

        Returns:
            Span finish function if tracing is enabled
        """
        self._sentry.add_breadcrumb(
            type="info",
            category="l0",
            message="L0 execution started",
            data=metadata,
            level="info",
            timestamp=time.time(),
        )

        # Note: Span creation requires sentry_sdk.start_span which may not
        # be available in all versions. Return None for now.
        return None

    def start_stream(self) -> None:
        """Start tracking stream consumption."""
        self._sentry.add_breadcrumb(
            type="info",
            category="l0.stream",
            message="Stream started",
            level="info",
            timestamp=time.time(),
        )

    def record_token(self, token: str | None = None) -> None:
        """Record a token received.

        Args:
            token: Token content (optional)
        """
        if self._config.breadcrumbs_for_tokens:
            message = f"Token: {token[:50]}" if token else "Token received"
            self._sentry.add_breadcrumb(
                type="debug",
                category="l0.token",
                message=message,
                level="debug",
                timestamp=time.time(),
            )

    def record_first_token(self, ttft_ms: float) -> None:
        """Record first token (TTFT).

        Args:
            ttft_ms: Time to first token in milliseconds
        """
        self._sentry.add_breadcrumb(
            type="info",
            category="l0.stream",
            message="First token received",
            data={"ttft_ms": ttft_ms},
            level="info",
            timestamp=time.time(),
        )

    def record_network_error(
        self,
        error: Exception,
        error_type: str,
        retried: bool,
    ) -> None:
        """Record a network error.

        Args:
            error: The error
            error_type: Error type/category
            retried: Whether the request was retried
        """
        self._sentry.add_breadcrumb(
            type="error",
            category="l0.network",
            message=f"Network error: {error_type}",
            data={
                "error_type": error_type,
                "message": str(error),
                "retried": retried,
            },
            level="error",
            timestamp=time.time(),
        )

        if self._config.capture_network_errors and not retried:
            # Only capture if not retried (final failure)
            self._sentry.capture_exception(
                error,
                tags={
                    "error_type": error_type,
                    "component": "l0.network",
                },
                extra={"retried": retried},
            )

    def record_retry(
        self,
        attempt: int,
        reason: str,
        is_network_error: bool = False,
    ) -> None:
        """Record a retry attempt.

        Args:
            attempt: Attempt number
            reason: Reason for retry
            is_network_error: Whether this is a network error retry
        """
        self._sentry.add_breadcrumb(
            type="info",
            category="l0.retry",
            message=f"Retry attempt {attempt}",
            data={
                "attempt": attempt,
                "reason": reason,
                "is_network_error": is_network_error,
            },
            level="warning",
            timestamp=time.time(),
        )

    def record_guardrail_violations(
        self,
        violations: list[dict[str, Any]],
    ) -> None:
        """Record guardrail violations.

        Args:
            violations: List of violation details
        """
        severity_order = ["debug", "info", "warning", "error", "fatal"]
        min_severity = self._config.min_guardrail_severity
        min_idx = (
            severity_order.index(min_severity) if min_severity in severity_order else 0
        )

        for violation in violations:
            rule = violation.get("rule", "unknown")
            severity = violation.get("severity", "error")
            message = violation.get("message", "")
            recoverable = violation.get("recoverable", False)

            # Add breadcrumb for all violations
            self._sentry.add_breadcrumb(
                type="error",
                category="l0.guardrail",
                message=f"Guardrail violation: {rule}",
                data={
                    "rule": rule,
                    "severity": severity,
                    "message": message,
                    "recoverable": recoverable,
                },
                level=severity if severity in severity_order else "error",
                timestamp=time.time(),
            )

            # Capture as message if meets threshold
            if self._config.capture_guardrail_violations:
                severity_idx = (
                    severity_order.index(severity) if severity in severity_order else 1
                )
                if severity_idx >= min_idx:
                    self._sentry.capture_message(
                        f"Guardrail violation: {message or rule}",
                        level=severity,
                    )

    def record_drift(self, detected: bool, types: list[str]) -> None:
        """Record drift detection.

        Args:
            detected: Whether drift was detected
            types: Types of drift detected
        """
        if detected:
            self._sentry.add_breadcrumb(
                type="error",
                category="l0.drift",
                message=f"Drift detected: {', '.join(types)}",
                data={"types": types},
                level="warning",
                timestamp=time.time(),
            )

    def complete_stream(self, token_count: int) -> None:
        """Complete stream tracking.

        Args:
            token_count: Total tokens in stream
        """
        self._sentry.add_breadcrumb(
            type="info",
            category="l0.stream",
            message="Stream completed",
            data={"token_count": token_count},
            level="info",
            timestamp=time.time(),
        )

    def complete_execution(self, telemetry: Telemetry) -> None:
        """Complete execution tracking with telemetry.

        Args:
            telemetry: Final telemetry data
        """
        # Set context with telemetry data
        self._sentry.set_context(
            "l0_telemetry",
            {
                "session_id": telemetry.session_id,
                "duration_ms": (telemetry.timing.duration or 0) * 1000,
                "tokens": telemetry.metrics.token_count,
                "tokens_per_second": telemetry.metrics.tokens_per_second,
                "ttft_ms": (telemetry.metrics.time_to_first_token or 0) * 1000,
                "retries": telemetry.retries.total_retries,
                "network_errors": telemetry.retries.network_retries,
                "guardrail_violations": len(telemetry.guardrails.violations),
            },
        )

        # Add final breadcrumb
        self._sentry.add_breadcrumb(
            type="info",
            category="l0",
            message="L0 execution completed",
            data={
                "duration_ms": (telemetry.timing.duration or 0) * 1000,
                "tokens": telemetry.metrics.token_count,
                "retries": telemetry.retries.total_retries,
            },
            level="info",
            timestamp=time.time(),
        )

    def record_failure(
        self,
        error: Exception,
        telemetry: Telemetry | None = None,
    ) -> None:
        """Record execution failure.

        Args:
            error: The failure error
            telemetry: Optional telemetry context
        """
        if telemetry:
            self._sentry.set_context(
                "l0_telemetry",
                {
                    "session_id": telemetry.session_id,
                    "duration_ms": (telemetry.timing.duration or 0) * 1000,
                    "tokens": telemetry.metrics.token_count,
                    "retries": telemetry.retries.total_retries,
                    "network_errors": telemetry.retries.network_retries,
                },
            )

        self._sentry.capture_exception(
            error,
            tags={"component": "l0"},
            extra={
                "telemetry": {
                    "session_id": telemetry.session_id if telemetry else None,
                    "duration_ms": (
                        (telemetry.timing.duration or 0) * 1000 if telemetry else None
                    ),
                    "tokens": telemetry.metrics.token_count if telemetry else None,
                }
                if telemetry
                else None,
            },
        )


def create_sentry_handler(
    sentry: SentryClient,
    config: SentryConfig | None = None,
) -> Callable[[ObservabilityEvent], None]:
    """Create a Sentry event handler for L0 observability.

    This is the recommended way to integrate Sentry with L0.
    The handler subscribes to L0 events and records errors, breadcrumbs, and traces.

    Args:
        sentry: Sentry client (import sentry_sdk)
        config: Configuration options

    Returns:
        Event handler function

    Example:
        ```python
        import sentry_sdk
        from l0.monitoring import create_sentry_handler, combine_events

        sentry_sdk.init(dsn="...")

        result = await l0.run(
            stream=lambda: client.chat.completions.create(...),
            on_event=create_sentry_handler(sentry_sdk),
        )

        # Or combine with other handlers:
        result = await l0.run(
            stream=lambda: client.chat.completions.create(...),
            on_event=combine_events(
                create_sentry_handler(sentry_sdk),
                create_opentelemetry_handler(tracer),
            ),
        )
        ```
    """
    integration = Sentry(sentry, config)
    finish_span: Callable[[], None] | None = None

    def handler(event: ObservabilityEvent) -> None:
        nonlocal finish_span
        event_type = event.type
        meta = event.meta

        if event_type == ObservabilityEventType.SESSION_START:
            finish_span = integration.start_execution(
                "l0.execution",
                {
                    "attempt": meta.get("attempt"),
                    "is_retry": meta.get("is_retry"),
                    "is_fallback": meta.get("is_fallback"),
                },
            )
            integration.start_stream()

        elif event_type == ObservabilityEventType.RETRY_ATTEMPT:
            integration.record_retry(
                attempt=meta.get("attempt", 1),
                reason=meta.get("reason", "unknown"),
                is_network_error=meta.get("is_network", False),
            )

        elif event_type == ObservabilityEventType.ERROR:
            error_msg = meta.get("error", "Error")
            error = (
                error_msg
                if isinstance(error_msg, Exception)
                else Exception(str(error_msg))
            )
            integration.record_network_error(
                error=error,
                error_type=meta.get("failure_type", "unknown"),
                retried=meta.get("recovery_strategy") == "retry",
            )

        elif event_type == ObservabilityEventType.GUARDRAIL_RULE_RESULT:
            violations = meta.get("violations", [])
            if violations:
                integration.record_guardrail_violations(violations)

        elif event_type == ObservabilityEventType.DRIFT_CHECK_RESULT:
            if meta.get("detected"):
                metrics = meta.get("metrics", {})
                drift_types = list(metrics.keys()) if metrics else []
                integration.record_drift(True, drift_types)

        elif event_type == ObservabilityEventType.COMPLETE:
            token_count = meta.get("token_count", 0)
            if hasattr(meta.get("state"), "token_count"):
                token_count = meta["state"].token_count
            integration.complete_stream(token_count)
            if finish_span:
                finish_span()
                finish_span = None

    return handler


async def with_sentry(
    sentry: SentryClient,
    fn: Callable[[], Any],
    config: SentryConfig | None = None,
) -> Any:
    """Wrap L0 execution with Sentry tracking.

    Args:
        sentry: Sentry client
        fn: Async function to wrap
        config: Configuration options

    Returns:
        Result of fn()

    Example:
        ```python
        import sentry_sdk
        from l0.monitoring import with_sentry

        result = await with_sentry(
            sentry_sdk,
            lambda: l0.run(
                stream=lambda: client.chat.completions.create(...),
                monitoring={"enabled": True},
            ),
        )
        ```
    """
    integration = Sentry(sentry, config)
    integration.start_execution()

    try:
        result = await fn()

        if hasattr(result, "telemetry") and result.telemetry:
            integration.complete_execution(result.telemetry)

        return result
    except Exception as error:
        integration.record_failure(error)
        raise


class SentryExporterConfig(BaseModel):
    """Sentry exporter configuration.

    Usage:
        ```python
        from l0.monitoring import SentryExporterConfig, SentryExporter

        config = SentryExporterConfig(
            dsn="https://xxx@sentry.io/123",
            environment="production",
        )

        exporter = SentryExporter(config)
        exporter.capture_error(error, telemetry)
        ```

    Attributes:
        dsn: Sentry DSN (Data Source Name)
        environment: Environment name (production, staging, etc.)
        release: Release/version identifier
        server_name: Server name for grouping
        sample_rate: Error sample rate (0.0 to 1.0)
        traces_sample_rate: Transaction sample rate for performance
        profiles_sample_rate: Profile sample rate
        enabled: Enable/disable Sentry
        debug: Enable Sentry debug mode
        attach_stacktrace: Attach stacktrace to all events
        send_default_pii: Send personally identifiable information
        max_breadcrumbs: Maximum number of breadcrumbs
        tags: Default tags for all events
    """

    dsn: str | None = None
    environment: str | None = None
    release: str | None = None
    server_name: str | None = None
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    profiles_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    enabled: bool = True
    debug: bool = False
    attach_stacktrace: bool = True
    send_default_pii: bool = False
    max_breadcrumbs: int = Field(default=100, ge=0)
    tags: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_env(cls) -> SentryExporterConfig:
        """Create config from environment variables.

        Reads:
            - SENTRY_DSN
            - SENTRY_ENVIRONMENT
            - SENTRY_RELEASE

        Returns:
            SentryExporterConfig from environment
        """
        import os

        return cls(
            dsn=os.getenv("SENTRY_DSN"),
            environment=os.getenv("SENTRY_ENVIRONMENT"),
            release=os.getenv("SENTRY_RELEASE"),
        )


class SentryExporter:
    """Export L0 errors and telemetry to Sentry.

    Usage:
        ```python
        from l0.monitoring import SentryExporterConfig, SentryExporter

        config = SentryExporterConfig(
            dsn="https://xxx@sentry.io/123",
            environment="production",
        )

        exporter = SentryExporter(config)

        # Initialize Sentry
        exporter.init()

        # Capture error with telemetry context
        try:
            result = await l0.run(...)
        except Exception as e:
            exporter.capture_error(e, monitor.get_telemetry())

        # Or use as Monitor callback
        def on_complete(telemetry: Telemetry) -> None:
            if telemetry.error.occurred:
                exporter.capture_telemetry_error(telemetry)

        monitor = Monitor(on_complete=on_complete)
        ```

    Requires:
        pip install sentry-sdk
    """

    def __init__(self, config: SentryExporterConfig) -> None:
        """Initialize Sentry exporter.

        Args:
            config: Sentry configuration
        """
        self.config = config
        self._initialized = False

    def init(self) -> None:
        """Initialize Sentry SDK.

        Call this once at application startup.
        """
        if self._initialized or not self.config.enabled or not self.config.dsn:
            return

        try:
            import sentry_sdk
        except ImportError as e:
            raise ImportError(
                "Sentry SDK not installed. Install with: pip install ai2070-l0[observability]"
            ) from e

        sentry_sdk.init(
            dsn=self.config.dsn,
            environment=self.config.environment,
            release=self.config.release,
            server_name=self.config.server_name,
            sample_rate=self.config.sample_rate,
            traces_sample_rate=self.config.traces_sample_rate,
            profiles_sample_rate=self.config.profiles_sample_rate,
            debug=self.config.debug,
            attach_stacktrace=self.config.attach_stacktrace,
            send_default_pii=self.config.send_default_pii,
            max_breadcrumbs=self.config.max_breadcrumbs,
        )

        # Set default tags
        for key, value in self.config.tags.items():
            sentry_sdk.set_tag(key, value)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure Sentry is initialized."""
        if not self._initialized:
            self.init()

    def capture_error(
        self,
        error: Exception,
        telemetry: Telemetry | None = None,
        **extra: Any,
    ) -> str | None:
        """Capture an error with optional telemetry context.

        Args:
            error: The exception to capture
            telemetry: Optional telemetry for context
            **extra: Additional context data

        Returns:
            Sentry event ID, or None if not sent
        """
        if not self.config.enabled or not self.config.dsn:
            return None

        self._ensure_initialized()

        try:
            import sentry_sdk
        except ImportError:
            return None

        with sentry_sdk.push_scope() as scope:
            # Add telemetry context
            if telemetry:
                self._add_telemetry_context(scope, telemetry)

            # Add extra context
            for key, value in extra.items():
                scope.set_extra(key, value)

            return cast(str | None, sentry_sdk.capture_exception(error))

    def capture_telemetry_error(self, telemetry: Telemetry) -> str | None:
        """Capture an error from telemetry data.

        Use this when you have telemetry with error info but no exception object.

        Args:
            telemetry: Telemetry with error information

        Returns:
            Sentry event ID, or None if not sent
        """
        if not self.config.enabled or not self.config.dsn:
            return None

        if not telemetry.error.occurred:
            return None

        self._ensure_initialized()

        try:
            import sentry_sdk
        except ImportError:
            return None

        with sentry_sdk.push_scope() as scope:
            self._add_telemetry_context(scope, telemetry)

            return cast(
                str | None,
                sentry_sdk.capture_message(
                    telemetry.error.message or "Unknown L0 error",
                    level="error",
                ),
            )

    def capture_message(
        self,
        message: str,
        level: str = "info",
        telemetry: Telemetry | None = None,
        **extra: Any,
    ) -> str | None:
        """Capture a message with optional telemetry context.

        Args:
            message: Message to capture
            level: Log level (debug, info, warning, error, fatal)
            telemetry: Optional telemetry for context
            **extra: Additional context data

        Returns:
            Sentry event ID, or None if not sent
        """
        if not self.config.enabled or not self.config.dsn:
            return None

        self._ensure_initialized()

        try:
            import sentry_sdk
        except ImportError:
            return None

        with sentry_sdk.push_scope() as scope:
            if telemetry:
                self._add_telemetry_context(scope, telemetry)

            for key, value in extra.items():
                scope.set_extra(key, value)

            return cast(str | None, sentry_sdk.capture_message(message, level=level))

    def add_breadcrumb(
        self,
        message: str,
        category: str = "l0",
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add a breadcrumb for debugging.

        Args:
            message: Breadcrumb message
            category: Category for grouping
            level: Log level
            data: Additional data
        """
        if not self.config.enabled or not self.config.dsn:
            return

        self._ensure_initialized()

        try:
            import sentry_sdk

            sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level=level,
                data=data,
            )
        except ImportError:
            pass

    def set_user(
        self,
        user_id: str | None = None,
        email: str | None = None,
        username: str | None = None,
        **extra: Any,
    ) -> None:
        """Set user context for error tracking.

        Args:
            user_id: User ID
            email: User email
            username: Username
            **extra: Additional user attributes
        """
        if not self.config.enabled or not self.config.dsn:
            return

        self._ensure_initialized()

        try:
            import sentry_sdk

            user_data: dict[str, Any] = {}
            if user_id:
                user_data["id"] = user_id
            if email:
                user_data["email"] = email
            if username:
                user_data["username"] = username
            user_data.update(extra)

            sentry_sdk.set_user(user_data)
        except ImportError:
            pass

    def _add_telemetry_context(self, scope: Any, telemetry: Telemetry) -> None:
        """Add telemetry data to Sentry scope.

        Args:
            scope: Sentry scope
            telemetry: Telemetry data
        """
        # Tags for filtering
        scope.set_tag("l0.stream_id", telemetry.stream_id)
        if telemetry.model:
            scope.set_tag("l0.model", telemetry.model)
        if telemetry.session_id:
            scope.set_tag("l0.session_id", telemetry.session_id)
        if telemetry.error.category:
            scope.set_tag("l0.error.category", telemetry.error.category.value)

        # Context for details
        scope.set_context(
            "l0_timing",
            {
                "duration": telemetry.timing.duration,
                "time_to_first_token": telemetry.metrics.time_to_first_token,
                "started_at": telemetry.timing.started_at.isoformat()
                if telemetry.timing.started_at
                else None,
                "completed_at": telemetry.timing.completed_at.isoformat()
                if telemetry.timing.completed_at
                else None,
            },
        )

        scope.set_context(
            "l0_metrics",
            {
                "token_count": telemetry.metrics.token_count,
                "tokens_per_second": telemetry.metrics.tokens_per_second,
                "content_length": telemetry.content_length,
            },
        )

        scope.set_context(
            "l0_retries",
            {
                "total": telemetry.retries.total_retries,
                "model": telemetry.retries.model_retries,
                "network": telemetry.retries.network_retries,
                "last_error": telemetry.retries.last_error,
            },
        )

        if telemetry.guardrails.violations:
            scope.set_context(
                "l0_guardrails",
                {
                    "rules_checked": telemetry.guardrails.rules_checked,
                    "violations": telemetry.guardrails.violations,
                    "passed": telemetry.guardrails.passed,
                },
            )

        if telemetry.error.occurred:
            scope.set_context(
                "l0_error",
                {
                    "message": telemetry.error.message,
                    "category": telemetry.error.category.value
                    if telemetry.error.category
                    else None,
                    "code": telemetry.error.code,
                    "recoverable": telemetry.error.recoverable,
                },
            )

        # Add metadata
        if telemetry.metadata:
            scope.set_context("l0_metadata", telemetry.metadata)

    def flush(self, timeout: float = 2.0) -> None:
        """Flush pending events to Sentry.

        Args:
            timeout: Timeout in seconds
        """
        if not self._initialized:
            return

        try:
            import sentry_sdk

            sentry_sdk.flush(timeout=timeout)
        except ImportError:
            pass

    def close(self) -> None:
        """Close Sentry client."""
        if not self._initialized:
            return

        try:
            import sentry_sdk

            client = sentry_sdk.get_client()
            if client:
                client.close()
        except ImportError:
            pass

        self._initialized = False
