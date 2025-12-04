"""Sentry integration for L0 monitoring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .telemetry import Telemetry

if TYPE_CHECKING:
    pass


class SentryConfig(BaseModel):
    """Sentry configuration.

    Usage:
        ```python
        from l0.monitoring import SentryConfig, SentryExporter

        config = SentryConfig(
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
    def from_env(cls) -> SentryConfig:
        """Create config from environment variables.

        Reads:
            - SENTRY_DSN
            - SENTRY_ENVIRONMENT
            - SENTRY_RELEASE

        Returns:
            SentryConfig from environment
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
        from l0.monitoring import SentryConfig, SentryExporter

        config = SentryConfig(
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

    def __init__(self, config: SentryConfig) -> None:
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
                "Sentry SDK not installed. Install with: pip install l0[observability]"
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

            return sentry_sdk.capture_exception(error)

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

            return sentry_sdk.capture_message(
                telemetry.error.message or "Unknown L0 error",
                level="error",
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

            return sentry_sdk.capture_message(message, level=level)

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
