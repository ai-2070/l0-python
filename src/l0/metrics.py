"""Simple metrics collection for L0 runtime.

Lightweight counters - OpenTelemetry is opt-in via adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetricsSnapshot:
    """Snapshot of all metrics at a point in time."""

    requests: int
    tokens: int
    retries: int
    network_retry_count: int
    errors: int
    violations: int
    drift_detections: int
    fallbacks: int
    completions: int
    timeouts: int


class Metrics:
    """Simple metrics for L0 runtime.

    Just counters - no histograms, no complex aggregations.
    OpenTelemetry integration is separate and optional.

    Usage:
        from l0 import Metrics

        # Create a new instance
        metrics = Metrics()

        # Increment counters
        metrics.requests += 1
        metrics.tokens += 150
        metrics.completions += 1

        # Get snapshot
        snap = metrics.snapshot()
        print(f"Total tokens: {snap.tokens}")

        # Reset all counters
        metrics.reset()

        # Use global metrics (scoped API)
        global_metrics = Metrics.get_global()
        global_metrics.requests += 1
        Metrics.reset_global()
    """

    # Global metrics instance (singleton)
    _global_instance: "Metrics | None" = None

    __slots__ = (
        "requests",
        "tokens",
        "retries",
        "network_retry_count",
        "errors",
        "violations",
        "drift_detections",
        "fallbacks",
        "completions",
        "timeouts",
    )

    def __init__(self) -> None:
        """Initialize all counters to zero."""
        self.requests: int = 0
        """Total stream requests."""

        self.tokens: int = 0
        """Total tokens processed."""

        self.retries: int = 0
        """Total retry attempts."""

        self.network_retry_count: int = 0
        """Network retries (subset of retries)."""

        self.errors: int = 0
        """Total errors encountered."""

        self.violations: int = 0
        """Guardrail violations."""

        self.drift_detections: int = 0
        """Drift detections."""

        self.fallbacks: int = 0
        """Fallback activations."""

        self.completions: int = 0
        """Successful completions."""

        self.timeouts: int = 0
        """Timeouts (initial + inter-token)."""

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.requests = 0
        self.tokens = 0
        self.retries = 0
        self.network_retry_count = 0
        self.errors = 0
        self.violations = 0
        self.drift_detections = 0
        self.fallbacks = 0
        self.completions = 0
        self.timeouts = 0

    def snapshot(self) -> MetricsSnapshot:
        """Get snapshot of all metrics.

        Returns:
            MetricsSnapshot with current counter values.
        """
        return MetricsSnapshot(
            requests=self.requests,
            tokens=self.tokens,
            retries=self.retries,
            network_retry_count=self.network_retry_count,
            errors=self.errors,
            violations=self.violations,
            drift_detections=self.drift_detections,
            fallbacks=self.fallbacks,
            completions=self.completions,
            timeouts=self.timeouts,
        )

    def to_dict(self) -> dict[str, int]:
        """Serialize metrics to dictionary.

        Returns:
            Dictionary with all counter values.
        """
        return {
            "requests": self.requests,
            "tokens": self.tokens,
            "retries": self.retries,
            "network_retry_count": self.network_retry_count,
            "errors": self.errors,
            "violations": self.violations,
            "drift_detections": self.drift_detections,
            "fallbacks": self.fallbacks,
            "completions": self.completions,
            "timeouts": self.timeouts,
        }

    def __repr__(self) -> str:
        return (
            f"Metrics(requests={self.requests}, tokens={self.tokens}, "
            f"retries={self.retries}, errors={self.errors}, "
            f"completions={self.completions})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Scoped API (class methods)
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def create(cls) -> "Metrics":
        """Create a new metrics instance.

        Returns:
            A new Metrics instance with all counters at zero.
        """
        return cls()

    @classmethod
    def get_global(cls) -> "Metrics":
        """Get or create global metrics instance.

        Returns:
            The global Metrics singleton instance.
        """
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def reset_global(cls) -> None:
        """Reset global metrics counters to zero."""
        if cls._global_instance is not None:
            cls._global_instance.reset()


# Legacy standalone functions (for backwards compatibility)
def create_metrics() -> Metrics:
    """Create a new metrics instance.

    Deprecated: Use Metrics.create() instead.
    """
    return Metrics.create()


def get_global_metrics() -> Metrics:
    """Get or create global metrics instance.

    Deprecated: Use Metrics.get_global() instead.
    """
    return Metrics.get_global()


def reset_global_metrics() -> None:
    """Reset global metrics counters to zero.

    Deprecated: Use Metrics.reset_global() instead.
    """
    Metrics.reset_global()
