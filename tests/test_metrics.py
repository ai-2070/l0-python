"""Tests for Metrics."""

import pytest

from l0 import (
    Metrics,
    MetricsSnapshot,
    create_metrics,
    get_global_metrics,
    reset_global_metrics,
)


class TestMetrics:
    """Tests for Metrics class."""

    def test_initial_values(self) -> None:
        m = Metrics()
        assert m.requests == 0
        assert m.tokens == 0
        assert m.retries == 0
        assert m.network_retry_count == 0
        assert m.errors == 0
        assert m.violations == 0
        assert m.drift_detections == 0
        assert m.fallbacks == 0
        assert m.completions == 0
        assert m.timeouts == 0

    def test_increment_counters(self) -> None:
        m = Metrics()
        m.requests += 1
        m.tokens += 150
        m.retries += 2
        m.completions += 1

        assert m.requests == 1
        assert m.tokens == 150
        assert m.retries == 2
        assert m.completions == 1

    def test_reset(self) -> None:
        m = Metrics()
        m.requests = 10
        m.tokens = 1000
        m.errors = 5

        m.reset()

        assert m.requests == 0
        assert m.tokens == 0
        assert m.errors == 0

    def test_snapshot(self) -> None:
        m = Metrics()
        m.requests = 5
        m.tokens = 500
        m.completions = 4
        m.errors = 1

        snap = m.snapshot()

        assert isinstance(snap, MetricsSnapshot)
        assert snap.requests == 5
        assert snap.tokens == 500
        assert snap.completions == 4
        assert snap.errors == 1

    def test_snapshot_is_copy(self) -> None:
        m = Metrics()
        m.requests = 5

        snap = m.snapshot()
        m.requests = 10

        assert snap.requests == 5  # Snapshot unchanged
        assert m.requests == 10

    def test_to_dict(self) -> None:
        m = Metrics()
        m.requests = 3
        m.tokens = 300

        d = m.to_dict()

        assert isinstance(d, dict)
        assert d["requests"] == 3
        assert d["tokens"] == 300
        assert "completions" in d
        assert "errors" in d

    def test_repr(self) -> None:
        m = Metrics()
        m.requests = 1
        m.tokens = 100
        m.completions = 1

        r = repr(m)

        assert "Metrics" in r
        assert "requests=1" in r
        assert "tokens=100" in r
        assert "completions=1" in r

    def test_create_class_method(self) -> None:
        m = Metrics.create()
        assert isinstance(m, Metrics)
        assert m.requests == 0

    def test_get_global(self) -> None:
        # Reset first to ensure clean state
        Metrics.reset_global()

        g1 = Metrics.get_global()
        g2 = Metrics.get_global()

        assert g1 is g2  # Same instance

    def test_get_global_persists(self) -> None:
        Metrics.reset_global()

        g = Metrics.get_global()
        g.requests = 42

        g2 = Metrics.get_global()
        assert g2.requests == 42

    def test_reset_global(self) -> None:
        g = Metrics.get_global()
        g.requests = 100
        g.tokens = 5000

        Metrics.reset_global()

        g2 = Metrics.get_global()
        assert g2.requests == 0
        assert g2.tokens == 0


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_fields(self) -> None:
        snap = MetricsSnapshot(
            requests=1,
            tokens=100,
            retries=2,
            network_retry_count=1,
            errors=0,
            violations=0,
            drift_detections=0,
            fallbacks=0,
            completions=1,
            timeouts=0,
        )

        assert snap.requests == 1
        assert snap.tokens == 100
        assert snap.retries == 2
        assert snap.network_retry_count == 1
        assert snap.completions == 1


class TestLegacyFunctions:
    """Tests for legacy standalone functions."""

    def test_create_metrics(self) -> None:
        m = create_metrics()
        assert isinstance(m, Metrics)

    def test_get_global_metrics(self) -> None:
        reset_global_metrics()
        g = get_global_metrics()
        assert isinstance(g, Metrics)

    def test_reset_global_metrics(self) -> None:
        g = get_global_metrics()
        g.requests = 50

        reset_global_metrics()

        assert get_global_metrics().requests == 0
