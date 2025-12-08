"""
L0 Performance Benchmark Suite

Tests L0 layer performance overhead with ms-level precision and tokens/s throughput metrics.
Designed to measure the cost of the reliability substrate, not LLM inference.

Simulates high-throughput scenarios (1000+ tokens/s) expected from Nvidia Blackwell.

Scenarios tested:
- Baseline: Raw async iteration without L0
- L0 Core: Minimal L0 wrapper (no guardrails)
- L0 + Guardrails: With JSON/markdown validation
- L0 + Drift Detection: With drift analysis enabled
- L0 + Full Stack: All features enabled
"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from l0.adapters import AdaptedEvent, Adapters
from l0.drift import DriftDetector
from l0.guardrails import (
    GuardrailRule,
    GuardrailViolation,
    json_rule,
    markdown_rule,
    pattern_rule,
    zero_output_rule,
)
from l0.runtime import _internal_run
from l0.types import CheckIntervals, Event, EventType, State

# ============================================================================
# High-Precision Timer
# ============================================================================


@dataclass
class TimingResult:
    """High-precision timing result."""

    start_time: float
    end_time: float
    duration_ms: float
    duration_ns: int


class Timer:
    """High-precision timer using time.perf_counter_ns."""

    def __init__(self) -> None:
        self._start_ns = time.perf_counter_ns()
        self._start_time = time.perf_counter()

    def stop(self) -> TimingResult:
        end_time = time.perf_counter()
        end_ns = time.perf_counter_ns()
        duration_ns = end_ns - self._start_ns
        duration_ms = duration_ns / 1_000_000

        return TimingResult(
            start_time=self._start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            duration_ns=duration_ns,
        )


# ============================================================================
# Mock Stream Generator
# ============================================================================


@dataclass
class MockStreamConfig:
    """Configuration for mock token stream."""

    token_count: int
    avg_token_size: int = 4
    inter_token_delay_ms: float = 0
    content_type: str = "text"  # "text" | "json" | "markdown"
    realistic: bool = True


def generate_token_content(index: int, config: MockStreamConfig) -> str:
    """Generate realistic token content based on content type."""
    if not config.realistic:
        return "x" * config.avg_token_size

    if config.content_type == "json":
        return _generate_json_token(index, config.token_count)
    elif config.content_type == "markdown":
        return _generate_markdown_token(index, config.token_count)
    else:
        return _generate_text_token(index, config.avg_token_size)


def _generate_json_token(index: int, total: int) -> str:
    """Generate valid JSON structure progressively."""
    if index == 0:
        return "{"
    if index == total - 1:
        return "}"
    if index == 1:
        return '"data": ['
    if index == total - 2:
        return "]"

    item_index = index - 2
    if item_index % 4 == 0:
        return '{"id": '
    if item_index % 4 == 1:
        return str(item_index)
    if item_index % 4 == 2:
        return ', "value": "item"'
    return "}, "


def _generate_markdown_token(index: int, total: int) -> str:
    """Generate markdown content."""
    patterns = [
        "# ",
        "Heading\n\n",
        "This ",
        "is ",
        "a ",
        "paragraph ",
        "with ",
        "**bold** ",
        "and ",
        "_italic_ ",
        "text.\n\n",
        "- ",
        "List ",
        "item\n",
        "```\n",
        "code ",
        "block\n",
        "```\n",
    ]
    return patterns[index % len(patterns)]


def _generate_text_token(index: int, avg_size: int) -> str:
    """Generate text tokens."""
    words = [
        "the ",
        "quick ",
        "brown ",
        "fox ",
        "jumps ",
        "over ",
        "lazy ",
        "dog ",
        "and ",
        "runs ",
        "through ",
        "forest ",
        "while ",
        "birds ",
        "sing ",
        "songs ",
    ]
    return words[index % len(words)]


async def create_mock_token_stream(
    config: MockStreamConfig,
) -> AsyncIterator[Event]:
    """Create a mock async iterable stream that simulates LLM token streaming."""
    for i in range(config.token_count):
        if config.inter_token_delay_ms > 0:
            await asyncio.sleep(config.inter_token_delay_ms / 1000)

        yield Event(
            type=EventType.TOKEN,
            text=generate_token_content(i, config),
            timestamp=time.time(),
        )

    yield Event(type=EventType.COMPLETE, timestamp=time.time())


def create_mock_stream_factory(config: MockStreamConfig):
    """Create a stream factory for l0 consumption."""
    return lambda: create_mock_token_stream(config)


# ============================================================================
# Test Adapter for Benchmarks
# ============================================================================


class BenchmarkAdapter:
    """Fast adapter for benchmark tests - minimal overhead."""

    name = "benchmark"

    def detect(self, stream: Any) -> bool:
        """Detect async generators."""
        return hasattr(stream, "__anext__")

    async def wrap(
        self, stream: Any, options: Any = None
    ) -> AsyncIterator[AdaptedEvent[Any]]:
        """Pass through events with minimal wrapping."""
        async for event in stream:
            yield AdaptedEvent(event=event, raw_chunk=None)


# ============================================================================
# Benchmark Metrics
# ============================================================================


@dataclass
class BenchmarkMetrics:
    """Benchmark result metrics."""

    scenario: str
    token_count: int
    duration_ms: float
    tokens_per_second: float
    avg_token_time_us: float
    time_to_first_token_ms: float
    memory_delta_bytes: int
    overhead_percent: float | None = None


@dataclass
class BenchmarkRun:
    """Single benchmark run result."""

    metrics: BenchmarkMetrics
    raw_timing: TimingResult
    events: list[Event]
    state: State | None


@dataclass
class BenchmarkSuiteResult:
    """Result from running benchmark suite."""

    runs: list[BenchmarkRun]
    avg: BenchmarkMetrics
    min: BenchmarkMetrics
    max: BenchmarkMetrics
    std_dev: float


# ============================================================================
# Benchmark Runners
# ============================================================================


async def run_benchmark(
    scenario: str,
    stream_config: MockStreamConfig,
    guardrails: list[GuardrailRule] | None = None,
    detect_drift: bool = False,
    check_intervals: CheckIntervals | None = None,
) -> BenchmarkRun:
    """Run a single benchmark iteration with L0."""
    events: list[Event] = []
    first_token_time: float | None = None

    # Force GC for more accurate memory measurement
    gc.collect()
    mem_before = _get_memory_usage()

    timer = Timer()
    start_time = time.perf_counter()

    # Create drift detector if needed
    drift_detector = DriftDetector() if detect_drift else None

    result = await _internal_run(
        stream=create_mock_stream_factory(stream_config),
        guardrails=guardrails or [],
        drift_detector=drift_detector,
        check_intervals=check_intervals,
    )

    async for event in result:
        if event.is_token and first_token_time is None:
            first_token_time = time.perf_counter()
        events.append(event)

    timing = timer.stop()

    gc.collect()
    mem_after = _get_memory_usage()

    token_events = [e for e in events if e.is_token]
    tokens_per_second = (
        (len(token_events) / timing.duration_ms) * 1000 if timing.duration_ms > 0 else 0
    )

    return BenchmarkRun(
        metrics=BenchmarkMetrics(
            scenario=scenario,
            token_count=len(token_events),
            duration_ms=timing.duration_ms,
            tokens_per_second=tokens_per_second,
            avg_token_time_us=(
                (timing.duration_ms / len(token_events)) * 1000
                if len(token_events) > 0
                else 0
            ),
            time_to_first_token_ms=(
                (first_token_time - start_time) * 1000
                if first_token_time is not None
                else 0
            ),
            memory_delta_bytes=mem_after - mem_before,
        ),
        raw_timing=timing,
        events=events,
        state=result.state if hasattr(result, "state") else None,
    )


async def run_baseline_benchmark(
    scenario: str,
    stream_config: MockStreamConfig,
) -> BenchmarkRun:
    """Run baseline benchmark (raw async iteration without L0)."""
    events: list[Event] = []
    first_token_time: float | None = None

    gc.collect()
    mem_before = _get_memory_usage()

    timer = Timer()
    start_time = time.perf_counter()

    async for event in create_mock_token_stream(stream_config):
        if event.is_token and first_token_time is None:
            first_token_time = time.perf_counter()
        events.append(event)

    timing = timer.stop()

    gc.collect()
    mem_after = _get_memory_usage()

    token_events = [e for e in events if e.is_token]
    tokens_per_second = (
        (len(token_events) / timing.duration_ms) * 1000 if timing.duration_ms > 0 else 0
    )

    return BenchmarkRun(
        metrics=BenchmarkMetrics(
            scenario=scenario,
            token_count=len(token_events),
            duration_ms=timing.duration_ms,
            tokens_per_second=tokens_per_second,
            avg_token_time_us=(
                (timing.duration_ms / len(token_events)) * 1000
                if len(token_events) > 0
                else 0
            ),
            time_to_first_token_ms=(
                (first_token_time - start_time) * 1000
                if first_token_time is not None
                else 0
            ),
            memory_delta_bytes=mem_after - mem_before,
        ),
        raw_timing=timing,
        events=events,
        state=None,
    )


async def run_benchmark_suite(
    scenario: str,
    stream_config: MockStreamConfig,
    guardrails: list[GuardrailRule] | None = None,
    detect_drift: bool = False,
    check_intervals: CheckIntervals | None = None,
    iterations: int = 5,
    is_baseline: bool = False,
) -> BenchmarkSuiteResult:
    """Run multiple iterations and compute statistics."""
    runs: list[BenchmarkRun] = []

    # Warm-up run (discarded)
    if is_baseline:
        await run_baseline_benchmark(scenario, stream_config)
    else:
        await run_benchmark(
            scenario, stream_config, guardrails, detect_drift, check_intervals
        )

    # Actual benchmark runs
    for _ in range(iterations):
        if is_baseline:
            run = await run_baseline_benchmark(scenario, stream_config)
        else:
            run = await run_benchmark(
                scenario, stream_config, guardrails, detect_drift, check_intervals
            )
        runs.append(run)

    # Calculate statistics
    durations = [r.metrics.duration_ms for r in runs]
    avg_duration = sum(durations) / len(durations)
    variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
    std_dev = variance**0.5

    avg_tokens_per_second = sum(r.metrics.tokens_per_second for r in runs) / len(runs)
    avg_time_to_first_token = sum(r.metrics.time_to_first_token_ms for r in runs) / len(
        runs
    )

    sorted_by_duration = sorted(runs, key=lambda r: r.metrics.duration_ms)

    return BenchmarkSuiteResult(
        runs=runs,
        avg=BenchmarkMetrics(
            scenario=scenario,
            token_count=stream_config.token_count,
            duration_ms=avg_duration,
            tokens_per_second=avg_tokens_per_second,
            avg_token_time_us=(
                (avg_duration / stream_config.token_count) * 1000
                if stream_config.token_count > 0
                else 0
            ),
            time_to_first_token_ms=avg_time_to_first_token,
            memory_delta_bytes=int(
                sum(r.metrics.memory_delta_bytes for r in runs) / len(runs)
            ),
        ),
        min=sorted_by_duration[0].metrics,
        max=sorted_by_duration[-1].metrics,
        std_dev=std_dev,
    )


def _get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    import sys

    # Use sys.getsizeof on key objects or tracemalloc if available
    try:
        import tracemalloc

        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            return current
    except ImportError:
        pass

    # Fallback: use a rough estimate
    return 0


# ============================================================================
# Benchmark Report
# ============================================================================


def format_report(
    scenarios: dict[str, BenchmarkSuiteResult],
    baseline: BenchmarkMetrics,
) -> str:
    """Format benchmark results as a report."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("L0 PERFORMANCE BENCHMARK REPORT (Python)")
    lines.append("=" * 80)
    lines.append("")

    # Header
    lines.append(
        "| Scenario                    | Tokens/s   | Avg (ms)  | TTFT (ms) | Overhead |"
    )
    lines.append(
        "|-----------------------------|------------|-----------|-----------|----------|"
    )

    # Baseline first
    lines.append(
        f"| {baseline.scenario:<27} | {baseline.tokens_per_second:>10.0f} | "
        f"{baseline.duration_ms:>9.2f} | {baseline.time_to_first_token_ms:>9.2f} | baseline |"
    )

    # Other scenarios
    for name, data in scenarios.items():
        if name == "Baseline":
            continue

        overhead = (
            ((data.avg.duration_ms - baseline.duration_ms) / baseline.duration_ms) * 100
            if baseline.duration_ms > 0
            else 0
        )

        lines.append(
            f"| {name:<27} | {data.avg.tokens_per_second:>10.0f} | "
            f"{data.avg.duration_ms:>9.2f} | {data.avg.time_to_first_token_ms:>9.2f} | "
            f"{overhead:>6.1f}% |"
        )

    lines.append("")
    lines.append("Legend:")
    lines.append("  Tokens/s  = Throughput (higher is better)")
    lines.append("  Avg (ms)  = Average total duration (lower is better)")
    lines.append("  TTFT (ms) = Time to first token (lower is better)")
    lines.append("  Overhead  = % slower than baseline (lower is better)")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def register_benchmark_adapter():
    """Register and cleanup the benchmark adapter for tests."""
    Adapters.register(BenchmarkAdapter())
    yield
    Adapters.reset()


# ============================================================================
# Test Configurations
# ============================================================================

CONFIGS = {
    "standard": MockStreamConfig(
        token_count=500,
        avg_token_size=4,
        inter_token_delay_ms=0,
        content_type="text",
        realistic=True,
    ),
    "high_throughput": MockStreamConfig(
        token_count=2000,
        avg_token_size=4,
        inter_token_delay_ms=0,
        content_type="text",
        realistic=True,
    ),
    "stress": MockStreamConfig(
        token_count=10000,
        avg_token_size=4,
        inter_token_delay_ms=0,
        content_type="text",
        realistic=False,
    ),
    "json": MockStreamConfig(
        token_count=1000,
        avg_token_size=4,
        inter_token_delay_ms=0,
        content_type="json",
        realistic=True,
    ),
    "markdown": MockStreamConfig(
        token_count=1000,
        avg_token_size=6,
        inter_token_delay_ms=0,
        content_type="markdown",
        realistic=True,
    ),
}


# ============================================================================
# Test Suites
# ============================================================================


class TestBaselineVsL0Core:
    """Baseline vs L0 Core benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_baseline_raw_streaming_performance(self):
        """Measure baseline raw streaming performance."""
        result = await run_benchmark_suite(
            "Baseline",
            CONFIGS["high_throughput"],
            iterations=3,
            is_baseline=True,
        )

        assert result.avg.token_count == CONFIGS["high_throughput"].token_count
        assert result.avg.tokens_per_second > 0

        print(f"\nBaseline: {result.avg.tokens_per_second:.0f} tokens/s")
        print(f"  Duration: {result.avg.duration_ms:.2f} ms")
        print(f"  Std Dev: {result.std_dev:.2f} ms")

    @pytest.mark.asyncio
    async def test_measure_l0_core_overhead(self):
        """Measure L0 core overhead (no guardrails)."""
        baseline = await run_benchmark_suite(
            "Baseline",
            CONFIGS["high_throughput"],
            iterations=3,
            is_baseline=True,
        )

        l0_core = await run_benchmark_suite(
            "L0 Core",
            CONFIGS["high_throughput"],
            guardrails=[],
            detect_drift=False,
            iterations=3,
        )

        overhead = (
            (
                (l0_core.avg.duration_ms - baseline.avg.duration_ms)
                / baseline.avg.duration_ms
            )
            * 100
            if baseline.avg.duration_ms > 0
            else 0
        )

        print(f"\nL0 Core overhead: {overhead:.1f}%")
        print(f"  Baseline: {baseline.avg.tokens_per_second:.0f} tokens/s")
        print(f"  L0 Core: {l0_core.avg.tokens_per_second:.0f} tokens/s")

        # L0 core should still achieve high throughput
        assert l0_core.avg.tokens_per_second > 10000


class TestGuardrailsPerformance:
    """Guardrails performance impact benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_json_guardrail_overhead(self):
        """Measure JSON guardrail overhead."""
        no_guardrails = await run_benchmark_suite(
            "No Guardrails",
            CONFIGS["json"],
            guardrails=[],
            iterations=3,
        )

        with_json = await run_benchmark_suite(
            "JSON Guardrail",
            CONFIGS["json"],
            guardrails=[json_rule()],
            iterations=3,
        )

        overhead = (
            (
                (with_json.avg.duration_ms - no_guardrails.avg.duration_ms)
                / no_guardrails.avg.duration_ms
            )
            * 100
            if no_guardrails.avg.duration_ms > 0
            else 0
        )

        print(f"\nJSON Guardrail overhead: {overhead:.1f}%")
        print(f"  Without: {no_guardrails.avg.tokens_per_second:.0f} tokens/s")
        print(f"  With JSON: {with_json.avg.tokens_per_second:.0f} tokens/s")

        assert with_json.avg.token_count == CONFIGS["json"].token_count

    @pytest.mark.asyncio
    async def test_measure_multiple_guardrails_overhead(self):
        """Measure multiple guardrails overhead."""
        no_guardrails = await run_benchmark_suite(
            "No Guardrails",
            CONFIGS["standard"],
            guardrails=[],
            iterations=3,
        )

        with_multiple = await run_benchmark_suite(
            "Multiple Guardrails",
            CONFIGS["standard"],
            guardrails=[json_rule(), markdown_rule(), zero_output_rule()],
            iterations=3,
        )

        overhead = (
            (
                (with_multiple.avg.duration_ms - no_guardrails.avg.duration_ms)
                / no_guardrails.avg.duration_ms
            )
            * 100
            if no_guardrails.avg.duration_ms > 0
            else 0
        )

        print(f"\nMultiple Guardrails overhead: {overhead:.1f}%")
        print(f"  Without: {no_guardrails.avg.tokens_per_second:.0f} tokens/s")
        print(f"  With 3 rules: {with_multiple.avg.tokens_per_second:.0f} tokens/s")

        assert with_multiple.avg.token_count == CONFIGS["standard"].token_count

    @pytest.mark.asyncio
    async def test_measure_pattern_guardrail_overhead(self):
        """Measure pattern guardrail overhead."""
        no_guardrails = await run_benchmark_suite(
            "No Guardrails",
            CONFIGS["standard"],
            guardrails=[],
            iterations=3,
        )

        with_patterns = await run_benchmark_suite(
            "Pattern Guardrail",
            CONFIGS["standard"],
            guardrails=[pattern_rule()],
            iterations=3,
        )

        overhead = (
            (
                (with_patterns.avg.duration_ms - no_guardrails.avg.duration_ms)
                / no_guardrails.avg.duration_ms
            )
            * 100
            if no_guardrails.avg.duration_ms > 0
            else 0
        )

        print(f"\nPattern Guardrail overhead: {overhead:.1f}%")
        print(f"  Without: {no_guardrails.avg.tokens_per_second:.0f} tokens/s")
        print(f"  With patterns: {with_patterns.avg.tokens_per_second:.0f} tokens/s")

        assert with_patterns.avg.token_count == CONFIGS["standard"].token_count


class TestDriftDetectionPerformance:
    """Drift detection performance impact benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_drift_detection_overhead(self):
        """Measure drift detection overhead."""
        no_drift = await run_benchmark_suite(
            "No Drift Detection",
            CONFIGS["high_throughput"],
            guardrails=[],
            detect_drift=False,
            iterations=3,
        )

        with_drift = await run_benchmark_suite(
            "With Drift Detection",
            CONFIGS["high_throughput"],
            guardrails=[],
            detect_drift=True,
            iterations=3,
        )

        overhead = (
            (
                (with_drift.avg.duration_ms - no_drift.avg.duration_ms)
                / no_drift.avg.duration_ms
            )
            * 100
            if no_drift.avg.duration_ms > 0
            else 0
        )

        print(f"\nDrift Detection overhead: {overhead:.1f}%")
        print(f"  Without: {no_drift.avg.tokens_per_second:.0f} tokens/s")
        print(f"  With drift: {with_drift.avg.tokens_per_second:.0f} tokens/s")

        assert with_drift.avg.token_count == CONFIGS["high_throughput"].token_count


class TestCheckIntervalImpact:
    """Check interval performance impact benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_impact_of_guardrail_check_interval(self):
        """Measure impact of guardrail check interval."""
        frequent_checks = await run_benchmark_suite(
            "Check every 1 token",
            CONFIGS["standard"],
            guardrails=[json_rule()],
            check_intervals=CheckIntervals(guardrails=1),
            iterations=3,
        )

        normal_checks = await run_benchmark_suite(
            "Check every 5 tokens",
            CONFIGS["standard"],
            guardrails=[json_rule()],
            check_intervals=CheckIntervals(guardrails=5),
            iterations=3,
        )

        infrequent_checks = await run_benchmark_suite(
            "Check every 20 tokens",
            CONFIGS["standard"],
            guardrails=[json_rule()],
            check_intervals=CheckIntervals(guardrails=20),
            iterations=3,
        )

        print("\nCheck Interval Impact:")
        print(f"  Every 1 token: {frequent_checks.avg.tokens_per_second:.0f} tokens/s")
        print(f"  Every 5 tokens: {normal_checks.avg.tokens_per_second:.0f} tokens/s")
        print(
            f"  Every 20 tokens: {infrequent_checks.avg.tokens_per_second:.0f} tokens/s"
        )

        # Less frequent checks should be faster
        assert infrequent_checks.avg.tokens_per_second >= (
            frequent_checks.avg.tokens_per_second * 0.9
        )


class TestFullStackPerformance:
    """Full stack performance benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_full_l0_stack_overhead(self):
        """Measure full L0 stack overhead."""
        baseline = await run_benchmark_suite(
            "Baseline",
            CONFIGS["high_throughput"],
            iterations=3,
            is_baseline=True,
        )

        full_stack = await run_benchmark_suite(
            "Full L0 Stack",
            CONFIGS["high_throughput"],
            guardrails=[json_rule(), markdown_rule(), zero_output_rule()],
            detect_drift=True,
            check_intervals=CheckIntervals(guardrails=5, drift=10, checkpoint=10),
            iterations=3,
        )

        overhead = (
            (
                (full_stack.avg.duration_ms - baseline.avg.duration_ms)
                / baseline.avg.duration_ms
            )
            * 100
            if baseline.avg.duration_ms > 0
            else 0
        )

        print(f"\n{'=' * 60}")
        print("FULL L0 STACK BENCHMARK")
        print("=" * 60)
        print(f"Tokens processed: {CONFIGS['high_throughput'].token_count}")
        print(
            f"Baseline: {baseline.avg.tokens_per_second:.0f} tokens/s "
            f"({baseline.avg.duration_ms:.2f} ms)"
        )
        print(
            f"Full Stack: {full_stack.avg.tokens_per_second:.0f} tokens/s "
            f"({full_stack.avg.duration_ms:.2f} ms)"
        )
        print(f"Overhead: {overhead:.1f}%")
        print(f"Time to First Token: {full_stack.avg.time_to_first_token_ms:.2f} ms")
        print("=" * 60)

        # Full stack should still achieve reasonable throughput
        assert full_stack.avg.tokens_per_second > 1000


class TestStressTests:
    """Stress test benchmarks."""

    @pytest.mark.asyncio
    async def test_handle_10000_tokens_efficiently(self):
        """Handle 10,000 tokens efficiently."""
        result = await run_benchmark_suite(
            "Stress Test (10k tokens)",
            CONFIGS["stress"],
            guardrails=[json_rule()],
            iterations=3,
        )

        print(f"\nStress Test (10,000 tokens):")
        print(f"  Throughput: {result.avg.tokens_per_second:.0f} tokens/s")
        print(f"  Duration: {result.avg.duration_ms:.2f} ms")
        print(f"  Memory delta: {result.avg.memory_delta_bytes / 1024 / 1024:.2f} MB")

        assert result.avg.token_count == 10000
        # Should process at least 5000 tokens/s even under stress
        assert result.avg.tokens_per_second > 5000

    @pytest.mark.asyncio
    async def test_maintain_linear_scaling(self):
        """Maintain linear scaling with token count."""
        small_config = MockStreamConfig(
            token_count=500, content_type="text", realistic=True
        )
        medium_config = MockStreamConfig(
            token_count=2000, content_type="text", realistic=True
        )
        large_config = MockStreamConfig(
            token_count=5000, content_type="text", realistic=True
        )

        small = await run_benchmark_suite(
            "500 tokens", small_config, guardrails=[], iterations=3
        )
        medium = await run_benchmark_suite(
            "2000 tokens", medium_config, guardrails=[], iterations=3
        )
        large = await run_benchmark_suite(
            "5000 tokens", large_config, guardrails=[], iterations=3
        )

        print("\nScaling Test:")
        print(
            f"  500 tokens: {small.avg.duration_ms:.2f} ms "
            f"({small.avg.tokens_per_second:.0f} t/s)"
        )
        print(
            f"  2000 tokens: {medium.avg.duration_ms:.2f} ms "
            f"({medium.avg.tokens_per_second:.0f} t/s)"
        )
        print(
            f"  5000 tokens: {large.avg.duration_ms:.2f} ms "
            f"({large.avg.tokens_per_second:.0f} t/s)"
        )

        # Throughput should remain relatively stable (within 50% variance)
        avg_throughput = (
            small.avg.tokens_per_second
            + medium.avg.tokens_per_second
            + large.avg.tokens_per_second
        ) / 3

        assert small.avg.tokens_per_second > avg_throughput * 0.5
        assert large.avg.tokens_per_second > avg_throughput * 0.5


class TestComprehensiveReport:
    """Comprehensive benchmark report."""

    @pytest.mark.asyncio
    async def test_generate_full_benchmark_report(self):
        """Generate full benchmark report."""
        scenarios: dict[str, BenchmarkSuiteResult] = {}

        # Run all scenarios
        scenario_configs: list[tuple[str, dict[str, Any]]] = [
            ("Baseline", {"is_baseline": True}),
            (
                "L0 Core (no features)",
                {"guardrails": [], "detect_drift": False},
            ),
            ("L0 + JSON Guardrail", {"guardrails": [json_rule()]}),
            (
                "L0 + All Guardrails",
                {"guardrails": [json_rule(), markdown_rule(), zero_output_rule()]},
            ),
            ("L0 + Drift Detection", {"guardrails": [], "detect_drift": True}),
            (
                "L0 Full Stack",
                {
                    "guardrails": [json_rule(), markdown_rule(), zero_output_rule()],
                    "detect_drift": True,
                },
            ),
        ]

        baseline_result: BenchmarkMetrics | None = None

        for name, options in scenario_configs:
            is_baseline = options.pop("is_baseline", False)
            result = await run_benchmark_suite(
                name,
                CONFIGS["high_throughput"],
                iterations=3,
                is_baseline=is_baseline,
                **options,
            )

            if is_baseline:
                baseline_result = result.avg

            scenarios[name] = result

        # Print report
        assert baseline_result is not None
        print("\n" + format_report(scenarios, baseline_result))

        # Validate all scenarios completed
        assert len(scenarios) == len(scenario_configs)


class TestLatencyDistribution:
    """Latency distribution benchmarks."""

    @pytest.mark.asyncio
    async def test_measure_token_latency_percentiles(self):
        """Measure token latency percentiles."""
        token_latencies: list[float] = []
        last_token_time = time.perf_counter()

        config = MockStreamConfig(token_count=1000, content_type="text", realistic=True)

        result = await _internal_run(
            stream=create_mock_stream_factory(config),
            guardrails=[json_rule()],
        )

        async for event in result:
            if event.is_token:
                now = time.perf_counter()
                token_latencies.append((now - last_token_time) * 1000)  # ms
                last_token_time = now

        # Calculate percentiles
        sorted_latencies = sorted(token_latencies)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.5)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]
        max_latency = sorted_latencies[-1]

        print("\nToken Latency Distribution:")
        print(f"  p50: {p50 * 1000:.0f} us")
        print(f"  p95: {p95 * 1000:.0f} us")
        print(f"  p99: {p99 * 1000:.0f} us")
        print(f"  max: {max_latency * 1000:.0f} us")

        # p50 should be very low (< 1ms per token processing)
        assert p50 < 1
