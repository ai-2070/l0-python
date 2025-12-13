# Benchmarks

Performance benchmarks measuring L0 overhead on high-throughput streaming.

## Test Environment

- **CPU**: Apple M1 Pro
- **Runtime**: Python 3.13 with pytest-asyncio
- **Methodology**: Mock token streams with zero inter-token delay to measure pure L0 overhead

## Results

| Scenario | Tokens/s | Avg Duration | TTFT | Overhead |
|----------|----------|--------------|------|----------|
| Baseline (raw streaming) | 1,486,663 | 1.35 ms | 0.01 ms | - |
| L0 Core (no features) | 643,240 | 3.11 ms | 0.06 ms | 131% |
| L0 + JSON Guardrail | 525,381 | 3.81 ms | 0.07 ms | 183% |
| L0 + All Guardrails | 400,643 | 4.99 ms | 0.07 ms | 271% |
| L0 + Drift Detection | 125,256 | 15.97 ms | 0.08 ms | 1086% |
| L0 Full Stack | 112,414 | 17.79 ms | 0.06 ms | 1221% |

**Legend:**
- **Tokens/s** = Throughput (higher is better)
- **Avg Duration** = Average total duration for 2000 tokens
- **TTFT** = Time to first token (lower is better)
- **Overhead** = % slower than baseline

## Key Optimizations

L0 includes several optimizations for high-throughput streaming:

### 1. Incremental JSON State Tracking
Instead of re-scanning the entire content on each guardrail check, L0 tracks JSON structure incrementally:
- **O(delta)** per token instead of **O(content)**
- Only performs full content scan at stream completion

### 2. Sliding Window Drift Detection
Drift detection uses a sliding window (default 500 characters) instead of scanning full content:
- Meta commentary, tone shift, repetition checks operate on window only
- Configurable via `DriftConfig.sliding_window_size`

### 3. Tunable Check Intervals
Default intervals optimized for high throughput:
- **Guardrails**: Every 15 tokens (was 5)
- **Drift**: Every 25 tokens (was 10)
- **Checkpoint**: Every 20 tokens (was 10)

Configure via `check_intervals`:
```python
from l0.guardrails import json_rule
from l0.types import CheckIntervals
import l0

result = await l0.run(
    stream=my_stream,
    guardrails=[json_rule()],
    check_intervals=CheckIntervals(guardrails=15, drift=25, checkpoint=20),
)
```

## Blackwell Ready

Even with full guardrails, drift detection, and checkpointing enabled, L0 sustains **112K+ tokens/s** - well above current LLM inference speeds and ready for Nvidia Blackwell's 1000+ tokens/s streaming.

| GPU Generation | Expected Tokens/s | L0 Headroom |
|----------------|-------------------|-------------|
| Current (H100) | ~100-200 | 560-1120x |
| Blackwell (B200) | ~1000+ | 112x |

## Running Benchmarks

```bash
uv run pytest tests/test_benchmark.py::TestComprehensiveReport -v -s
```

To run all benchmark tests:
```bash
uv run pytest tests/test_benchmark.py -v
```

## Benchmark Scenarios

### Baseline
Raw async iteration without L0 - measures the cost of the mock stream itself.

### L0 Core
Minimal L0 wrapper with no guardrails or drift detection. Measures the base cost of the L0 runtime.

### L0 + JSON Guardrail
L0 with `json_rule()` enabled. Tests incremental JSON structure validation.

### L0 + All Guardrails
L0 with `json_rule()`, `markdown_rule()`, and `zero_output_rule()`. Tests multiple guardrail overhead.

### L0 + Drift Detection
L0 with drift detection enabled. Tests sliding window analysis overhead.

### L0 Full Stack
L0 with all features: JSON, Markdown, zero-output guardrails, drift detection, and checkpointing. Represents real-world production usage.
