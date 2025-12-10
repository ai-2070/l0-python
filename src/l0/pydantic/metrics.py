"""Pydantic models for L0 metrics types.

These models mirror the dataclasses in l0.metrics for runtime validation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MetricsSnapshotModel(BaseModel):
    """Pydantic model for a snapshot of all metrics."""

    model_config = ConfigDict(extra="forbid")

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
