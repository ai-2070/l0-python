"""Telemetry exporters for various formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .telemetry import Telemetry


class TelemetryExporter:
    """Export telemetry data to various formats.

    Usage:
        ```python
        from l0.monitoring import TelemetryExporter, Telemetry

        telemetry = monitoring.get_telemetry()

        # Export to JSON
        json_str = TelemetryExporter.to_json(telemetry)

        # Export multiple to CSV
        csv_str = TelemetryExporter.to_csv([telemetry1, telemetry2])

        # Export to log format
        log_str = TelemetryExporter.to_log_format(telemetry)

        # Export to metrics format (Prometheus-compatible)
        metrics = TelemetryExporter.to_metrics(telemetry, prefix="l0")
        ```
    """

    @staticmethod
    def to_json(telemetry: Telemetry, *, indent: int | None = 2) -> str:
        """Export telemetry to JSON string.

        Args:
            telemetry: Telemetry data to export
            indent: JSON indentation (None for compact)

        Returns:
            JSON string representation
        """
        return telemetry.model_dump_json(indent=indent)

    @staticmethod
    def to_dict(telemetry: Telemetry) -> dict[str, Any]:
        """Export telemetry to dictionary.

        Args:
            telemetry: Telemetry data to export

        Returns:
            Dictionary representation
        """
        return telemetry.model_dump(mode="json")

    @staticmethod
    def to_csv(telemetry_list: list[Telemetry]) -> str:
        """Export multiple telemetry records to CSV.

        Args:
            telemetry_list: List of telemetry records

        Returns:
            CSV string with headers
        """
        if not telemetry_list:
            return ""

        output = io.StringIO()
        fieldnames = [
            "stream_id",
            "session_id",
            "model",
            "started_at",
            "completed_at",
            "duration",
            "time_to_first_token",
            "token_count",
            "tokens_per_second",
            "avg_inter_token_latency",
            "model_retries",
            "network_retries",
            "total_retries",
            "guardrail_violations",
            "error_occurred",
            "error_message",
            "error_category",
            "completed",
            "aborted",
            "content_length",
        ]

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for t in telemetry_list:
            writer.writerow(
                {
                    "stream_id": t.stream_id,
                    "session_id": t.session_id,
                    "model": t.model,
                    "started_at": t.timing.started_at.isoformat()
                    if t.timing.started_at
                    else None,
                    "completed_at": t.timing.completed_at.isoformat()
                    if t.timing.completed_at
                    else None,
                    "duration": t.timing.duration,
                    "time_to_first_token": t.metrics.time_to_first_token,
                    "token_count": t.metrics.token_count,
                    "tokens_per_second": t.metrics.tokens_per_second,
                    "avg_inter_token_latency": t.metrics.avg_inter_token_latency,
                    "model_retries": t.retries.model_retries,
                    "network_retries": t.retries.network_retries,
                    "total_retries": t.retries.total_retries,
                    "guardrail_violations": len(t.guardrails.violations),
                    "error_occurred": t.error.occurred,
                    "error_message": t.error.message,
                    "error_category": t.error.category.value
                    if t.error.category
                    else None,
                    "completed": t.completed,
                    "aborted": t.aborted,
                    "content_length": t.content_length,
                }
            )

        return output.getvalue()

    @staticmethod
    def to_log_format(telemetry: Telemetry) -> str:
        """Export telemetry to structured log format.

        Args:
            telemetry: Telemetry data to export

        Returns:
            Log-friendly string representation
        """
        parts = [
            f"stream_id={telemetry.stream_id}",
        ]

        if telemetry.model:
            parts.append(f"model={telemetry.model}")

        if telemetry.timing.duration is not None:
            parts.append(f"duration={telemetry.timing.duration:.3f}s")

        if telemetry.metrics.time_to_first_token is not None:
            parts.append(f"ttft={telemetry.metrics.time_to_first_token:.3f}s")

        parts.append(f"tokens={telemetry.metrics.token_count}")

        if telemetry.metrics.tokens_per_second is not None:
            parts.append(f"tokens_per_sec={telemetry.metrics.tokens_per_second:.1f}")

        if telemetry.retries.total_retries > 0:
            parts.append(f"retries={telemetry.retries.total_retries}")

        if telemetry.guardrails.violations:
            parts.append(f"violations={len(telemetry.guardrails.violations)}")

        if telemetry.error.occurred:
            parts.append(f"error={telemetry.error.message}")

        parts.append(f"completed={telemetry.completed}")

        if telemetry.aborted:
            parts.append("aborted=true")

        return " ".join(parts)

    @staticmethod
    def to_metrics(
        telemetry: Telemetry,
        *,
        prefix: str = "l0",
        labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Export telemetry to Prometheus-compatible metrics format.

        Args:
            telemetry: Telemetry data to export
            prefix: Metric name prefix
            labels: Additional labels to attach

        Returns:
            Dictionary of metric names to values with labels
        """
        base_labels = {
            "stream_id": telemetry.stream_id,
        }
        if telemetry.model:
            base_labels["model"] = telemetry.model
        if telemetry.session_id:
            base_labels["session_id"] = telemetry.session_id
        if labels:
            base_labels.update(labels)

        metrics: dict[str, Any] = {}

        # Token metrics
        metrics[f"{prefix}_tokens_total"] = {
            "value": telemetry.metrics.token_count,
            "type": "counter",
            "labels": base_labels,
        }

        if telemetry.metrics.tokens_per_second is not None:
            metrics[f"{prefix}_tokens_per_second"] = {
                "value": telemetry.metrics.tokens_per_second,
                "type": "gauge",
                "labels": base_labels,
            }

        # Timing metrics
        if telemetry.timing.duration is not None:
            metrics[f"{prefix}_duration_seconds"] = {
                "value": telemetry.timing.duration,
                "type": "gauge",
                "labels": base_labels,
            }

        if telemetry.metrics.time_to_first_token is not None:
            metrics[f"{prefix}_ttft_seconds"] = {
                "value": telemetry.metrics.time_to_first_token,
                "type": "gauge",
                "labels": base_labels,
            }

        if telemetry.metrics.avg_inter_token_latency is not None:
            metrics[f"{prefix}_inter_token_latency_seconds"] = {
                "value": telemetry.metrics.avg_inter_token_latency,
                "type": "gauge",
                "labels": base_labels,
            }

        # Retry metrics
        metrics[f"{prefix}_retries_total"] = {
            "value": telemetry.retries.total_retries,
            "type": "counter",
            "labels": base_labels,
        }

        metrics[f"{prefix}_model_retries_total"] = {
            "value": telemetry.retries.model_retries,
            "type": "counter",
            "labels": base_labels,
        }

        metrics[f"{prefix}_network_retries_total"] = {
            "value": telemetry.retries.network_retries,
            "type": "counter",
            "labels": base_labels,
        }

        # Guardrail metrics
        metrics[f"{prefix}_guardrail_violations_total"] = {
            "value": len(telemetry.guardrails.violations),
            "type": "counter",
            "labels": base_labels,
        }

        # Error metrics
        error_labels = {**base_labels}
        if telemetry.error.category:
            error_labels["category"] = telemetry.error.category.value
        metrics[f"{prefix}_errors_total"] = {
            "value": 1 if telemetry.error.occurred else 0,
            "type": "counter",
            "labels": error_labels,
        }

        # Completion metrics
        metrics[f"{prefix}_completed_total"] = {
            "value": 1 if telemetry.completed else 0,
            "type": "counter",
            "labels": base_labels,
        }

        metrics[f"{prefix}_aborted_total"] = {
            "value": 1 if telemetry.aborted else 0,
            "type": "counter",
            "labels": base_labels,
        }

        return metrics

    @staticmethod
    def to_jsonl(telemetry_list: list[Telemetry]) -> str:
        """Export multiple telemetry records to JSON Lines format.

        Args:
            telemetry_list: List of telemetry records

        Returns:
            JSONL string (one JSON object per line)
        """
        lines = [t.model_dump_json() for t in telemetry_list]
        return "\n".join(lines)
