"""Tests for L0 Pydantic models."""

import pytest
from pydantic import ValidationError

from l0.pydantic import (
    AggregatedTelemetryModel,
    AgreementModel,
    BackoffStrategyModel,
    ChunkResultModel,
    ConsensusAnalysisModel,
    # Consensus types
    ConsensusOutputModel,
    ConsensusResultModel,
    DisagreementModel,
    # Window types
    DocumentChunkModel,
    DriftConfigModel,
    # Drift types
    DriftResultModel,
    # Error types
    ErrorCodeModel,
    ErrorContextModel,
    EventEnvelopeModel,
    EventModel,
    EventTypeModel,
    GuardrailResultModel,
    GuardrailResultSummaryModel,
    # Guardrail types
    GuardrailViolationModel,
    # Metrics types
    MetricsSnapshotModel,
    # Parallel types
    ParallelResultModel,
    PipelineResultModel,
    # Pool types
    PoolStatsModel,
    RaceResultModel,
    # Event sourcing types
    RecordedEventModel,
    RecordedEventTypeModel,
    RecoveryPolicyModel,
    RetryModel,
    # State machine types
    RuntimeStateModel,
    # Core types
    StateModel,
    StateTransitionModel,
    # Pipeline types
    StepResultModel,
    TimeoutModel,
    WindowStatsModel,
)


class TestCoreTypes:
    """Tests for core Pydantic models."""

    def test_state_model_defaults(self):
        """Test StateModel with default values."""
        state = StateModel()
        assert state.content == ""
        assert state.token_count == 0
        assert state.completed is False
        assert state.violations == []

    def test_state_model_with_values(self):
        """Test StateModel with explicit values."""
        state = StateModel(
            content="Hello world",
            token_count=3,
            completed=True,
        )
        assert state.content == "Hello world"
        assert state.token_count == 3
        assert state.completed is True

    def test_state_model_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        state = StateModel(content="test", token_count=5)
        json_str = state.model_dump_json()
        restored = StateModel.model_validate_json(json_str)
        assert restored.content == state.content
        assert restored.token_count == state.token_count

    def test_retry_model_defaults(self):
        """Test RetryModel with defaults."""
        retry = RetryModel()
        assert retry.attempts == 3
        assert retry.base_delay == 1.0
        assert retry.strategy == BackoffStrategyModel.FIXED_JITTER

    def test_retry_model_custom(self):
        """Test RetryModel with custom values."""
        retry = RetryModel(
            attempts=5,
            base_delay=2.0,
            strategy=BackoffStrategyModel.EXPONENTIAL,
        )
        assert retry.attempts == 5
        assert retry.base_delay == 2.0
        assert retry.strategy == BackoffStrategyModel.EXPONENTIAL

    def test_timeout_model(self):
        """Test TimeoutModel."""
        timeout = TimeoutModel(initial_token=10000, inter_token=5000)
        assert timeout.initial_token == 10000
        assert timeout.inter_token == 5000

    def test_event_model(self):
        """Test EventModel."""
        event = EventModel(type=EventTypeModel.TOKEN, text="hello")
        assert event.type == EventTypeModel.TOKEN
        assert event.text == "hello"


class TestErrorTypes:
    """Tests for error Pydantic models."""

    def test_error_code_enum(self):
        """Test ErrorCodeModel enum values."""
        assert ErrorCodeModel.STREAM_ABORTED == "STREAM_ABORTED"
        assert ErrorCodeModel.NETWORK_ERROR == "NETWORK_ERROR"

    def test_recovery_policy_model(self):
        """Test RecoveryPolicyModel."""
        policy = RecoveryPolicyModel(
            retry_enabled=True,
            fallback_enabled=True,
            max_retries=5,
        )
        assert policy.retry_enabled is True
        assert policy.max_retries == 5

    def test_error_context_model(self):
        """Test ErrorContextModel."""
        context = ErrorContextModel(
            code=ErrorCodeModel.NETWORK_ERROR,
            token_count=100,
        )
        assert context.code == ErrorCodeModel.NETWORK_ERROR
        assert context.token_count == 100


class TestGuardrailTypes:
    """Tests for guardrail Pydantic models."""

    def test_guardrail_violation_model(self):
        """Test GuardrailViolationModel."""
        violation = GuardrailViolationModel(
            rule="max_length",
            message="Content too long",
            severity="error",
            recoverable=True,
        )
        assert violation.rule == "max_length"
        assert violation.severity == "error"
        assert violation.recoverable is True

    def test_guardrail_result_model(self):
        """Test GuardrailResultModel."""
        result = GuardrailResultModel(
            passed=False,
            violations=[
                GuardrailViolationModel(
                    rule="test",
                    message="test violation",
                    severity="warning",
                )
            ],
            should_retry=True,
            should_halt=False,
            summary=GuardrailResultSummaryModel(
                total=1,
                fatal=0,
                errors=0,
                warnings=1,
            ),
        )
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.should_retry is True


class TestConsensusTypes:
    """Tests for consensus Pydantic models."""

    def test_consensus_output_model(self):
        """Test ConsensusOutputModel."""
        output = ConsensusOutputModel(
            index=0,
            text="Hello",
            value="Hello",
            success=True,
            duration_ms=100.0,
        )
        assert output.index == 0
        assert output.success is True

    def test_agreement_model(self):
        """Test AgreementModel."""
        agreement = AgreementModel(
            content="agreed value",
            count=3,
            ratio=0.75,
            indices=[0, 1, 2],
        )
        assert agreement.count == 3
        assert agreement.ratio == 0.75

    def test_consensus_result_model(self):
        """Test ConsensusResultModel."""
        result = ConsensusResultModel(
            consensus="final answer",
            confidence=0.9,
            outputs=[
                ConsensusOutputModel(
                    index=0,
                    text="answer",
                    value="answer",
                    success=True,
                )
            ],
            agreements=[],
            disagreements=[],
            analysis=ConsensusAnalysisModel(
                total_outputs=1,
                successful_outputs=1,
            ),
            status="success",
        )
        assert result.confidence == 0.9
        assert result.status == "success"


class TestParallelTypes:
    """Tests for parallel Pydantic models."""

    def test_parallel_result_model(self):
        """Test ParallelResultModel."""
        result = ParallelResultModel(
            results=["a", "b", None],
            errors=[None, None, "failed"],
            success_count=2,
            failure_count=1,
            duration=1.5,
        )
        assert result.success_count == 2
        assert result.failure_count == 1

    def test_race_result_model(self):
        """Test RaceResultModel."""
        result = RaceResultModel(value="winner", winner_index=0)
        assert result.value == "winner"
        assert result.winner_index == 0

    def test_aggregated_telemetry_model(self):
        """Test AggregatedTelemetryModel."""
        telemetry = AggregatedTelemetryModel(
            total_tokens=1000,
            total_duration=5.0,
            avg_tokens_per_second=200.0,
        )
        assert telemetry.total_tokens == 1000


class TestPipelineTypes:
    """Tests for pipeline Pydantic models."""

    def test_step_result_model(self):
        """Test StepResultModel."""
        result = StepResultModel(
            step_name="summarize",
            step_index=0,
            input="long text",
            output="summary",
            raw_content="summary",
            status="success",
            duration=500,
        )
        assert result.step_name == "summarize"
        assert result.status == "success"

    def test_pipeline_result_model(self):
        """Test PipelineResultModel."""
        result = PipelineResultModel(
            name="my-pipeline",
            output="final output",
            steps=[],
            status="success",
            duration=1000,
        )
        assert result.name == "my-pipeline"
        assert result.status == "success"


class TestPoolTypes:
    """Tests for pool Pydantic models."""

    def test_pool_stats_model(self):
        """Test PoolStatsModel."""
        stats = PoolStatsModel(
            total_executed=10,
            total_succeeded=8,
            total_failed=2,
            total_duration=5.0,
        )
        assert stats.total_executed == 10
        assert stats.total_succeeded == 8


class TestWindowTypes:
    """Tests for window Pydantic models."""

    def test_document_chunk_model(self):
        """Test DocumentChunkModel."""
        chunk = DocumentChunkModel(
            index=0,
            content="First chunk",
            start_pos=0,
            end_pos=11,
            token_count=3,
            char_count=11,
            is_first=True,
            is_last=False,
            total_chunks=5,
        )
        assert chunk.index == 0
        assert chunk.is_first is True

    def test_window_stats_model(self):
        """Test WindowStatsModel."""
        stats = WindowStatsModel(
            total_chunks=5,
            total_chars=5000,
            total_tokens=1250,
            avg_chunk_size=1000,
            avg_chunk_tokens=250,
            overlap_size=100,
            strategy="token",
        )
        assert stats.total_chunks == 5
        assert stats.strategy == "token"


class TestDriftTypes:
    """Tests for drift Pydantic models."""

    def test_drift_result_model_no_drift(self):
        """Test DriftResultModel with no drift."""
        result = DriftResultModel(
            detected=False,
            confidence=0.0,
            types=[],
        )
        assert result.detected is False
        assert result.types == []

    def test_drift_result_model_with_drift(self):
        """Test DriftResultModel with drift detected."""
        result = DriftResultModel(
            detected=True,
            confidence=0.85,
            types=["meta_commentary", "tone_shift"],
            details="Meta commentary and tone shift detected",
        )
        assert result.detected is True
        assert "meta_commentary" in result.types

    def test_drift_config_model(self):
        """Test DriftConfigModel."""
        config = DriftConfigModel(
            detect_tone_shift=True,
            detect_meta_commentary=True,
            repetition_threshold=5,
        )
        assert config.detect_tone_shift is True
        assert config.repetition_threshold == 5


class TestStateMachineTypes:
    """Tests for state machine Pydantic models."""

    def test_runtime_state_enum(self):
        """Test RuntimeStateModel enum values."""
        assert RuntimeStateModel.INIT == "init"
        assert RuntimeStateModel.STREAMING == "streaming"
        assert RuntimeStateModel.COMPLETE == "complete"

    def test_state_transition_model(self):
        """Test StateTransitionModel."""
        transition = StateTransitionModel(
            from_state=RuntimeStateModel.INIT,
            to_state=RuntimeStateModel.STREAMING,
            timestamp=1234567890.0,
        )
        assert transition.from_state == RuntimeStateModel.INIT
        assert transition.to_state == RuntimeStateModel.STREAMING


class TestEventSourcingTypes:
    """Tests for event sourcing Pydantic models."""

    def test_recorded_event_model(self):
        """Test RecordedEventModel."""
        event = RecordedEventModel(
            type=RecordedEventTypeModel.TOKEN,
            ts=1234567890.0,
            data={"value": "hello", "index": 0},
        )
        assert event.type == RecordedEventTypeModel.TOKEN
        assert event.data["value"] == "hello"

    def test_event_envelope_model(self):
        """Test EventEnvelopeModel."""
        envelope = EventEnvelopeModel(
            stream_id="test-stream",
            seq=0,
            event=RecordedEventModel(
                type=RecordedEventTypeModel.START,
                ts=1234567890.0,
            ),
        )
        assert envelope.stream_id == "test-stream"
        assert envelope.seq == 0


class TestMetricsTypes:
    """Tests for metrics Pydantic models."""

    def test_metrics_snapshot_model(self):
        """Test MetricsSnapshotModel."""
        snapshot = MetricsSnapshotModel(
            requests=100,
            tokens=5000,
            retries=10,
            network_retry_count=5,
            errors=2,
            violations=3,
            drift_detections=1,
            fallbacks=0,
            completions=95,
            timeouts=3,
        )
        assert snapshot.requests == 100
        assert snapshot.tokens == 5000
        assert snapshot.completions == 95


class TestValidation:
    """Tests for Pydantic validation behavior."""

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected for strict models."""
        with pytest.raises(ValidationError):
            StateModel(content="test", unknown_field="should fail")

    def test_type_coercion(self):
        """Test that types are properly coerced."""
        # String to int coercion
        state = StateModel(token_count="5")  # type: ignore
        assert state.token_count == 5

    def test_json_schema_generation(self):
        """Test JSON schema generation."""
        schema = StateModel.model_json_schema()
        assert "properties" in schema
        assert "content" in schema["properties"]
        assert "token_count" in schema["properties"]
