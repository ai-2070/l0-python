"""Pydantic models for L0 types.

This module provides Pydantic BaseModel equivalents for all L0 dataclasses,
enabling runtime validation, JSON serialization, and schema generation.

Usage:
    from l0.pydantic import StateModel, RetryModel, TimeoutModel

    # Validate data
    state = StateModel(content="Hello", token_count=5)

    # Serialize to JSON
    json_data = state.model_dump_json()

    # Generate JSON schema
    schema = StateModel.model_json_schema()
"""

# Core types
# Consensus types
from .consensus import (
    AgreementModel,
    AgreementTypeModel,
    ConflictResolutionModel,
    ConsensusAnalysisModel,
    ConsensusOutputModel,
    ConsensusPresetModel,
    ConsensusResultModel,
    DisagreementModel,
    DisagreementSeverityModel,
    DisagreementValueModel,
    FieldAgreementModel,
    FieldConsensusModel,
    StrategyModel,
)

# Drift types
from .drift import (
    DriftConfigModel,
    DriftResultModel,
)

# Error types
from .errors import (
    ErrorCodeModel,
    ErrorContextModel,
    FailureTypeModel,
    NetworkErrorAnalysisModel,
    NetworkErrorTypeModel,
    RecoveryPolicyModel,
    RecoveryStrategyModel,
    SerializedErrorModel,
)

# Event sourcing types
from .event_sourcing import (
    EventEnvelopeModel,
    RecordedEventModel,
    RecordedEventTypeModel,
    ReplayComparisonModel,
    ReplayedStateModel,
    SnapshotModel,
    StreamMetadataModel,
)
from .event_sourcing import (
    SerializedErrorModel as EventSourcingSerializedErrorModel,
)

# Event/observability types
from .events import (
    CompleteEventModel,
    ErrorEventModel,
    ObservabilityEventModel,
    ObservabilityEventTypeModel,
    RetryAttemptEventModel,
    SessionEndEventModel,
    SessionStartEventModel,
)

# Guardrail types
from .guardrails import (
    GuardrailConfigModel,
    GuardrailContextModel,
    GuardrailResultModel,
    GuardrailResultSummaryModel,
    GuardrailStateModel,
    GuardrailViolationModel,
    JsonAnalysisModel,
    LatexAnalysisModel,
    MarkdownAnalysisModel,
)

# Metrics types
from .metrics import (
    MetricsSnapshotModel,
)

# Parallel types
from .parallel import (
    AggregatedTelemetryModel,
    ParallelOptionsModel,
    ParallelResultModel,
    RaceResultModel,
)

# Pipeline types
from .pipeline import (
    PipelineOptionsModel,
    PipelineResultModel,
    PipelineStepModel,
    StepContextModel,
    StepResultModel,
)

# Pool types
from .pool import (
    PoolOptionsModel,
    PoolStatsModel,
)

# State machine types
from .state_machine import (
    RuntimeStateModel,
    StateTransitionModel,
)
from .types import (
    BackoffStrategyModel,
    CheckIntervalsModel,
    ContentTypeModel,
    DataPayloadModel,
    ErrorCategoryModel,
    ErrorTypeDelaysModel,
    EventModel,
    EventTypeModel,
    ProgressModel,
    RetryableErrorTypeModel,
    RetryModel,
    StateModel,
    TelemetryContinuationModel,
    TelemetryDriftModel,
    TelemetryGuardrailsModel,
    TelemetryMetricsModel,
    TelemetryModel,
    TelemetryNetworkModel,
    TimeoutModel,
)

# Window types
from .window import (
    ChunkResultModel,
    ContextRestorationOptionsModel,
    DocumentChunkModel,
    ProcessingStatsModel,
    WindowConfigModel,
    WindowStatsModel,
)

__all__ = [
    # Core types
    "BackoffStrategyModel",
    "CheckIntervalsModel",
    "ContentTypeModel",
    "DataPayloadModel",
    "ErrorCategoryModel",
    "ErrorTypeDelaysModel",
    "EventModel",
    "EventTypeModel",
    "ProgressModel",
    "RetryableErrorTypeModel",
    "RetryModel",
    "StateModel",
    "TelemetryContinuationModel",
    "TelemetryDriftModel",
    "TelemetryGuardrailsModel",
    "TelemetryMetricsModel",
    "TelemetryModel",
    "TelemetryNetworkModel",
    "TimeoutModel",
    # Error types
    "ErrorCodeModel",
    "ErrorContextModel",
    "FailureTypeModel",
    "NetworkErrorAnalysisModel",
    "NetworkErrorTypeModel",
    "RecoveryPolicyModel",
    "RecoveryStrategyModel",
    "SerializedErrorModel",
    # Guardrail types
    "GuardrailConfigModel",
    "GuardrailContextModel",
    "GuardrailResultModel",
    "GuardrailResultSummaryModel",
    "GuardrailStateModel",
    "GuardrailViolationModel",
    "JsonAnalysisModel",
    "LatexAnalysisModel",
    "MarkdownAnalysisModel",
    # Event/observability types
    "CompleteEventModel",
    "ErrorEventModel",
    "ObservabilityEventModel",
    "ObservabilityEventTypeModel",
    "RetryAttemptEventModel",
    "SessionEndEventModel",
    "SessionStartEventModel",
    # Consensus types
    "AgreementModel",
    "AgreementTypeModel",
    "ConflictResolutionModel",
    "ConsensusAnalysisModel",
    "ConsensusOutputModel",
    "ConsensusPresetModel",
    "ConsensusResultModel",
    "DisagreementModel",
    "DisagreementSeverityModel",
    "DisagreementValueModel",
    "FieldAgreementModel",
    "FieldConsensusModel",
    "StrategyModel",
    # Parallel types
    "AggregatedTelemetryModel",
    "ParallelOptionsModel",
    "ParallelResultModel",
    "RaceResultModel",
    # Pipeline types
    "PipelineOptionsModel",
    "PipelineResultModel",
    "PipelineStepModel",
    "StepContextModel",
    "StepResultModel",
    # Pool types
    "PoolOptionsModel",
    "PoolStatsModel",
    # Window types
    "ChunkResultModel",
    "ContextRestorationOptionsModel",
    "DocumentChunkModel",
    "ProcessingStatsModel",
    "WindowConfigModel",
    "WindowStatsModel",
    # Drift types
    "DriftConfigModel",
    "DriftResultModel",
    # State machine types
    "RuntimeStateModel",
    "StateTransitionModel",
    # Event sourcing types
    "EventEnvelopeModel",
    "EventSourcingSerializedErrorModel",
    "RecordedEventModel",
    "RecordedEventTypeModel",
    "ReplayComparisonModel",
    "ReplayedStateModel",
    "SnapshotModel",
    "StreamMetadataModel",
    # Metrics types
    "MetricsSnapshotModel",
]
