"""Pydantic models for L0 consensus types.

These models mirror the dataclasses in l0.consensus for runtime validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrategyModel(str, Enum):
    """Consensus strategy."""

    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    BEST = "best"


class ConflictResolutionModel(str, Enum):
    """Conflict resolution strategy."""

    VOTE = "vote"
    MERGE = "merge"
    BEST = "best"
    FAIL = "fail"


class AgreementTypeModel(str, Enum):
    """Type of agreement."""

    EXACT = "exact"
    SIMILAR = "similar"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


class DisagreementSeverityModel(str, Enum):
    """Severity of disagreement."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class AgreementModel(BaseModel):
    """Pydantic model for what outputs agreed on."""

    model_config = ConfigDict(extra="forbid")

    content: Any
    path: str | None = None
    count: int = 0
    ratio: float = 0.0
    indices: list[int] = Field(default_factory=list)
    type: AgreementTypeModel = AgreementTypeModel.EXACT


class DisagreementValueModel(BaseModel):
    """Pydantic model for a single value in a disagreement."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    count: int
    indices: list[int]


class DisagreementModel(BaseModel):
    """Pydantic model for where outputs differed."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    values: list[DisagreementValueModel] = Field(default_factory=list)
    severity: DisagreementSeverityModel = DisagreementSeverityModel.MINOR
    resolution: str | None = None
    resolution_confidence: float | None = None


class ConsensusAnalysisModel(BaseModel):
    """Pydantic model for detailed consensus statistics."""

    model_config = ConfigDict(extra="forbid")

    total_outputs: int = 0
    successful_outputs: int = 0
    failed_outputs: int = 0
    identical_outputs: int = 0
    similarity_matrix: list[list[float]] = Field(default_factory=list)
    average_similarity: float = 0.0
    min_similarity: float = 0.0
    max_similarity: float = 0.0
    total_agreements: int = 0
    total_disagreements: int = 0
    strategy: str = ""
    conflict_resolution: str = ""
    duration_ms: float = 0.0


class FieldAgreementModel(BaseModel):
    """Pydantic model for per-field consensus information."""

    model_config = ConfigDict(extra="forbid")

    path: str
    value: Any
    agreement: float
    votes: dict[str, int]
    values: list[Any]
    unanimous: bool
    confidence: float


class FieldConsensusModel(BaseModel):
    """Pydantic model for field-by-field consensus."""

    model_config = ConfigDict(extra="forbid")

    fields: dict[str, FieldAgreementModel] = Field(default_factory=dict)
    overall_agreement: float = 0.0
    agreed_fields: list[str] = Field(default_factory=list)
    disagreed_fields: list[str] = Field(default_factory=list)


class ConsensusOutputModel(BaseModel):
    """Pydantic model for individual output from a stream."""

    model_config = ConfigDict(extra="forbid")

    index: int
    text: str
    value: Any
    success: bool
    data: Any = None
    l0_result: Any = None
    structured_result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    weight: float = 1.0
    similarities: list[float] | None = None


class ConsensusResultModel(BaseModel):
    """Pydantic model for result of consensus operation."""

    model_config = ConfigDict(extra="forbid")

    consensus: Any
    confidence: float
    outputs: list[ConsensusOutputModel]
    agreements: list[AgreementModel]
    disagreements: list[DisagreementModel]
    analysis: ConsensusAnalysisModel
    type: Literal["text", "structured"] = "text"
    field_consensus: FieldConsensusModel | None = None
    status: Literal["success", "partial", "failed"] = "success"


class ConsensusPresetModel(BaseModel):
    """Pydantic model for preset consensus configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: StrategyModel
    threshold: float
    resolve_conflicts: ConflictResolutionModel
    minimum_agreement: float
