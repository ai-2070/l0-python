"""Integration tests for consensus features with real API calls.

These tests make multiple API calls per test (2-3 concurrent calls).
Requires OPENAI_API_KEY to be set.

Run with: pytest tests/integration/test_consensus.py -v
"""

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

import l0
from l0 import Consensus
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Test Schemas
# ─────────────────────────────────────────────────────────────────────────────


class Person(BaseModel):
    """Simple person schema for structured consensus."""

    name: str
    age: int


class MathResult(BaseModel):
    """Math result schema."""

    answer: int
    explanation: str


# ─────────────────────────────────────────────────────────────────────────────
# Basic Consensus Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestBasicConsensus:
    """Basic consensus integration tests."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_unanimous_consensus_on_fact(self, client: "AsyncOpenAI") -> None:
        """Test unanimous consensus on a factual question."""

        async def ask_capital():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "What is the capital of France? Answer with just the city name.",
                    }
                ],
                max_tokens=10,
            )
            return (stream.choices[0].message.content or "").strip()

        result = await Consensus.run(
            [ask_capital, ask_capital, ask_capital],
            strategy="unanimous",
            threshold=0.8,
        )

        assert "Paris" in result.consensus
        assert result.confidence >= 0.8
        assert result.status == "success"
        assert len(result.outputs) == 3

    @pytest.mark.asyncio
    async def test_majority_consensus(self, client: "AsyncOpenAI") -> None:
        """Test majority consensus strategy."""

        async def simple_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Is 2+2 equal to 4? Answer yes or no only.",
                    }
                ],
                max_tokens=5,
            )
            return (stream.choices[0].message.content or "").strip().lower()

        result = await Consensus.run(
            [simple_response, simple_response],
            strategy="majority",
        )

        assert "yes" in result.consensus.lower()
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_best_strategy(self, client: "AsyncOpenAI") -> None:
        """Test best-of strategy (takes first result)."""

        async def get_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello'"}],
                max_tokens=5,
            )
            return stream.choices[0].message.content

        result = await Consensus.run(
            [get_response, get_response],
            strategy="best",
        )

        assert result.consensus is not None
        assert result.status == "success"


# ─────────────────────────────────────────────────────────────────────────────
# Preset Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestConsensusPresets:
    """Tests for consensus preset methods."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_strict_preset(self, client: "AsyncOpenAI") -> None:
        """Test Consensus.strict() preset."""

        async def fact_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "What is 5 + 5? Answer with just the number.",
                    }
                ],
                max_tokens=5,
            )
            return (stream.choices[0].message.content or "").strip()

        result = await Consensus.strict([fact_response, fact_response])

        assert "10" in result.consensus
        assert result.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_standard_preset(self, client: "AsyncOpenAI") -> None:
        """Test Consensus.standard() preset."""

        async def yes_no_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Is Python a programming language? Answer yes or no.",
                    }
                ],
                max_tokens=5,
            )
            return (stream.choices[0].message.content or "").strip().lower()

        result = await Consensus.standard(
            [yes_no_response, yes_no_response, yes_no_response]
        )

        assert "yes" in result.consensus
        assert result.status == "success"


# ─────────────────────────────────────────────────────────────────────────────
# Structured Consensus Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestStructuredConsensus:
    """Tests for structured consensus with Pydantic schemas."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_person_consensus(self, client: "AsyncOpenAI") -> None:
        """Test consensus with structured Person output."""
        import re

        def extract_json(text: str) -> str:
            """Extract JSON from potentially markdown-wrapped response."""
            # Try to extract from markdown code block
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                return match.group(1).strip()
            # Otherwise return as-is (might be pure JSON)
            return text.strip()

        async def get_person():
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return only valid JSON, no markdown: {"name": "Alice", "age": 30}',
                    }
                ],
                max_tokens=50,
            )
            content = response.choices[0].message.content or ""
            json_str = extract_json(content)
            return Person.model_validate_json(json_str)

        result = await Consensus.run(
            [get_person, get_person],
            strategy="unanimous",
            schema=Person,
        )

        assert result.type == "structured"
        assert result.field_consensus is not None
        assert "name" in result.field_consensus.fields
        assert "age" in result.field_consensus.fields

    @pytest.mark.asyncio
    async def test_math_result_consensus(self, client: "AsyncOpenAI") -> None:
        """Test consensus on math problem with structured output."""

        async def solve_math():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": """What is 15 + 27? Return JSON:
                        {"answer": <number>, "explanation": "<brief explanation>"}""",
                    }
                ],
                max_tokens=50,
            )
            content = stream.choices[0].message.content or ""
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return MathResult.model_validate_json(content.strip())

        result = await Consensus.run(
            [solve_math, solve_math],
            strategy="unanimous",
            schema=MathResult,
            threshold=0.9,
        )

        assert result.field_consensus is not None
        assert result.field_consensus.fields["answer"].value == 42


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Consensus Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestWeightedConsensus:
    """Tests for weighted consensus strategy."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_weighted_with_different_weights(self, client: "AsyncOpenAI") -> None:
        """Test weighted consensus with different weight assignments."""

        async def get_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'yes'"}],
                max_tokens=5,
            )
            return (stream.choices[0].message.content or "").strip()

        result = await Consensus.run(
            [get_response, get_response, get_response],
            strategy="weighted",
            weights=[1.0, 2.0, 1.0],
        )

        assert result.consensus is not None
        assert result.status == "success"


# ─────────────────────────────────────────────────────────────────────────────
# Observability Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestConsensusObservability:
    """Tests for consensus observability events."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_observability_events_emitted(self, client: "AsyncOpenAI") -> None:
        """Test that consensus emits observability events."""
        events_received = []

        def on_event(event: l0.ObservabilityEvent) -> None:
            events_received.append(event.type.value)

        async def simple_task():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=5,
            )
            return stream.choices[0].message.content

        await Consensus.run(
            [simple_task, simple_task],
            strategy="best",
            on_event=on_event,
        )

        assert "CONSENSUS_START" in events_received
        assert "CONSENSUS_STREAM_START" in events_received
        assert "CONSENSUS_STREAM_END" in events_received
        assert "CONSENSUS_OUTPUT_COLLECTED" in events_received
        assert "CONSENSUS_ANALYSIS" in events_received
        assert "CONSENSUS_RESOLUTION" in events_received
        assert "CONSENSUS_END" in events_received


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestConsensusAnalysis:
    """Tests for consensus analysis."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_result_has_analysis(self, client: "AsyncOpenAI") -> None:
        """Test that consensus result includes analysis."""

        async def get_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'analysis test'"}],
                max_tokens=10,
            )
            return stream.choices[0].message.content

        result = await Consensus.run(
            [get_response, get_response, get_response],
            strategy="majority",
        )

        assert result.analysis is not None
        assert result.analysis.total_outputs == 3
        assert result.analysis.successful_outputs >= 2
        assert result.analysis.strategy == "majority"
        assert result.analysis.duration_ms > 0

    @pytest.mark.asyncio
    async def test_similarity_matrix_computed(self, client: "AsyncOpenAI") -> None:
        """Test that similarity matrix is computed."""

        async def get_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello'"}],
                max_tokens=5,
            )
            return stream.choices[0].message.content

        result = await Consensus.run(
            [get_response, get_response],
            strategy="best",
        )

        matrix = result.analysis.similarity_matrix
        assert len(matrix) >= 2
        assert matrix[0][0] == 1.0
        assert matrix[1][1] == 1.0

    @pytest.mark.asyncio
    async def test_validate_consensus_result(self, client: "AsyncOpenAI") -> None:
        """Test Consensus.validate() helper."""

        async def fact_response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "What color is the sky? One word."}
                ],
                max_tokens=5,
            )
            return (stream.choices[0].message.content or "").strip().lower()

        result = await Consensus.run(
            [fact_response, fact_response, fact_response],
            strategy="unanimous",
            threshold=0.7,
        )

        is_valid = Consensus.validate(result, min_confidence=0.5)
        assert is_valid is True


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Resolution Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestConflictResolution:
    """Tests for conflict resolution strategies."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_vote_resolution(self, client: "AsyncOpenAI") -> None:
        """Test vote-based conflict resolution."""

        async def response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'vote'"}],
                max_tokens=5,
            )
            return stream.choices[0].message.content

        result = await Consensus.run(
            [response, response, response],
            strategy="majority",
            resolve_conflicts="vote",
        )

        assert result.consensus is not None
        assert result.analysis.conflict_resolution == "vote"

    @pytest.mark.asyncio
    async def test_best_resolution(self, client: "AsyncOpenAI") -> None:
        """Test best-of conflict resolution with weights."""

        async def response():
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'best'"}],
                max_tokens=5,
            )
            return stream.choices[0].message.content

        result = await Consensus.run(
            [response, response],
            strategy="majority",
            resolve_conflicts="best",
            weights=[1.0, 2.0],
            minimum_agreement=0.0,
        )

        assert result.consensus is not None
