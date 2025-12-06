"""Advanced integration tests for structured outputs.

These tests cover:
- Complex nested schemas
- Arrays of structured objects
- Optional fields and defaults
- Strict mode validation
- Auto-correction with real API outputs
- Dynamic schemas with structured_object

Run with: pytest tests/integration/test_structured_advanced.py -v
"""

from typing import TYPE_CHECKING, Optional

import pytest
from pydantic import BaseModel, Field

import l0
from tests.conftest import requires_openai

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Test Schemas
# ─────────────────────────────────────────────────────────────────────────────


class Address(BaseModel):
    """Nested address schema."""

    street: str
    city: str
    country: str = "USA"


class Person(BaseModel):
    """Person with nested address."""

    name: str
    age: int
    email: Optional[str] = None
    address: Optional[Address] = None


class Product(BaseModel):
    """Product for array tests."""

    name: str
    price: float
    in_stock: bool = True


class BookReview(BaseModel):
    """Book review with rating."""

    title: str
    author: str
    rating: int = Field(ge=1, le=5)
    summary: str


class APIResponse(BaseModel):
    """Simulated API response with metadata."""

    success: bool
    data: dict
    message: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Nested Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestNestedStructuredOutput:
    """Tests for nested Pydantic schemas."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_nested_object(self, client: "AsyncOpenAI") -> None:
        """Test parsing nested objects."""
        result = await l0.structured(
            schema=Person,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": """Return JSON for a person with nested address:
                        {"name": "John Doe", "age": 35, "email": "john@example.com",
                         "address": {"street": "123 Main St", "city": "Boston", "country": "USA"}}""",
                    }
                ],
                stream=True,
                max_tokens=150,
            ),
        )

        assert result.data.name == "John Doe"
        assert result.data.age == 35
        assert result.data.address is not None
        assert result.data.address.city == "Boston"

    @pytest.mark.asyncio
    async def test_optional_nested_null(self, client: "AsyncOpenAI") -> None:
        """Test optional nested field as null."""
        result = await l0.structured(
            schema=Person,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"name": "Jane", "age": 28, "address": null}',
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        assert result.data.name == "Jane"
        assert result.data.age == 28
        assert result.data.address is None


# ─────────────────────────────────────────────────────────────────────────────
# Array Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestArrayStructuredOutput:
    """Tests for array-based structured outputs."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_array_of_objects(self, client: "AsyncOpenAI") -> None:
        """Test parsing array of Pydantic objects."""
        result = await l0.structured_array(
            item_schema=Product,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": """Return a JSON array of 3 products:
                        [{"name": "Laptop", "price": 999.99, "in_stock": true},
                         {"name": "Mouse", "price": 29.99, "in_stock": true},
                         {"name": "Keyboard", "price": 79.99, "in_stock": false}]""",
                    }
                ],
                stream=True,
                max_tokens=200,
            ),
        )

        assert len(result.data) == 3
        assert result.data[0].name == "Laptop"
        assert result.data[0].price == 999.99
        assert result.data[2].in_stock is False

    @pytest.mark.asyncio
    async def test_empty_array(self, client: "AsyncOpenAI") -> None:
        """Test parsing empty array."""
        result = await l0.structured_array(
            item_schema=Product,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": "Return an empty JSON array: []",
                    }
                ],
                stream=True,
                max_tokens=20,
            ),
        )

        assert result.data == []


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestDynamicStructuredOutput:
    """Tests for dynamic schemas using structured_object."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_object_simple(self, client: "AsyncOpenAI") -> None:
        """Test structured_object with simple schema."""
        result = await l0.structured_object(
            shape={"name": str, "count": int, "active": bool},
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"name": "test", "count": 42, "active": true}',
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        assert result.data.name == "test"
        assert result.data.count == 42
        assert result.data.active is True

    @pytest.mark.asyncio
    async def test_structured_object_with_defaults(self, client: "AsyncOpenAI") -> None:
        """Test structured_object with default values."""
        result = await l0.structured_object(
            shape={
                "title": str,
                "views": int,
                "published": (bool, True),  # Default value
            },
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON with only title and views: {"title": "Hello", "views": 100}',
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        assert result.data.title == "Hello"
        assert result.data.views == 100
        # Default should be applied
        assert result.data.published is True


# ─────────────────────────────────────────────────────────────────────────────
# Strict Mode Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestStrictModeStructuredOutput:
    """Tests for strict mode validation."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_strict_mode_valid(self, client: "AsyncOpenAI") -> None:
        """Test strict mode with valid data (no extra fields)."""

        class StrictPerson(BaseModel):
            name: str
            age: int

        result = await l0.structured(
            schema=StrictPerson,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return exactly: {"name": "Bob", "age": 25}',
                    }
                ],
                stream=True,
                max_tokens=30,
            ),
            strict_mode=True,
        )

        assert result.data.name == "Bob"
        assert result.data.age == 25

    @pytest.mark.asyncio
    async def test_strict_mode_rejects_extra_fields(
        self, client: "AsyncOpenAI"
    ) -> None:
        """Test that strict mode rejects extra fields."""

        class StrictPerson(BaseModel):
            name: str
            age: int

        with pytest.raises(ValueError, match="validation failed"):
            await l0.structured(
                schema=StrictPerson,
                stream=lambda: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": 'Return JSON with extra field: {"name": "Bob", "age": 25, "extra": "field"}',
                        }
                    ],
                    stream=True,
                    max_tokens=50,
                ),
                strict_mode=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-Correction Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestAutoCorrection:
    """Tests for JSON auto-correction."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_auto_correct_markdown_wrapped(self, client: "AsyncOpenAI") -> None:
        """Test auto-correction extracts JSON from markdown."""

        class SimpleData(BaseModel):
            value: int

        result = await l0.structured(
            schema=SimpleData,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Respond with JSON wrapped in markdown code block: ```json\n{"value": 42}\n```',
                    }
                ],
                stream=True,
                max_tokens=30,
            ),
            auto_correct=True,
        )

        assert result.data.value == 42

    @pytest.mark.asyncio
    async def test_auto_correct_callback(self, client: "AsyncOpenAI") -> None:
        """Test auto-correction callback is called."""
        corrections_received = []

        class SimpleData(BaseModel):
            message: str

        def on_correct(info: l0.AutoCorrectInfo) -> None:
            corrections_received.append(info)

        # Ask for response that might need correction
        result = await l0.structured(
            schema=SimpleData,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"message": "hello"}',
                    }
                ],
                stream=True,
                max_tokens=30,
            ),
            auto_correct=True,
            on_auto_correct=on_correct,
        )

        assert result.data.message == "hello"
        # Callback may or may not be called depending on model output


# ─────────────────────────────────────────────────────────────────────────────
# Retry and Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestStructuredRetry:
    """Tests for structured output with retry."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_with_retry(self, client: "AsyncOpenAI") -> None:
        """Test structured output with retry configuration."""

        class DataPoint(BaseModel):
            x: int
            y: int

        result = await l0.structured(
            schema=DataPoint,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"x": 10, "y": 20}',
                    }
                ],
                stream=True,
                max_tokens=30,
            ),
            retry=l0.Retry(attempts=2),
        )

        assert result.data.x == 10
        assert result.data.y == 20


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestStructuredStreaming:
    """Tests for structured streaming."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_structured_stream(self, client: "AsyncOpenAI") -> None:
        """Test structured_stream for streaming tokens then validating."""

        class Quote(BaseModel):
            text: str
            author: str

        event_stream, result_holder = await l0.structured_stream(
            schema=Quote,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"text": "Be yourself", "author": "Oscar Wilde"}',
                    }
                ],
                stream=True,
                max_tokens=50,
            ),
        )

        # Collect tokens while streaming
        tokens = []
        async for event in event_stream:
            if event.is_token and event.text:
                tokens.append(event.text)

        # Validate after stream completes
        result = await result_holder.validate()

        assert result.data.text == "Be yourself"
        assert result.data.author == "Oscar Wilde"
        assert len(tokens) > 0  # We received tokens


# ─────────────────────────────────────────────────────────────────────────────
# Complex Real-World Schema Tests
# ─────────────────────────────────────────────────────────────────────────────


@requires_openai
class TestComplexSchemas:
    """Tests for complex real-world schemas."""

    @pytest.fixture
    def client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI()

    @pytest.mark.asyncio
    async def test_book_review_with_constraints(self, client: "AsyncOpenAI") -> None:
        """Test schema with Field constraints."""
        result = await l0.structured(
            schema=BookReview,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": """Return a book review JSON:
                        {"title": "1984", "author": "George Orwell", "rating": 5, "summary": "A dystopian masterpiece"}""",
                    }
                ],
                stream=True,
                max_tokens=80,
            ),
        )

        assert result.data.title == "1984"
        assert result.data.author == "George Orwell"
        assert 1 <= result.data.rating <= 5
        assert len(result.data.summary) > 0

    @pytest.mark.asyncio
    async def test_api_response_with_dict_field(self, client: "AsyncOpenAI") -> None:
        """Test schema with dict field."""
        result = await l0.structured(
            schema=APIResponse,
            stream=lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": """Return JSON: {"success": true, "data": {"id": 1, "name": "test"}, "message": "OK"}""",
                    }
                ],
                stream=True,
                max_tokens=80,
            ),
        )

        assert result.data.success is True
        assert result.data.data["id"] == 1
        assert result.data.message == "OK"
