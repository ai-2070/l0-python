# Contributing to L0

Thank you for your interest in contributing to L0! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Scope Policy](#scope-policy)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Adding New Features](#adding-new-features)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Scope Policy

L0 does not accept integrations, drivers, storage adapters, or external service bindings into the core repository. These must live in separate repositories, maintained by their authors.

All adapters must be maintained out-of-tree. The L0 core will remain small, dependency-free, and integration-agnostic.

**What belongs in core:**

- Runtime features (retry, fallback, continuation, drift detection)
- Guardrail rules and engine
- Format helpers
- Type definitions
- Core utilities

**What belongs in separate repos:**

- Database adapters (Redis, PostgreSQL, MongoDB, etc.)
- Cloud service integrations (AWS, GCP, Azure)
- Monitoring backends (Datadog, custom exporters, etc.)
- LLM provider-specific extensions

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/ai-2070/l0-python.git
   cd l0-python
   ```
3. **Install dependencies** (see [Development Setup](#development-setup))
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install dependencies

Using uv (recommended):

```bash
uv sync --extra dev
```

Or using pip:

```bash
pip install -e ".[dev]"
```

### Optional extras

```bash
# OpenAI support
uv sync --extra openai

# LiteLLM support (100+ providers)
uv sync --extra litellm

# Observability (OpenTelemetry, Sentry)
uv sync --extra observability

# All extras
uv sync --all-extras
```

### Run tests

```bash
pytest
```

### Run type checking

```bash
mypy src/l0
```

### Run linting

```bash
ruff check src/l0 tests
ruff format src/l0 tests
```

## Project Structure

```
src/l0/
├── __init__.py          # Main entry point and public API
├── types.py             # Core type definitions
├── runtime.py           # Main l0.run() runtime
├── stream.py            # Stream handling
├── retry.py             # Retry logic with backoff strategies
├── guardrails.py        # Guardrail rules and engine
├── errors.py            # Error types and categorization
├── client.py            # Client utilities
├── adapters.py          # Provider adapters (OpenAI, LiteLLM)
├── consensus.py         # Multi-model consensus
├── continuation.py      # Stream continuation
├── parallel.py          # Parallel execution (race, batch)
├── structured.py        # Structured output with Pydantic
├── state.py             # State management
├── events.py            # Event definitions
├── window.py            # Token windowing
├── multimodal.py        # Multimodal support
├── format.py            # Format namespace
├── logging.py           # Logging utilities
├── version.py           # Version info
├── _utils.py            # Internal utilities
├── formatting/          # Format helpers
│   ├── context.py
│   ├── memory.py
│   ├── output.py
│   ├── strings.py
│   └── tools.py
├── monitoring/          # Observability
│   ├── config.py
│   ├── exporter.py
│   ├── monitor.py
│   ├── otel.py
│   ├── sentry.py
│   └── telemetry.py
└── eventsourcing/       # Event sourcing
    ├── adapters.py
    ├── recorder.py
    ├── replayer.py
    ├── sourcing.py
    ├── store.py
    └── types.py

tests/
├── conftest.py          # Pytest fixtures
├── test_*.py            # Unit tests
└── integration/         # Integration tests
    ├── test_openai.py
    └── test_litellm.py
```

## Making Changes

### 1. Choose what to work on

- Check [GitHub Issues](https://github.com/ai-2070/l0-python/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Discuss major changes in an issue first

### 2. Write your code

- Follow the [Coding Standards](#coding-standards)
- Add type hints for all public APIs
- Include docstrings for functions and classes
- Keep functions small and focused

### 3. Test your changes

- Add tests for new functionality
- Ensure existing tests pass
- Test with real LLM APIs if possible

## Testing

### Unit Tests

Place tests in `tests/` directory with `test_` prefix:

```python
# tests/test_guardrails.py
import pytest
from l0 import Guardrails
from l0.types import State

def test_json_rule_detects_unbalanced_braces():
    rule = Guardrails.json()
    state = State(content='{"name": "Alice"')
    violations = rule.check(state)
    assert len(violations) > 0
```

### Async Tests

Use `pytest-asyncio` for async tests:

```python
# tests/test_runtime.py
import pytest
from l0 import run

@pytest.mark.asyncio
async def test_run_with_guardrails():
    result = await run(
        stream=lambda: mock_stream('{"test": true}'),
        guardrails=Guardrails.recommended(),
    )
    text = await result.read()
    assert text == '{"test": true}'
```

### Integration Tests

Integration tests live in `tests/integration/`:

```python
# tests/integration/test_openai.py
import pytest
from openai import AsyncOpenAI
import l0

@pytest.mark.asyncio
async def test_openai_streaming():
    client = AsyncOpenAI()
    result = await l0.run(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
        ),
    )
    text = await result.read()
    assert len(text) > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=l0 --cov-report=term-missing

# Run specific test file
pytest tests/test_guardrails.py

# Run specific test
pytest tests/test_guardrails.py::test_json_rule_detects_unbalanced_braces

# Run integration tests (requires API keys)
pytest tests/integration/
```

## Submitting Changes

### Before submitting

1. **Run tests**: `pytest`
2. **Run type checker**: `mypy src/l0`
3. **Run linter**: `ruff check src/l0 tests`
4. **Format code**: `ruff format src/l0 tests`
5. **Update documentation**: Update README.md or API.md if needed

### Pull Request Process

1. **Push your branch** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changed and why
   - Include examples if applicable

3. **Respond to feedback**:
   - Address review comments
   - Push updates to your branch
   - Be open to suggestions

4. **Wait for approval**:
   - Maintainers will review your PR
   - CI checks must pass
   - At least one approval required

## Coding Standards

### Python Style

```python
# Good: Clear types, docstrings
def has_meaningful_content(content: str) -> bool:
    """Check if content is meaningful.

    Args:
        content: Content to check.

    Returns:
        True if meaningful.
    """
    if not content:
        return False
    return content.strip().length > 0


# Bad: No types, no docs
def check(c):
    return c and c.strip().length > 0
```

### Naming Conventions

- **Functions/methods**: `snake_case` - `format_context`, `detect_drift`
- **Classes**: `PascalCase` - `RetryManager`, `DriftDetector`
- **Constants**: `UPPER_SNAKE_CASE` - `BAD_PATTERNS`, `MAX_RETRIES`
- **Type aliases**: `PascalCase` - `L0Options`, `GuardrailRule`
- **Files**: `snake_case.py` - `retry.py`, `zero_token.py`
- **Private**: `_prefix` - `_internal_helper`, `_utils.py`

### Code Organization

1. **Imports first**, in order:

   ```python
   # Standard library
   from collections.abc import AsyncIterator
   from typing import Any

   # Third-party
   from pydantic import BaseModel

   # Local imports
   from l0.types import State
   from l0._utils import helper
   ```

2. **Types before implementation**:

   ```python
   class MyOptions(BaseModel):
       enabled: bool = True


   def my_function(options: MyOptions) -> str:
       # implementation
       ...
   ```

3. **Export in `__init__.py`**:

   ```python
   # src/l0/__init__.py
   from l0.runtime import run
   from l0.guardrails import Guardrails

   __all__ = ["run", "Guardrails"]
   ```

### Documentation

- All public functions need docstrings
- Use Google-style docstrings
- Include `Args`, `Returns`, and `Raises` sections
- Add `Example` for complex functions

Example:

```python
def format_tool(
    tool: ToolDefinition,
    options: FormatToolOptions | None = None,
) -> str:
    """Format tool/function definition in a model-friendly way.

    Args:
        tool: Tool definition to format.
        options: Formatting options.

    Returns:
        Formatted tool definition string.

    Raises:
        ValueError: If tool definition is invalid.

    Example:
        >>> tool = create_tool("get_weather", "Get weather", [])
        >>> formatted = format_tool(tool, FormatToolOptions(style="json-schema"))
    """
    # implementation
    ...
```

## Adding New Features

### Adding a Guardrail Rule

1. Add the rule to `src/l0/guardrails.py`:

   ```python
   def my_rule() -> GuardrailRule:
       """Create a rule that checks for X."""

       def check(state: State) -> list[GuardrailViolation]:
           violations: list[GuardrailViolation] = []
           # Your validation logic
           return violations

       return GuardrailRule(
           name="my-rule",
           check=check,
           streaming=True,
           severity="error",
       )
   ```

2. Add tests in `tests/test_guardrails.py`

3. Export from `src/l0/__init__.py` if public

4. Add to preset if appropriate

5. Document in API.md

### Adding a Format Helper

1. Create function in appropriate `src/l0/formatting/` file

2. Add type hints

3. Add docstring

4. Export from `src/l0/formatting/__init__.py`

5. Add to `src/l0/format.py` namespace if needed

6. Document in API.md with examples

### Adding a Utility Function

1. Add to `src/l0/_utils.py` (internal) or appropriate module

2. Keep functions pure (no side effects)

3. Add comprehensive tests

4. Export and document if public

## Type Hints

- Always provide explicit type hints
- Avoid `Any` - use `object` or specific types if possible
- Use `|` for unions (Python 3.10+)
- Use `collections.abc` for abstract types

```python
# Good
from collections.abc import AsyncIterator, Callable

def process(
    config: MyConfig,
    handler: Callable[[str], None] | None = None,
) -> AsyncIterator[Event]:
    ...


# Bad
def process(config, handler=None):
    ...
```

## Error Handling

- Use specific exception classes
- Provide clear error messages
- Include context in error messages

```python
if not content:
    raise ValueError("Content is required and cannot be empty")

# Use custom exceptions for domain errors
from l0.errors import L0Error, GuardrailViolationError

raise GuardrailViolationError(
    rule="json",
    message="Invalid JSON: unbalanced braces",
)
```

## Performance Considerations

- Avoid unnecessary iterations
- Use early returns to avoid deep nesting
- Consider memory usage for streaming operations
- Use `async for` for async iteration

```python
# Good: Early return
def check(content: str) -> bool:
    if not content:
        return False
    if len(content) < 10:
        return False
    return perform_expensive_check(content)


# Bad: Nested conditions
def check(content: str) -> bool:
    if content:
        if len(content) >= 10:
            return perform_expensive_check(content)
    return False
```

## Questions?

- Open a [GitHub Issue](https://github.com/ai-2070/l0-python/issues)
- Start a [Discussion](https://github.com/ai-2070/l0-python/discussions)
- Check existing issues and discussions

## Recognition

All contributors will be recognized in the project. Thank you for making L0 better!

---

Happy contributing!
