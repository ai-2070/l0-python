# Structured Output Guide

Guaranteed valid JSON matching your schema. Supports Pydantic and JSON Schema.

## Quick Start

```python
from pydantic import BaseModel
import l0

class User(BaseModel):
    name: str
    age: int
    email: str

result = await l0.structured(
    schema=User,
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Generate a user profile as JSON"}],
        stream=True,
    ),
)

# Type-safe access
print(result.data.name)   # str
print(result.data.age)    # int
print(result.data.email)  # str
```

---

## Features

| Feature           | Description                                               |
| ----------------- | --------------------------------------------------------- |
| Schema validation | Pydantic and JSON Schema support                          |
| Auto-correction   | Fixes trailing commas, missing braces, markdown fences    |
| Retry on failure  | Automatic retry when validation fails                     |
| Fallback models   | Try cheaper models if primary fails                       |
| Type safety       | Full type inference from Pydantic schema                  |
| Helper functions  | `structured_object`, `structured_array` for common patterns |

---

## API

### structured(schema, stream, ...)

```python
from l0 import structured, Retry, Timeout

result = await structured(
    # Required
    schema=User,                    # Pydantic model class
    stream=lambda: client.chat.completions.create(...),

    # Optional
    fallbacks=[...],                # Fallback model streams
    auto_correct=True,              # Fix common JSON issues (default: True)
    strict_mode=False,              # Reject unknown fields (default: False)
    retry=Retry(attempts=2),        # Retry on validation failure (default: 1 attempt)
    detect_zero_tokens=False,       # Detect zero-token outputs (default: False)

    # Timeout (milliseconds)
    timeout=Timeout(
        initial_token=6000,         # Max wait for first token (default: 5000ms)
        inter_token=5000,           # Max gap between tokens (default: 10000ms)
    ),

    # Monitoring
    monitoring=False,               # Enable telemetry (default: False)

    # Callbacks
    on_validation_error=lambda error, attempt: ...,
    on_auto_correct=lambda info: ...,
    on_retry=lambda attempt, reason: ...,
    on_event=lambda event: ...,

    # Adapter
    adapter="openai",               # Optional adapter hint
)

# Result
result.data              # Validated Pydantic model instance
result.raw               # Raw JSON string
result.corrected         # bool - was auto-corrected
result.corrections       # list[str] - corrections applied
result.state             # L0 State with token counts, retries, etc.
result.structured_state  # StructuredState with validation metrics
result.telemetry         # StructuredTelemetry (if monitoring enabled)
result.errors            # list[Exception] - errors during retries
result.abort()           # Abort the stream
```

### structured_stream(schema, stream, ...)

Stream tokens with validation at the end:

```python
from l0 import structured_stream

stream, result = await structured_stream(
    schema=User,
    stream=lambda: client.chat.completions.create(...),
)

async for event in stream:
    if event.is_token:
        print(event.text, end="")

validated = await result.validate()
print(validated.data)
```

### structured_object(shape, stream, ...)

Helper for creating object schemas inline:

```python
from l0 import structured_object

result = await structured_object(
    {
        "amount": int,
        "approved": bool,
        "note": (str, ""),  # (type, default) for optional with default
    },
    stream=lambda: client.chat.completions.create(...),
)

print(result.data.amount)
print(result.data.approved)
```

### structured_array(item_schema, stream, ...)

Helper for creating array schemas:

```python
from l0 import structured_array
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

result = await structured_array(
    Item,
    stream=lambda: client.chat.completions.create(...),
)

for item in result.data:
    print(f"{item.name}: ${item.price}")
```

---

## Auto-Correction

Automatically fixes common LLM JSON issues:

| Issue              | Example                  | Fixed               |
| ------------------ | ------------------------ | ------------------- |
| Missing brace      | `{"name": "Alice"`       | `{"name": "Alice"}` |
| Missing bracket    | `[1, 2, 3`               | `[1, 2, 3]`         |
| Trailing comma     | `{"a": 1,}`              | `{"a": 1}`          |
| Markdown fence     | ` ```json {...} ``` `    | `{...}`             |
| Text prefix        | `Sure! {"a": 1}`         | `{"a": 1}`          |
| Single quotes      | `{'a': 1}`               | `{"a": 1}`          |
| Comments           | `{"a": 1 /* comment */}` | `{"a": 1}`          |
| Control characters | Unescaped newlines       | Escaped properly    |

### Correction Types

All correction types that can be applied:

- `close_brace` - Added missing closing brace
- `close_bracket` - Added missing closing bracket
- `remove_trailing_comma` - Removed trailing comma
- `strip_markdown_fence` - Removed markdown code fence
- `strip_json_prefix` - Removed "json" prefix
- `remove_prefix_text` - Removed text before JSON
- `remove_suffix_text` - Removed text after JSON
- `fix_quotes` - Fixed quote issues
- `remove_comments` - Removed JSON comments
- `escape_control_chars` - Escaped control characters
- `fill_missing_fields` - Added missing required fields
- `remove_unknown_fields` - Removed unknown fields (strict mode)
- `coerce_types` - Coerced types to match schema
- `extract_json` - Extracted JSON from surrounding text

```python
from l0 import structured, AutoCorrectInfo

def on_correction(info: AutoCorrectInfo):
    print(f"Original: {info.original}")
    print(f"Corrected: {info.corrected}")
    print(f"Corrections: {info.corrections}")
    print(f"Success: {info.success}")

result = await structured(
    schema=User,
    stream=stream,
    auto_correct=True,
    on_auto_correct=on_correction,
)

if result.corrected:
    print(f"Fixes applied: {result.corrections}")
```

---

## Schema Support

### Pydantic (Default)

```python
from pydantic import BaseModel
from l0 import structured

class User(BaseModel):
    name: str
    age: int

result = await structured(schema=User, stream=stream)
```

### JSON Schema

L0 supports JSON Schema via a user-provided adapter:

```python
from l0 import (
    structured,
    register_json_schema_adapter,
    wrap_json_schema,
    JSONSchemaValidationSuccess,
    JSONSchemaValidationFailure,
    JSONSchemaValidationError,
)

# Option 1: Use the built-in simple adapter (for basic schemas)
from l0.json_schema import SimpleJSONSchemaAdapter

register_json_schema_adapter(SimpleJSONSchemaAdapter())

# Option 2: Use jsonschema library (recommended for production)
import jsonschema

class JsonSchemaAdapter:
    def validate(self, schema, data):
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(data))
        if not errors:
            return JSONSchemaValidationSuccess(valid=True, data=data)
        return JSONSchemaValidationFailure(
            valid=False,
            errors=[
                JSONSchemaValidationError(
                    path="/".join(str(p) for p in e.absolute_path),
                    message=e.message,
                    keyword=e.validator,
                )
                for e in errors
            ],
        )

    def format_errors(self, errors):
        return "; ".join(f"{e.path}: {e.message}" for e in errors)

register_json_schema_adapter(JsonSchemaAdapter())

# Define your schema
user_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
    },
    "required": ["name", "age"],
}

# Use with structured() - note: returns dict, not Pydantic model
# For type safety, you'll need to handle the dict yourself
unified = wrap_json_schema(user_schema)
data = unified.parse({"name": "Alice", "age": 30})
```

---

## Pydantic Schema Examples

### Basic Types

```python
from pydantic import BaseModel

class Example(BaseModel):
    name: str
    age: int
    active: bool
    status: Literal["pending", "approved", "rejected"]
```

### Optional & Nullable

```python
from pydantic import BaseModel

class Example(BaseModel):
    name: str
    nickname: str | None = None      # Optional with default None
    middle_name: str | None           # Required but can be None
```

### Nested Objects

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    email: str
    address: Address
    metadata: dict[str, str] = {}
```

### Arrays

```python
from pydantic import BaseModel

class Example(BaseModel):
    tags: list[str]
    items: list[Item]  # List of nested objects
```

### Validation Constraints

```python
from pydantic import BaseModel, Field, EmailStr

class Example(BaseModel):
    amount: float = Field(gt=0, le=10000)
    email: EmailStr
    score: int = Field(ge=0, le=100)
    url: str = Field(pattern=r"^https?://")
```

---

## Fallback Models

```python
result = await structured(
    schema=User,
    stream=lambda: client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True,
    ),
    fallbacks=[
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        ),
        lambda: anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            stream=True,
        ),
    ],
)

if result.state and result.state.fallback_index > 0:
    print("Used fallback model")
```

---

## Error Handling

```python
from pydantic import ValidationError

try:
    result = await structured(
        schema=User,
        stream=stream,
        retry=Retry(attempts=3),
        on_validation_error=lambda error, attempt: (
            print(f"Attempt {attempt} failed: {error.error_count()} errors")
        ),
    )
except ValueError as e:
    # All retries exhausted
    print(f"Validation failed: {e}")
```

---

## State and Telemetry

### StructuredState

Extended state with validation metrics:

```python
from dataclasses import dataclass
from pydantic import ValidationError

@dataclass
class StructuredState:
    validation_failures: int          # Number of validation failures
    auto_corrections: int             # Number of auto-corrections applied
    validation_errors: list[ValidationError]  # Validation errors encountered
    correction_types: list[str]       # Types of corrections applied
    validation_time_ms: float | None  # Time spent on validation (ms)
```

### StructuredTelemetry

Telemetry with structured-specific metrics (when `monitoring=True`):

```python
from dataclasses import dataclass

@dataclass
class StructuredTelemetry:
    schema_name: str | None           # Schema name (e.g., "User")
    validation_attempts: int          # Number of validation attempts
    validation_failures: int          # Number of validation failures
    auto_corrections: int             # Number of auto-corrections applied
    correction_types: list[str]       # Types of corrections applied
    validation_success: bool          # Final validation success
    validation_time_ms: float | None  # Time spent on validation (ms)
```

---

## Presets

L0 provides configuration presets:

```python
from l0 import (
    structured,
    MINIMAL_STRUCTURED,
    RECOMMENDED_STRUCTURED,
    STRICT_STRUCTURED,
    StructuredConfig,
)

# Use preset values
result = await structured(
    schema=User,
    stream=stream,
    auto_correct=MINIMAL_STRUCTURED.auto_correct,
    retry=Retry(attempts=MINIMAL_STRUCTURED.attempts),
)

# Or spread preset configuration
# MINIMAL_STRUCTURED: auto_correct=False, attempts=1, strict_mode=False
# RECOMMENDED_STRUCTURED: auto_correct=True, attempts=2, strict_mode=False
# STRICT_STRUCTURED: auto_correct=True, attempts=3, strict_mode=True
```

### Preset Details

| Preset      | auto_correct | strict_mode | attempts |
| ----------- | ------------ | ----------- | -------- |
| minimal     | False        | False       | 1        |
| recommended | True         | False       | 2        |
| strict      | True         | True        | 3        |

---

## StructuredResult

The result object provides:

```python
@dataclass
class StructuredResult(Generic[T]):
    data: T                              # Validated Pydantic model instance
    raw: str                             # Raw JSON string before parsing
    corrected: bool                      # Whether auto-correction was applied
    corrections: list[str]               # List of corrections applied
    state: State | None                  # L0 runtime state
    structured_state: StructuredState | None  # Validation metrics
    telemetry: StructuredTelemetry | None     # Telemetry (if monitoring enabled)
    errors: list[Exception]              # Errors encountered during retries

    def abort(self) -> None:
        """Abort the structured stream."""
        ...

    @property
    def is_aborted(self) -> bool:
        """Check if abort was requested."""
        ...
```

---

## Best Practices

1. **Enable auto-correction** - Handles common LLM quirks
2. **Add fallback models** - Increases reliability
3. **Keep schemas focused** - Simpler schemas validate more reliably
4. **Monitor corrections** - Track what gets auto-corrected
5. **Use retry** - Transient failures are common
6. **Set `detect_zero_tokens=False`** - Default for structured output since valid JSON like `[]` or `{}` is acceptable

```python
from l0 import structured, Retry
import logging

logger = logging.getLogger(__name__)

# Recommended configuration
result = await structured(
    schema=User,
    stream=lambda: client.chat.completions.create(...),
    auto_correct=True,
    retry=Retry(attempts=2),
    fallbacks=[lambda: client.chat.completions.create(model="gpt-4o-mini", ...)],
    on_validation_error=lambda error, attempt: (
        logger.warning(f"Validation failed attempt {attempt}: {error.error_count()} errors")
    ),
)
```

---

## API Reference

### Functions

| Function | Description |
| -------- | ----------- |
| `structured(schema, stream, ...)` | Get validated structured output |
| `structured_stream(schema, stream, ...)` | Stream with validation at end |
| `structured_object(shape, stream, ...)` | Helper for inline object schemas |
| `structured_array(item_schema, stream, ...)` | Helper for array schemas |

### JSON Schema Functions

| Function | Description |
| -------- | ----------- |
| `register_json_schema_adapter(adapter)` | Register JSON Schema adapter |
| `unregister_json_schema_adapter()` | Unregister the adapter |
| `has_json_schema_adapter()` | Check if adapter is registered |
| `wrap_json_schema(schema)` | Wrap JSON Schema for use with L0 |

### Types

| Type | Description |
| ---- | ----------- |
| `StructuredResult[T]` | Result with validated data |
| `StructuredStreamResult[T]` | Streaming result holder |
| `StructuredState` | Validation metrics |
| `StructuredTelemetry` | Telemetry data |
| `StructuredConfig` | Configuration preset |
| `AutoCorrectInfo` | Auto-correction callback info |

### Presets

| Preset | Description |
| ------ | ----------- |
| `MINIMAL_STRUCTURED` | No auto-correction, single attempt |
| `RECOMMENDED_STRUCTURED` | Auto-correction, 2 attempts |
| `STRICT_STRUCTURED` | Auto-correction, strict mode, 3 attempts |
