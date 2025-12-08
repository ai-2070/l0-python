# Consensus

Multi-generation consensus for high-confidence results. Run multiple generations, compare outputs, and resolve disagreements.

## Quick Start

```python
from l0 import Consensus

result = await Consensus.run(
    [
        lambda: get_answer_from_model_a(),
        lambda: get_answer_from_model_b(),
        lambda: get_answer_from_model_c(),
    ],
)

print(result.consensus)       # Agreed output
print(result.confidence)      # 0-1 confidence score
print(result.agreements)      # What they agreed on
print(result.disagreements)   # Where they differed
```

## Strategies

```python
await Consensus.run(tasks, strategy="majority")  # Default
```

| Strategy | Behavior |
| -------- | -------- |
| `majority` | Take what most outputs agree on |
| `unanimous` | All must agree (fails otherwise) |
| `weighted` | Weight by model/confidence |
| `best` | Choose highest quality output |

## Conflict Resolution

```python
await Consensus.run(tasks, resolve_conflicts="vote")  # Default
```

| Resolution | Behavior |
| ---------- | -------- |
| `vote` | Take majority vote |
| `merge` | Combine all information |
| `best` | Choose highest confidence |
| `fail` | Raise error on disagreement |

---

## Scoped API

The `Consensus` class provides a clean scoped API with presets, utility methods, and similarity functions:

```python
from l0 import Consensus

# Run consensus
result = await Consensus.run(tasks, strategy="majority")

# Presets (async methods)
result = await Consensus.strict(tasks)    # All must agree
result = await Consensus.standard(tasks)  # Majority rules (default)
result = await Consensus.lenient(tasks)   # Flexible matching
result = await Consensus.best(tasks)      # Pick best single output

# Presets (config objects)
Consensus.STRICT    # ConsensusPreset for unanimous agreement
Consensus.STANDARD  # ConsensusPreset for majority rules
Consensus.LENIENT   # ConsensusPreset for flexible matching
Consensus.BEST      # ConsensusPreset for best output

# Quick check (synchronous)
if Consensus.quick(outputs, threshold=0.8):
    print("Consensus reached!")

# Get most common value
value = Consensus.get_value(outputs)

# Validate result
if Consensus.validate(result, min_confidence=0.8):
    print("Valid consensus")
```

---

## Structured Consensus

With Pydantic schema, consensus compares field-by-field:

```python
from pydantic import BaseModel
from l0 import Consensus

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

result = await Consensus.run(
    [
        lambda: get_structured_answer_a(),
        lambda: get_structured_answer_b(),
        lambda: get_structured_answer_c(),
    ],
    schema=Answer,
    strategy="majority",
)

# Type-safe access
print(result.consensus.answer)
print(result.field_consensus.fields["answer"].agreement)   # 0-1
print(result.field_consensus.fields["answer"].unanimous)   # True/False
```

---

## Options

```python
result = await Consensus.run(
    tasks,                             # Required: list of async callables (min 2)
    strategy="majority",               # Consensus strategy
    threshold=0.8,                     # Similarity threshold (0-1)
    resolve_conflicts="vote",          # How to resolve disagreements
    weights=[1.0, 1.0, 1.0],           # Weights for weighted strategy
    minimum_agreement=0.6,             # Min agreement ratio required
    schema=MySchema,                   # Pydantic schema for structured consensus
    on_event=my_event_handler,         # Observability callback
)
```

### Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `tasks` | `list[Callable[[], Awaitable[T]]]` | required | Async callables (min 2) |
| `strategy` | `Strategy` | `"majority"` | Consensus strategy |
| `threshold` | `float` | `0.8` | Similarity threshold |
| `resolve_conflicts` | `ConflictResolution` | `"vote"` | Conflict resolution method |
| `weights` | `list[float]` | `None` | Weights for each task |
| `minimum_agreement` | `float` | `0.6` | Min agreement ratio |
| `schema` | `type[BaseModel]` | `None` | Pydantic schema for structured consensus |
| `on_event` | `Callable` | `None` | Observability event callback |

---

## Result Structure

### ConsensusResult

```python
@dataclass
class ConsensusResult(Generic[T]):
    consensus: T                           # Final agreed output
    confidence: float                      # 0-1 overall confidence
    outputs: list[ConsensusOutput]         # Individual outputs
    agreements: list[Agreement]            # What matched
    disagreements: list[Disagreement]      # What differed
    analysis: ConsensusAnalysis            # Detailed stats
    type: Literal["text", "structured"]    # Result type
    field_consensus: FieldConsensus | None # For structured (field-by-field)
    status: Literal["success", "partial", "failed"]
```

### ConsensusOutput

```python
@dataclass
class ConsensusOutput:
    index: int                             # Output index
    text: str                              # Raw text output
    value: Any                             # Parsed value
    success: bool                          # Whether task succeeded
    data: Any                              # Parsed data (if structured)
    l0_result: Any                         # L0 result (if text-based)
    structured_result: Any                 # Structured result (if schema)
    error: str | None                      # Error message if failed
    duration_ms: float                     # Execution duration (ms)
    weight: float                          # Weight assigned
    similarities: list[float] | None       # Similarity scores with other outputs
```

### Agreement

```python
@dataclass
class Agreement:
    content: Any                           # Agreed content
    path: str | None                       # Field path (for structured)
    count: int                             # How many agreed
    ratio: float                           # Agreement ratio (0-1)
    indices: list[int]                     # Which outputs agreed
    type: AgreementType                    # "exact" | "similar" | "structural" | "semantic"
```

### Disagreement

```python
@dataclass
class Disagreement:
    path: str | None                       # Field path (for structured)
    values: list[DisagreementValue]        # Different values seen
    severity: DisagreementSeverity         # "minor" | "moderate" | "major" | "critical"
    resolution: str | None                 # How it was resolved
    resolution_confidence: float | None    # Confidence in resolution

@dataclass
class DisagreementValue:
    value: Any                             # The value
    count: int                             # How many had this value
    indices: list[int]                     # Which outputs
```

### Disagreement Severity

| Severity | Agreement Ratio | Description |
| -------- | --------------- | ----------- |
| `minor` | >= 80% | Strong majority |
| `moderate` | >= 60% | Weak majority |
| `major` | >= 40% | No clear majority |
| `critical` | < 40% | Complete split |

### ConsensusAnalysis

```python
@dataclass
class ConsensusAnalysis:
    total_outputs: int                     # Total tasks
    successful_outputs: int                # Tasks that succeeded
    failed_outputs: int                    # Tasks that failed
    identical_outputs: int                 # Exact matches
    similarity_matrix: list[list[float]]   # Pairwise similarities
    average_similarity: float              # Mean similarity
    min_similarity: float                  # Minimum similarity
    max_similarity: float                  # Maximum similarity
    total_agreements: int                  # Number of agreements
    total_disagreements: int               # Number of disagreements
    strategy: str                          # Strategy used
    conflict_resolution: str               # Resolution method used
    duration_ms: float                     # Total duration
```

### FieldConsensus

```python
@dataclass
class FieldConsensus:
    fields: dict[str, FieldAgreement]      # Per-field consensus info
    overall_agreement: float               # 0-1 overall field agreement
    agreed_fields: list[str]               # Fields with full agreement
    disagreed_fields: list[str]            # Fields with disagreement

@dataclass
class FieldAgreement:
    path: str                              # Field path
    value: Any                             # Consensus value for this field
    agreement: float                       # 0-1 agreement ratio
    votes: dict[str, int]                  # Vote counts per value
    values: list[Any]                      # All values seen
    unanimous: bool                        # Full agreement?
    confidence: float                      # 0-1 confidence
```

---

## Presets

```python
from l0 import Consensus

# Strict: all must agree
result = await Consensus.strict(tasks)
# Equivalent to:
# strategy="unanimous", threshold=1.0, resolve_conflicts="fail", minimum_agreement=1.0

# Standard: majority rules (default)
result = await Consensus.standard(tasks)
# Equivalent to:
# strategy="majority", threshold=0.8, resolve_conflicts="vote", minimum_agreement=0.6

# Lenient: flexible
result = await Consensus.lenient(tasks)
# Equivalent to:
# strategy="majority", threshold=0.7, resolve_conflicts="merge", minimum_agreement=0.5

# Best: choose highest quality
result = await Consensus.best(tasks)
# Equivalent to:
# strategy="best", threshold=0.5, resolve_conflicts="best", minimum_agreement=0.0
```

### Preset Configurations

| Preset | Strategy | Threshold | Resolve Conflicts | Min Agreement |
| ------ | -------- | --------- | ----------------- | ------------- |
| `STRICT` | `unanimous` | 1.0 | `fail` | 1.0 |
| `STANDARD` | `majority` | 0.8 | `vote` | 0.6 |
| `LENIENT` | `majority` | 0.7 | `merge` | 0.5 |
| `BEST` | `best` | 0.8 | `best` | 0.5 |

---

## Helper Functions

### Quick Consensus Check

```python
from l0 import Consensus, quick_consensus

outputs = ["answer A", "answer A", "answer B"]

# Using scoped API
Consensus.quick(outputs)        # False (not 80% agreement)
Consensus.quick(outputs, 0.6)   # True (66% >= 60%)

# Using module-level function
quick_consensus(outputs)        # False
quick_consensus(outputs, 0.6)   # True
```

### Get Consensus Value

```python
from l0 import Consensus, get_consensus_value

# Using scoped API
Consensus.get_value(["A", "A", "B"])   # "A"
Consensus.get_value([1, 2, 1, 1])      # 1

# Using module-level function
get_consensus_value(["A", "A", "B"])   # "A"
```

### Validate Consensus

```python
from l0 import Consensus, validate_consensus

# Using scoped API
Consensus.validate(result, min_confidence=0.8, max_disagreements=0)
# Returns True if confidence >= 0.8 and no major/critical disagreements

# Using module-level function
validate_consensus(result, min_confidence=0.8, max_disagreements=0)
```

---

## Multi-Model Consensus

Use different models for diverse perspectives:

```python
from openai import AsyncOpenAI
import l0

client = AsyncOpenAI()

async def get_gpt4_answer():
    result = await l0.run(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    )
    return await result.read()

async def get_gpt4_mini_answer():
    result = await l0.run(
        stream=lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ),
    )
    return await result.read()

result = await Consensus.run(
    [
        get_gpt4_answer,
        get_gpt4_answer,       # Run GPT-4o twice
        get_gpt4_mini_answer,
    ],
    strategy="majority",
)
```

## Weighted Consensus

Weight models differently:

```python
result = await Consensus.run(
    [
        lambda: get_expert_model_answer(),   # Expert model
        lambda: get_fast_model_answer(),     # Fast model
        lambda: get_fast_model_answer(),     # Fast model
    ],
    strategy="weighted",
    weights=[2.0, 1.0, 1.0],  # Expert counts double
)
```

---

## Use Cases

### Factual Accuracy

```python
# Ask same question 3 times, require unanimous agreement
result = await Consensus.strict([
    lambda: ask_model("What year was Python created?"),
    lambda: ask_model("What year was Python created?"),
    lambda: ask_model("What year was Python created?"),
])
```

### Code Generation

```python
# Generate code 3 times, pick best
result = await Consensus.best([
    lambda: generate_code(prompt),
    lambda: generate_code(prompt),
    lambda: generate_code(prompt),
])
```

### Data Extraction

```python
from pydantic import BaseModel
from l0 import Consensus

class ExtractedData(BaseModel):
    name: str
    email: str
    company: str

# Extract data with schema, field-by-field consensus
result = await Consensus.run(
    [
        lambda: extract_data(document),
        lambda: extract_data(document),
        lambda: extract_data(document),
    ],
    schema=ExtractedData,
    minimum_agreement=0.8,  # 80% must agree on each field
)

# Check per-field agreement
for field_name, info in result.field_consensus.fields.items():
    print(f"{field_name}: {info.agreement * 100:.0f}% agreement")
    if info.unanimous:
        print(f"  Unanimous: {info.value}")
```

---

## Similarity Utilities

Low-level comparison utilities used by consensus, available via the scoped API:

```python
from l0 import Consensus, ConsensusOutput

# Calculate pairwise similarity between outputs
matrix = Consensus.similarity_matrix(outputs)

# Compare two outputs
similarity = Consensus.output_similarity(output_a, output_b)

# Compare two structured objects
similarity = Consensus.structural_similarity(obj1, obj2)  # 0-1

# Find all agreements above threshold
agreements = Consensus.agreements(outputs, threshold=0.8)

# Find all disagreements
disagreements = Consensus.disagreements(outputs, threshold=0.8)

# Calculate field-level consensus
field_consensus = Consensus.field_consensus(outputs)

# Resolve using different strategies
majority_output = Consensus.majority(outputs, weights)
best_output = Consensus.best_output(outputs, weights)
merged_output = Consensus.merge(outputs)

# Check if consensus meets threshold
meets = Consensus.meets_agreement(agreements, len(outputs), threshold=0.8)
```

### Module-Level Functions

The same utilities are also available as standalone functions:

```python
from l0 import (
    calculate_similarity_matrix,
    calculate_output_similarity,
    calculate_structural_similarity,
    find_agreements,
    find_disagreements,
    calculate_field_consensus,
    resolve_majority,
    resolve_best,
    resolve_merge,
    meets_minimum_agreement,
)
```

---

## Types

```python
from l0 import (
    # Main types
    Consensus,
    ConsensusResult,
    ConsensusOutput,
    ConsensusAnalysis,
    ConsensusPreset,
    
    # Agreement types
    Agreement,
    Disagreement,
    DisagreementValue,
    
    # Field consensus
    FieldConsensus,
    FieldAgreement,
    FieldConsensusInfo,  # Alias for FieldAgreement
    
    # Convenience function
    consensus,  # Alias for Consensus.run
)

# Type aliases
Strategy = Literal["unanimous", "majority", "weighted", "best"]
ConflictResolution = Literal["vote", "merge", "best", "fail"]
AgreementType = Literal["exact", "similar", "structural", "semantic"]
DisagreementSeverity = Literal["minor", "moderate", "major", "critical"]
```
