# Guardrails

Guardrails are pure functions that validate streaming output without rewriting it. They detect issues and signal whether to retry.

## Quick Start

```python
import l0

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=l0.Guardrails.recommended(),
)
```

## Presets

```python
from l0 import Guardrails

# Presets return fresh rule lists
Guardrails.minimal()        # JSON + zero output
Guardrails.recommended()    # + Markdown, patterns
Guardrails.strict()         # + LaTeX, drift detection
Guardrails.json_only()      # JSON + strict JSON + zero output
Guardrails.markdown_only()  # Markdown + zero output
Guardrails.latex_only()     # LaTeX + zero output
Guardrails.none()           # Explicit opt-out
```

Or use the preset constants for TypeScript API parity:

```python
from l0.guardrails import (
    MINIMAL_GUARDRAILS,
    RECOMMENDED_GUARDRAILS,
    STRICT_GUARDRAILS,
    JSON_ONLY_GUARDRAILS,
    MARKDOWN_ONLY_GUARDRAILS,
    LATEX_ONLY_GUARDRAILS,
)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=RECOMMENDED_GUARDRAILS(),
)
```

---

## Built-in Rules

### JSON Rule

Validates JSON structure during streaming:

```python
from l0 import Guardrails

Guardrails.json()        # Balanced braces/brackets, streaming-aware
Guardrails.strict_json() # + Must be parseable, root must be object/array
```

**Detects:**

- Unbalanced `{}` and `[]`
- Unclosed strings
- Multiple consecutive commas
- Malformed patterns like `{,` or `[,`

### Markdown Rule

Validates Markdown structure:

```python
Guardrails.markdown()
```

**Detects:**

- Unclosed code fences (```)
- Inconsistent table columns

### LaTeX Rule

Validates LaTeX environments and math:

```python
Guardrails.latex()
```

**Detects:**

- Unclosed `\begin{env}` environments
- Mismatched environment names
- Unbalanced `\[...\]` and `$$...$$`
- Unbalanced inline math `$...$`

### Zero Output Rule

Detects empty or meaningless output:

```python
Guardrails.zero_output()
```

**Detects:**

- Empty output
- Whitespace-only output
- Punctuation-only output
- Repeated character noise

### Pattern Rule

Detects known bad patterns:

```python
from l0 import Guardrails

Guardrails.pattern()  # All built-in patterns

# Custom patterns
Guardrails.custom_pattern(
    [r"forbidden", r"blocked"],
    message="Custom violation",
    severity="error",
)

# Include/exclude categories
Guardrails.pattern(
    include_categories=["META_COMMENTARY", "REFUSAL"],
    exclude_categories=["HEDGING"],
)
```

**Built-in patterns:**

| Category | Examples |
|----------|----------|
| META_COMMENTARY | "As an AI...", "I'm an AI assistant" |
| HEDGING | "Sure!", "Certainly!", "Of course!" |
| REFUSAL | "I cannot provide...", "I'm not able to..." |
| INSTRUCTION_LEAK | `[SYSTEM]`, `<\|im_start\|>` |
| PLACEHOLDERS | `[INSERT ...]`, `{{placeholder}}` |
| FORMAT_COLLAPSE | "Here is the...", "Let me..." |

### Drift Detection Rules

Detect model looping and stalls:

```python
from l0 import Guardrails

# Detect token stalls
Guardrails.stall(max_gap=5.0)  # Trigger after 5 seconds without tokens

# Detect repetitive output
Guardrails.repetition(
    window=100,              # Character window size
    threshold=0.5,           # Similarity threshold (0-1)
    sentence_check=True,     # Check for repeated sentences
    sentence_repeat_count=3, # Trigger on 3+ repeats
)
```

---

## Violation Severity

| Severity | Behavior |
|----------|----------|
| `fatal` | Halt immediately, no retry |
| `error` | Trigger retry if recoverable |
| `warning` | Log but continue |

```python
from l0.guardrails import GuardrailViolation

# GuardrailViolation fields
violation.rule          # Name of the rule that was violated
violation.message       # Human-readable message
violation.severity      # "fatal" | "error" | "warning"
violation.recoverable   # Whether this violation is recoverable via retry
violation.position      # Position in content where violation occurred (optional)
violation.timestamp     # Timestamp when violation was detected (optional)
violation.context       # Additional context about the violation (optional)
violation.suggestion    # Suggested fix or action (optional)
```

---

## Custom Rules

### Simple Rule

```python
from l0.guardrails import GuardrailRule, GuardrailViolation

def check_no_swearing(state):
    violations = []
    if "damn" in state.content.lower() or "hell" in state.content.lower():
        violations.append(GuardrailViolation(
            rule="no-swearing",
            message="Profanity detected",
            severity="error",
            recoverable=True,
        ))
    return violations

no_swearing = GuardrailRule(
    name="no-swearing",
    description="Blocks profanity",
    streaming=False,      # Only check on complete
    severity="error",
    recoverable=True,
    check=check_no_swearing,
)

result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=[*Guardrails.recommended(), no_swearing],
)
```

### Streaming Rule

```python
from l0.guardrails import GuardrailRule, GuardrailViolation

def check_length_limit(state):
    if len(state.content) > 10000:
        return [GuardrailViolation(
            rule="length-limit",
            message="Output exceeds 10,000 characters",
            severity="fatal",
            recoverable=False,
        )]
    return []

length_limit = GuardrailRule(
    name="length-limit",
    description="Limits output length",
    streaming=True,      # Check during streaming
    severity="fatal",
    recoverable=False,
    check=check_length_limit,
)
```

### GuardrailContext

When using the engine directly, you can pass a `GuardrailContext`:

```python
from l0.guardrails import GuardrailContext

context = GuardrailContext(
    content="...",           # Full accumulated content
    completed=False,         # Stream finished?
    checkpoint=None,         # Previous checkpoint content
    delta=None,              # Latest chunk (streaming)
    token_count=0,           # Tokens received
    metadata=None,           # Stream metadata
    previous_violations=None,  # Previous violations
)
```

---

## Guardrail Engine

For advanced use cases, use the engine directly:

```python
from l0 import Guardrails
from l0.guardrails import GuardrailContext, create_guardrail_engine

# Create engine
engine = Guardrails.create_engine(
    Guardrails.recommended(),
    stop_on_fatal=True,
    enable_streaming=True,
    check_interval=100,
    on_violation=lambda v: print(f"Violation: {v.message}"),
)

# Check content
result = engine.check(GuardrailContext(
    content="...",
    completed=True,
    token_count=100,
))

print(result.passed)        # True/False
print(result.violations)    # list[GuardrailViolation]
print(result.should_retry)  # True/False
print(result.should_halt)   # True/False
print(result.summary)       # GuardrailResultSummary

# Or one-shot check
result = Guardrails.check_full(context, rules)
```

### Engine Methods

```python
engine.check(context)                    # Run all rules
engine.add_rule(rule)                    # Add rule
engine.remove_rule("rule-name")          # Remove rule
engine.get_state()                       # Get current state
engine.reset()                           # Reset state
engine.has_violations()                  # Any violations?
engine.has_fatal_violations()            # Any fatal?
engine.has_error_violations()            # Any errors?
engine.get_violations_by_rule("json")    # Violations for rule
engine.get_all_violations()              # All violations
```

### GuardrailResult

```python
from l0.guardrails import GuardrailResult, GuardrailResultSummary

# GuardrailResult fields
result.passed         # True if no violations
result.violations     # list[GuardrailViolation]
result.should_retry   # True if recoverable error violations
result.should_halt    # True if fatal or non-recoverable error

# GuardrailResultSummary fields
result.summary.total     # Total violations
result.summary.fatal     # Fatal count
result.summary.errors    # Error count
result.summary.warnings  # Warning count
```

---

## Analysis Functions

Low-level analysis utilities:

```python
from l0 import Guardrails

# JSON analysis
json_result = Guardrails.analyze_json('{"a": 1')
print(json_result.is_balanced)      # False
print(json_result.open_braces)      # 1
print(json_result.close_braces)     # 0
print(json_result.open_brackets)    # 0
print(json_result.close_brackets)   # 0
print(json_result.in_string)        # False
print(json_result.unclosed_string)  # False
print(json_result.issues)           # ["Unbalanced braces..."]

# Markdown analysis
md_result = Guardrails.analyze_markdown("```js\ncode")
print(md_result.is_balanced)           # False
print(md_result.in_fence)              # True
print(md_result.open_fences)           # 1
print(md_result.close_fences)          # 0
print(md_result.fence_languages)       # ["js"]
print(md_result.table_rows)            # 0
print(md_result.inconsistent_columns)  # False

# LaTeX analysis
tex_result = Guardrails.analyze_latex("\\begin{equation}")
print(tex_result.is_balanced)             # False
print(tex_result.open_environments)       # ["equation"]
print(tex_result.display_math_balanced)   # True
print(tex_result.inline_math_balanced)    # True
print(tex_result.bracket_math_balanced)   # True

# Content detection
Guardrails.looks_like_json('{"key": 1}')     # True
Guardrails.looks_like_markdown("# Header")   # True
Guardrails.looks_like_latex("\\frac{1}{2}")  # True

# Empty/noise detection
Guardrails.is_zero_output("")         # True
Guardrails.is_zero_output("   ")      # True
Guardrails.is_noise_only("...")       # True
Guardrails.is_noise_only("aaaaaa")    # True

# Pattern detection
matches = Guardrails.find_patterns(content, Guardrails.BAD_PATTERNS.META_COMMENTARY)
for pattern, match in matches:
    print(f"Found: {match.group()} at {match.start()}")
```

---

## Pattern Detection Functions

Detect specific pattern categories:

```python
from l0 import Guardrails
from l0.guardrails import GuardrailContext

context = GuardrailContext(content="As an AI, I cannot...", completed=True)

# Detect meta commentary ("As an AI...", "I'm a language model...")
violations = Guardrails.detect_meta_commentary(context)

# Detect hedging at start ("Sure!", "Certainly!")
violations = Guardrails.detect_excessive_hedging(context)

# Detect refusal patterns ("I cannot provide...", "I'm unable to...")
violations = Guardrails.detect_refusal(context)

# Detect instruction leakage ([SYSTEM], <|im_start|>)
violations = Guardrails.detect_instruction_leakage(context)

# Detect placeholders ([INSERT ...], {{placeholder}})
violations = Guardrails.detect_placeholders(context)

# Detect format collapse ("Here is the...", "Let me...")
violations = Guardrails.detect_format_collapse(context)

# Detect repeated sentences (threshold = minimum repeats)
violations = Guardrails.detect_repetition(context, threshold=2)

# Detect duplicated first and last sentence (loop indicator)
violations = Guardrails.detect_first_last_duplicate(context)
```

---

## Performance: Fast and Slow Paths

L0 uses a two-path strategy to avoid blocking the streaming loop:

### Fast Path (Synchronous)

Runs immediately on each chunk for quick checks:

- **Delta-only checks**: Only examines the latest chunk
- **Small content**: Full check if total content < 5KB
- **Instant violations**: Blocked words, obvious patterns

### Slow Path (Asynchronous)

Deferred to avoid blocking:

- **Large content**: Full content scan for content > 5KB
- **Complex rules**: Pattern matching, structure analysis
- **Non-blocking**: Results delivered via callback

```python
from l0 import Guardrails
from l0.guardrails import GuardrailContext

engine = Guardrails.create_engine(Guardrails.recommended())

# Fast/slow path with immediate result if possible
def handle_result(result):
    if result.should_halt:
        print("Halting!")

result = Guardrails.run_async_check(engine, context, handle_result)

if result is not None:
    print("Fast path:", result.passed)
else:
    print("Deferred to async callback")

# Always async version
result = await Guardrails.run_check_async(engine, context)
```

### Rule Complexity

| Rule | Complexity | When Checked |
|------|------------|--------------|
| `zero_output` | O(1) | Fast path |
| `json` | O(n) | Scans full content |
| `markdown` | O(n) | Scans full content |
| `latex` | O(n) | Scans full content |
| `pattern` | O(n x p) | Scans full content x patterns |

For long outputs, increase check intervals to reduce frequency:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=Guardrails.recommended(),
    check_intervals={"guardrails": 50},  # Check every 50 tokens instead of default 5
)
```

---

## BAD_PATTERNS

Access pattern categories directly:

```python
from l0 import Guardrails

# Access pattern categories
Guardrails.BAD_PATTERNS.META_COMMENTARY   # ["\\bas an ai\\b", ...]
Guardrails.BAD_PATTERNS.HEDGING           # ["^sure[,!]?\\s", ...]
Guardrails.BAD_PATTERNS.REFUSAL           # ["\\bi cannot provide\\b", ...]
Guardrails.BAD_PATTERNS.INSTRUCTION_LEAK  # ["\\[SYSTEM\\]", ...]
Guardrails.BAD_PATTERNS.PLACEHOLDERS      # ["\\[INSERT[^\\]]*\\]", ...]
Guardrails.BAD_PATTERNS.FORMAT_COLLAPSE   # ["^here is the\\b", ...]

# Get all patterns
all_patterns = Guardrails.BAD_PATTERNS.all_patterns()
```

---

## Integration with Retry

Guardrail violations integrate with retry logic:

```python
result = await l0.run(
    stream=lambda: client.chat.completions.create(...),
    guardrails=Guardrails.recommended(),
    retry=l0.Retry(
        max_attempts=3,
        retry_on=["guardrail_violation"],  # Retry on recoverable violations
    ),
)
```

| Violation Type | Counts Toward Limit |
|----------------|---------------------|
| `recoverable: True` | Yes |
| `recoverable: False` | No |

Zero output violations are `recoverable: True` because retry may help recover from transport issues.

---

## Type Aliases

The `Guardrails` class provides convenient type aliases:

```python
from l0 import Guardrails

# Use type aliases
rule: Guardrails.Rule = ...              # GuardrailRule
violation: Guardrails.Violation = ...    # GuardrailViolation
context: Guardrails.Context = ...        # GuardrailContext
result: Guardrails.Result = ...          # GuardrailResult
summary: Guardrails.ResultSummary = ...  # GuardrailResultSummary
state: Guardrails.State = ...            # GuardrailState
config: Guardrails.Config = ...          # GuardrailConfig
engine: Guardrails.Engine = ...          # GuardrailEngine

# Analysis result types
json_analysis: Guardrails.JsonAnalysis = ...
md_analysis: Guardrails.MarkdownAnalysis = ...
tex_analysis: Guardrails.LatexAnalysis = ...
```

---

## API Reference

### Presets

| Method | Description |
|--------|-------------|
| `Guardrails.minimal()` | JSON + zero output |
| `Guardrails.recommended()` | JSON, Markdown, patterns, zero output |
| `Guardrails.strict()` | All rules including drift detection |
| `Guardrails.json_only()` | JSON validation only |
| `Guardrails.markdown_only()` | Markdown validation only |
| `Guardrails.latex_only()` | LaTeX validation only |
| `Guardrails.none()` | No guardrails |

### Rules

| Method | Description |
|--------|-------------|
| `Guardrails.json()` | Balanced JSON structure |
| `Guardrails.strict_json()` | Parseable JSON validation |
| `Guardrails.markdown()` | Markdown structure |
| `Guardrails.latex()` | LaTeX environments and math |
| `Guardrails.pattern(...)` | Built-in bad patterns |
| `Guardrails.custom_pattern(...)` | Custom pattern rule |
| `Guardrails.zero_output()` | Empty/noise detection |
| `Guardrails.stall(max_gap)` | Token stall detection |
| `Guardrails.repetition(...)` | Repetition/loop detection |

### Analysis

| Method | Description |
|--------|-------------|
| `Guardrails.analyze_json(content)` | Analyze JSON structure |
| `Guardrails.analyze_markdown(content)` | Analyze Markdown structure |
| `Guardrails.analyze_latex(content)` | Analyze LaTeX structure |
| `Guardrails.looks_like_json(content)` | Check if content is JSON-like |
| `Guardrails.looks_like_markdown(content)` | Check if content is Markdown-like |
| `Guardrails.looks_like_latex(content)` | Check if content is LaTeX-like |
| `Guardrails.is_zero_output(content)` | Check if content is empty |
| `Guardrails.is_noise_only(content)` | Check if content is noise |
| `Guardrails.find_patterns(content, patterns)` | Find pattern matches |

### Pattern Detection

| Method | Description |
|--------|-------------|
| `Guardrails.detect_meta_commentary(ctx)` | Detect AI self-reference |
| `Guardrails.detect_excessive_hedging(ctx)` | Detect hedging phrases |
| `Guardrails.detect_refusal(ctx)` | Detect refusal patterns |
| `Guardrails.detect_instruction_leakage(ctx)` | Detect prompt leakage |
| `Guardrails.detect_placeholders(ctx)` | Detect placeholders |
| `Guardrails.detect_format_collapse(ctx)` | Detect meta-instructions |
| `Guardrails.detect_repetition(ctx, threshold)` | Detect repeated sentences |
| `Guardrails.detect_first_last_duplicate(ctx)` | Detect loop indicator |

### Engine & Checking

| Method | Description |
|--------|-------------|
| `Guardrails.create_engine(rules, ...)` | Create guardrail engine |
| `Guardrails.check(state, rules)` | Run rules, return violations |
| `Guardrails.check_full(context, rules)` | Run rules, return full result |
| `Guardrails.run_async_check(engine, ctx, cb)` | Fast/slow path check |
| `Guardrails.run_check_async(engine, ctx)` | Async check |
