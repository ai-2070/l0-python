# Formatting Helpers

L0 provides utilities for formatting context, memory, output instructions, tool definitions, and strings.

## Context Formatting

Wrap documents and instructions with proper delimiters:

```python
from l0.formatting import format_context, format_document, format_instructions

# XML delimiters (default)
format_context("User manual content", label="Documentation")
# <documentation>
# User manual content
# </documentation>

# Markdown delimiters
format_context("Content", label="Context", delimiter="markdown")
# # Context
#
# Content

# Bracket delimiters
format_context("Content", delimiter="brackets")
# [CONTEXT]
# ==============================
# Content
# ==============================

# No delimiters
format_context("Content", delimiter="none")

# Custom delimiters
format_context(
    "Content",
    custom_delimiter_start="<<<START>>>",
    custom_delimiter_end="<<<END>>>",
)
# <<<START>>>
# Content
# <<<END>>>

# Document with metadata
format_document("Report content", {"title": "Q4 Report", "author": "Team"})

# System instructions
format_instructions("You are a helpful assistant.")
```

### Options

```python
format_context(
    content,
    label="Context",              # Label for section (default: "Context")
    delimiter="xml",              # xml | markdown | brackets | none
    dedent=True,                  # Remove common indentation (default: True)
    normalize=True,               # Normalize whitespace (default: True)
    custom_delimiter_start="...", # Custom start delimiter
    custom_delimiter_end="...",   # Custom end delimiter
)
```

### Multiple Contexts

```python
from l0.formatting import format_multiple_contexts

format_multiple_contexts([
    {"content": "Document 1", "label": "Doc1"},
    {"content": "Document 2", "label": "Doc2"},
])
# <doc1>
# Document 1
# </doc1>
#
# <doc2>
# Document 2
# </doc2>
```

### Delimiter Escaping

Prevent injection attacks:

```python
from l0.formatting import escape_delimiters, unescape_delimiters

escape_delimiters("<script>alert('xss')</script>", "xml")
# &lt;script&gt;alert('xss')&lt;/script&gt;

unescape_delimiters("&lt;div&gt;", "xml")
# <div>

# Markdown escaping
escape_delimiters("# Header", "markdown")
# \# Header

# Bracket escaping
escape_delimiters("[section]", "brackets")
# \[section\]
```

### Types

```python
from l0.formatting import ContextOptions, ContextItem, DocumentMetadata, DelimiterType

# DelimiterType is a Literal type
delimiter: DelimiterType = "xml"  # "xml" | "markdown" | "brackets" | "none"

# ContextOptions dataclass
options = ContextOptions(
    label="Context",
    delimiter="xml",
    dedent=True,
    normalize=True,
    custom_delimiter_start=None,
    custom_delimiter_end=None,
)

# ContextItem for multiple contexts
item = ContextItem(content="Document content", label="doc")

# DocumentMetadata for documents
metadata = DocumentMetadata(
    title="Report",
    author="Team",
    date="2024-01-01",
    source="internal",
    extra={"version": "1.0"},
)
```

---

## Memory Formatting

Format conversation history for model context:

```python
from l0.formatting import format_memory, create_memory_entry

memory = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]

# Conversational style (default)
format_memory(memory)
# User: Hello
#
# Assistant: Hi there!

# Structured XML style
format_memory(memory, {"style": "structured"})
# <conversation_history>
#   <message role="user">Hello</message>
#   <message role="assistant">Hi there!</message>
# </conversation_history>

# Compact style
format_memory(memory, {"style": "compact"})
# U: Hello
# A: Hi there!
```

### Options

```python
from l0.formatting import MemoryFormatOptions

format_memory(memory, MemoryFormatOptions(
    max_entries=10,             # Limit entries (takes last N)
    include_timestamps=True,    # Add timestamps
    include_metadata=True,      # Add metadata
    style="conversational",     # conversational | structured | compact
))

# Or use a dict
format_memory(memory, {
    "max_entries": 10,
    "include_timestamps": True,
    "include_metadata": True,
    "style": "conversational",
})
```

### Memory Utilities

```python
from l0.formatting import (
    create_memory_entry,
    merge_memory,
    filter_memory_by_role,
    get_last_n_entries,
    calculate_memory_size,
    truncate_memory,
)

# Create timestamped entry
entry = create_memory_entry("user", "Hello", {"source": "chat"})
# MemoryEntry(role="user", content="Hello", timestamp=datetime(...), metadata={"source": "chat"})

# Merge and sort by timestamp
merged = merge_memory(memory1, memory2)

# Filter by role
user_messages = filter_memory_by_role(memory, "user")

# Get recent entries
recent = get_last_n_entries(memory, 5)

# Calculate size
size = calculate_memory_size(memory)  # character count

# Truncate to fit limit (keeps most recent)
truncated = truncate_memory(memory, 10000)  # max chars
```

### Memory Entry

```python
from l0.formatting import MemoryEntry, MemoryRole, MemoryStyle

# MemoryRole is a Literal type
role: MemoryRole = "user"  # "user" | "assistant" | "system"

# MemoryStyle is a Literal type
style: MemoryStyle = "conversational"  # "conversational" | "structured" | "compact"

# MemoryEntry dataclass
entry = MemoryEntry(
    role="user",
    content="Hello",
    timestamp=datetime.now(),  # optional
    metadata={"source": "chat"},  # optional
)
```

---

## Output Formatting

Generate instructions for requesting specific output formats:

```python
from l0.formatting import format_json_output, format_structured_output

# Strict JSON output
format_json_output({"strict": True})
# Respond with valid JSON only. Do not include any text before or after the JSON object.
# Do not wrap the JSON in markdown code blocks or backticks.
# Start your response with { and end with }.

# With schema
format_json_output({
    "strict": True,
    "schema": '{ "name": "string", "age": "number" }',
})

# Non-strict (allows surrounding text)
format_json_output({"strict": False})
# Respond with valid JSON.

# Control whether to include instructions
format_json_output({"include_instructions": False, "schema": "..."})

# Other formats
format_structured_output("yaml", {"strict": True})
format_structured_output("xml", {"strict": True})
format_structured_output("markdown")
format_structured_output("plain")
```

### Output Constraints

```python
from l0.formatting import (
    format_output_constraints,
    create_output_format_section,
    wrap_output_instruction,
)

format_output_constraints({
    "max_length": 500,
    "min_length": 100,
    "no_code_blocks": True,
    "no_markdown": True,
    "language": "Spanish",
    "tone": "professional",
})
# Keep your response under 500 characters.
# Provide at least 100 characters in your response.
# Do not use code blocks or backticks.
# Do not use Markdown formatting.
# Respond in Spanish.
# Use a professional tone.

# Complete format section
create_output_format_section("json", {
    "strict": True,
    "schema": '{ "result": "string" }',
    "constraints": {"max_length": 1000},
    "wrap": True,  # Wraps in <output_format> tags (default: True)
})

# Manual wrapping
wrap_output_instruction("Respond with JSON only")
# <output_format>
# Respond with JSON only
# </output_format>
```

### Extract & Clean

```python
from l0.formatting import extract_json_from_output, clean_output, validate_json_output

# Extract JSON from model output with extra text
extract_json_from_output('Here is the result: {"name": "John"}')
# {"name": "John"}

# Extract from code blocks
extract_json_from_output('```json\n{"name": "John"}\n```')
# {"name": "John"}

# Extract arrays
extract_json_from_output("The list: [1, 2, 3]")
# [1, 2, 3]

# Returns original if no JSON found
extract_json_from_output("No JSON here")
# No JSON here

# Clean common prefixes
clean_output("Sure, here is the result:\n```json\n{}\n```")
# {}

# Validate JSON
is_valid, error = validate_json_output('{"name": "John"}')
# (True, None)

is_valid, error = validate_json_output('{"name": }')
# (False, "Expecting value: line 1 column 10 (char 9)")
```

### Types

```python
from l0.formatting import (
    OutputFormat,
    JsonOutputOptions,
    StructuredOutputOptions,
    OutputConstraints,
    OutputFormatSectionOptions,
)

# OutputFormat is a Literal type
fmt: OutputFormat = "json"  # "json" | "yaml" | "xml" | "markdown" | "plain"

# JsonOutputOptions dataclass
json_opts = JsonOutputOptions(
    strict=False,
    schema=None,
    example=None,
)

# OutputConstraints dataclass
constraints = OutputConstraints(
    max_length=500,
    min_length=100,
    no_code_blocks=True,
    no_markdown=True,
    language="Spanish",
    tone="professional",
)
```

---

## Tool Formatting

Format tool/function definitions for LLM consumption:

```python
from l0.formatting import format_tool, create_tool, create_parameter

tool = create_tool("get_weather", "Get current weather", [
    create_parameter("location", "string", "City name", True),
    create_parameter("units", "string", "Temperature units", False),
])

# JSON Schema format (OpenAI function calling)
format_tool(tool, {"style": "json-schema"})
# {
#   "name": "get_weather",
#   "description": "Get current weather",
#   "parameters": {
#     "type": "object",
#     "properties": {...},
#     "required": ["location"]
#   }
# }

# TypeScript format
format_tool(tool, {"style": "typescript"})
# // Get current weather
# function get_weather(location: string, units?: string): void;

# Natural language
format_tool(tool, {"style": "natural"})
# Tool: get_weather
# Description: Get current weather
#
# Parameters:
#   - location (required): string - City name
#   - units (optional): string - Temperature units

# Natural language with examples
format_tool(tool, {"style": "natural", "include_examples": True})

# XML format
format_tool(tool, {"style": "xml"})
# <tool name="get_weather">
#   <description>Get current weather</description>
#   <parameters>
#     <parameter name="location" type="string" required="true" description="City name"/>
#     <parameter name="units" type="string" required="false" description="Temperature units"/>
#   </parameters>
# </tool>
```

### Tool Options

```python
from l0.formatting import ToolFormatOptions

format_tool(tool, ToolFormatOptions(
    style="json-schema",       # json-schema | typescript | natural | xml
    include_description=True,  # Include tool description
    include_types=True,        # Include type info in JSON schema
    include_examples=False,    # Add usage examples (natural style only)
))
```

### Multiple Tools

```python
from l0.formatting import format_tools, validate_tool

# Format array of tools
format_tools([tool1, tool2], {"style": "json-schema"})
# Returns list of dicts for json-schema, string for other styles

# Validate tool definition
errors = validate_tool(tool)
if errors:
    print("Invalid tool:", errors)
```

### Validation Rules

`validate_tool` checks:

- Tool name is required and must be a valid identifier (`[a-zA-Z_][a-zA-Z0-9_]*`)
- Tool description is required
- Parameters must be an array
- Each parameter needs a name (valid identifier) and type
- Valid types: `string`, `number`, `integer`, `boolean`, `array`, `object`

### Parse Function Calls

```python
from l0.formatting import parse_function_call, format_function_arguments

# Parse from model output - JSON format
result = parse_function_call('get_weather({"location": "NYC"})')
# FunctionCall(name="get_weather", arguments={"location": "NYC"})

print(result.name)       # get_weather
print(result.arguments)  # {"location": "NYC"}

# Parse keyword arguments
result = parse_function_call('get_weather(location="NYC", units="celsius")')
# FunctionCall(name="get_weather", arguments={"location": "NYC", "units": "celsius"})

# Returns None if no match
parse_function_call("No function call here")
# None

# Format arguments
format_function_arguments({"location": "NYC"}, pretty=True)
# {
#   "location": "NYC"
# }

format_function_arguments({"location": "NYC"}, pretty=False)
# {"location":"NYC"}
```

### Types

```python
from l0.formatting import (
    Tool,
    ToolParameter,
    ToolFormatStyle,
    ParameterType,
    FunctionCall,
)

# ToolFormatStyle is a Literal type
style: ToolFormatStyle = "json-schema"  # "json-schema" | "typescript" | "natural" | "xml"

# ParameterType is a Literal type
param_type: ParameterType = "string"  # "string" | "number" | "integer" | "boolean" | "array" | "object"

# ToolParameter dataclass
param = ToolParameter(
    name="location",
    type="string",
    description="City name",
    required=True,
    enum=["NYC", "LA", "Chicago"],  # optional
    default="NYC",                   # optional
    items={"type": "string"},        # for array types
    properties={},                   # for object types
)

# Tool dataclass
tool = Tool(
    name="get_weather",
    description="Get current weather",
    parameters=[param],
)

# FunctionCall dataclass
call = FunctionCall(
    name="get_weather",
    arguments={"location": "NYC"},
)
```

---

## String Utilities

Common string manipulation functions:

```python
from l0.formatting import (
    trim,
    escape,
    unescape,
    escape_html,
    unescape_html,
    escape_regex,
    sanitize,
    truncate,
    truncate_words,
    wrap,
    pad,
    remove_ansi,
)

# Trim whitespace
trim("  hello  ")  # "hello"

# Escape special characters
escape('Hello\n"World"')  # Hello\\n\\"World\\"
unescape("Hello\\nWorld")  # Hello\nWorld

# HTML entities
escape_html("<div>")  # &lt;div&gt;
unescape_html("&lt;div&gt;")  # <div>

# Regex escaping
escape_regex("file.txt")  # file\\.txt

# Sanitize (remove control chars except \n and \t)
sanitize("Hello\x00World")  # HelloWorld

# Truncate
truncate("Hello World", 8)  # "Hello..."
truncate("Hello World", 8, "...")  # "Hello..."

# Truncate at word boundary
truncate_words("Hello World Test", 12)  # "Hello..."

# Wrap to width
wrap("Hello World Test String", 10)
# Hello
# World Test
# String

# Pad string
pad("Hi", 10)  # "Hi        "
pad("Hi", 10, " ", "right")  # "        Hi"
pad("Hi", 10, " ", "center")  # "    Hi    "
pad("Hi", 10, "-", "left")  # "Hi--------"

# Remove ANSI codes
remove_ansi("\x1b[31mRed\x1b[0m")  # "Red"
```

### Alignment Type

```python
from l0.formatting import Alignment

align: Alignment = "left"  # "left" | "right" | "center"
```

---

## API Reference

### Context

| Function | Description |
|----------|-------------|
| `format_context(content, **options)` | Wrap content with delimiters |
| `format_multiple_contexts(items, delimiter)` | Format multiple contexts |
| `format_document(content, metadata, delimiter)` | Format document with metadata |
| `format_instructions(instructions, delimiter)` | Format system instructions |
| `escape_delimiters(content, delimiter)` | Escape delimiters for safety |
| `unescape_delimiters(content, delimiter)` | Unescape delimiters |

### Memory

| Function | Description |
|----------|-------------|
| `format_memory(memory, options)` | Format conversation history |
| `create_memory_entry(role, content, metadata)` | Create timestamped entry |
| `merge_memory(*memories)` | Merge and sort by timestamp |
| `filter_memory_by_role(memory, role)` | Filter by user/assistant/system |
| `get_last_n_entries(memory, n)` | Get last N entries |
| `calculate_memory_size(memory)` | Calculate character count |
| `truncate_memory(memory, max_size)` | Truncate to fit size limit |

### Output

| Function | Description |
|----------|-------------|
| `format_json_output(options)` | JSON output instructions |
| `format_structured_output(format, options)` | Format-specific instructions |
| `format_output_constraints(constraints)` | Length/tone/language constraints |
| `create_output_format_section(format, options)` | Complete format section |
| `wrap_output_instruction(instruction)` | Wrap in `<output_format>` tags |
| `extract_json_from_output(output)` | Extract JSON from text |
| `clean_output(output)` | Remove prefixes and wrappers |
| `validate_json_output(output)` | Validate JSON and return errors |

### Tools

| Function | Description |
|----------|-------------|
| `format_tool(tool, options)` | Format single tool definition |
| `format_tools(tools, options)` | Format multiple tools |
| `create_tool(name, description, params)` | Create tool definition |
| `create_parameter(name, type, desc, required)` | Create parameter |
| `validate_tool(tool)` | Validate tool structure |
| `parse_function_call(output)` | Parse function call from output |
| `format_function_arguments(args, pretty)` | Format arguments as JSON |

### Utilities

| Function | Description |
|----------|-------------|
| `trim(s)` | Trim whitespace |
| `escape(s)` / `unescape(s)` | Escape/unescape special chars |
| `escape_html(s)` / `unescape_html(s)` | HTML entity escaping |
| `escape_regex(s)` | Escape regex special chars |
| `sanitize(s)` | Remove control characters |
| `truncate(s, max, suffix)` | Truncate with suffix |
| `truncate_words(s, max, suffix)` | Truncate at word boundary |
| `wrap(s, width)` | Wrap to width |
| `pad(s, length, char, align)` | Pad left/right/center |
| `remove_ansi(s)` | Remove ANSI color codes |
