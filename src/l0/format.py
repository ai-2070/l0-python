# mypy: disable-error-code="valid-type"
"""Format namespace for L0 formatting utilities.

Provides a clean namespace for all formatting functions:

    import l0

    # Context formatting
    l0.Format.context("content", label="docs")
    l0.Format.document("content", {"title": "Report"})
    l0.Format.instructions("You are helpful")

    # Memory formatting
    l0.Format.memory(messages)
    l0.Format.memory_entry("user", "Hello")

    # Output formatting
    l0.Format.json_output(strict=True)
    l0.Format.structured_output("yaml")

    # Tool formatting
    l0.Format.tool(tool_def)
    l0.Format.tools([tool1, tool2])

    # String utilities
    l0.Format.escape_html("<div>")
    l0.Format.truncate("Hello World", 8)
"""

from __future__ import annotations

from typing import Any

from .formatting import (
    # Types
    Alignment,
    ContextItem,
    ContextOptions,
    DelimiterType,
    DocumentMetadata,
    FunctionCall,
    JsonOutputOptions,
    MemoryEntry,
    MemoryFormatOptions,
    MemoryRole,
    MemoryStyle,
    OutputConstraints,
    OutputFormat,
    OutputFormatSectionOptions,
    ParameterType,
    StructuredOutputOptions,
    Tool,
    ToolFormatOptions,
    ToolFormatStyle,
    ToolParameter,
    # Memory functions
    calculate_memory_size,
    # Output functions
    clean_output,
    create_memory_entry,
    create_output_format_section,
    # Tool functions
    create_parameter,
    create_tool,
    # String functions
    escape,
    # Context functions
    escape_delimiters,
    escape_html,
    escape_regex,
    extract_json_from_output,
    filter_memory_by_role,
    format_context,
    format_document,
    format_function_arguments,
    format_instructions,
    format_json_output,
    format_memory,
    format_multiple_contexts,
    format_output_constraints,
    format_structured_output,
    format_tool,
    format_tools,
    get_last_n_entries,
    merge_memory,
    pad,
    parse_function_call,
    remove_ansi,
    sanitize,
    trim,
    truncate,
    truncate_memory,
    truncate_words,
    unescape,
    unescape_delimiters,
    unescape_html,
    validate_json_output,
    validate_tool,
    wrap,
    wrap_output_instruction,
)


class Format:
    """Namespace for formatting utilities.

    Example:
        >>> import l0
        >>> l0.Format.context("Hello", label="greeting")
        '<greeting>\\nHello\\n</greeting>'

        >>> l0.Format.memory([{"role": "user", "content": "Hi"}])
        'User: Hi'

        >>> l0.Format.truncate("Hello World", 8)
        'Hello...'
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Types (for type hints and construction)
    # ─────────────────────────────────────────────────────────────────────────

    # Context types
    ContextOptions = ContextOptions
    ContextItem = ContextItem
    DocumentMetadata = DocumentMetadata
    DelimiterType = DelimiterType

    # Memory types
    MemoryEntry = MemoryEntry
    MemoryFormatOptions = MemoryFormatOptions
    MemoryRole = MemoryRole
    MemoryStyle = MemoryStyle

    # Output types
    JsonOutputOptions = JsonOutputOptions
    StructuredOutputOptions = StructuredOutputOptions
    OutputConstraints = OutputConstraints
    OutputFormatSectionOptions = OutputFormatSectionOptions
    OutputFormat = OutputFormat

    # Tool types
    Tool = Tool
    ToolParameter = ToolParameter
    ToolFormatOptions = ToolFormatOptions
    ToolFormatStyle = ToolFormatStyle
    ParameterType = ParameterType
    FunctionCall = FunctionCall

    # String types
    Alignment = Alignment

    # ─────────────────────────────────────────────────────────────────────────
    # Context Formatting
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def context(
        content: str,
        *,
        label: str = "context",
        delimiter: DelimiterType = "xml",
        dedent: bool = True,
        normalize: bool = True,
        custom_delimiter_start: str | None = None,
        custom_delimiter_end: str | None = None,
    ) -> str:
        """Wrap content with proper delimiters.

        Args:
            content: The content to wrap.
            label: The label for the context section.
            delimiter: The delimiter type - "xml", "markdown", "brackets", or "none".
            dedent: Whether to remove common leading whitespace.
            normalize: Whether to normalize whitespace (collapse multiple newlines).
            custom_delimiter_start: Custom start delimiter (overrides delimiter type).
            custom_delimiter_end: Custom end delimiter (overrides delimiter type).

        Returns:
            The formatted context string.

        Example:
            >>> Format.context("User manual", label="Documentation")
            '<documentation>\\nUser manual\\n</documentation>'

            >>> Format.context("Content", delimiter="markdown")
            '# Context\\n\\nContent'
        """
        return format_context(
            content,
            label=label,
            delimiter=delimiter,
            dedent=dedent,
            normalize=normalize,
            custom_delimiter_start=custom_delimiter_start,
            custom_delimiter_end=custom_delimiter_end,
        )

    @staticmethod
    def contexts(
        items: list[ContextItem] | list[dict[str, str]],
        *,
        delimiter: DelimiterType = "xml",
    ) -> str:
        """Format multiple contexts with the specified delimiter.

        Args:
            items: List of ContextItem objects or dicts with 'content' and 'label'.
            delimiter: The delimiter type for all contexts.

        Returns:
            The formatted contexts as a single string.
        """
        return format_multiple_contexts(items, delimiter=delimiter)

    @staticmethod
    def document(
        content: str,
        metadata: DocumentMetadata | dict[str, Any] | None = None,
        *,
        delimiter: DelimiterType = "xml",
    ) -> str:
        """Format a document with optional metadata.

        Args:
            content: The document content.
            metadata: Document metadata (title, author, date, source, etc.).
            delimiter: The delimiter type for formatting.

        Returns:
            The formatted document string.
        """
        return format_document(content, metadata, delimiter=delimiter)

    @staticmethod
    def instructions(
        instructions: str,
        *,
        delimiter: DelimiterType = "xml",
    ) -> str:
        """Format system instructions with proper delimiters.

        Args:
            instructions: The system instructions.
            delimiter: The delimiter type for formatting.

        Returns:
            The formatted instructions string.
        """
        return format_instructions(instructions, delimiter=delimiter)

    @staticmethod
    def escape_delimiters(content: str, delimiter: DelimiterType = "xml") -> str:
        """Escape delimiters in content to prevent injection attacks."""
        return escape_delimiters(content, delimiter)

    @staticmethod
    def unescape_delimiters(content: str, delimiter: DelimiterType = "xml") -> str:
        """Unescape delimiters in content."""
        return unescape_delimiters(content, delimiter)

    # ─────────────────────────────────────────────────────────────────────────
    # Memory Formatting
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def memory(
        memory: list[MemoryEntry] | list[dict[str, Any]],
        options: MemoryFormatOptions | dict[str, Any] | None = None,
    ) -> str:
        """Format conversation history for model context.

        Args:
            memory: List of MemoryEntry objects or dicts with 'role' and 'content'.
            options: Formatting options (max_entries, style, etc.).

        Returns:
            The formatted memory string.

        Example:
            >>> memory = [{"role": "user", "content": "Hello"}]
            >>> Format.memory(memory)
            'User: Hello'

            >>> Format.memory(memory, {"style": "compact"})
            'U: Hello'
        """
        return format_memory(memory, options)

    @staticmethod
    def memory_entry(
        role: MemoryRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create a timestamped memory entry.

        Args:
            role: The role - "user", "assistant", or "system".
            content: The message content.
            metadata: Optional metadata for the entry.

        Returns:
            A MemoryEntry with the current timestamp.
        """
        return create_memory_entry(role, content, metadata)

    @staticmethod
    def merge_memories(
        *memories: list[MemoryEntry] | list[dict[str, Any]],
    ) -> list[MemoryEntry]:
        """Merge multiple memory lists and sort by timestamp."""
        return merge_memory(*memories)

    @staticmethod
    def filter_memory(
        memory: list[MemoryEntry] | list[dict[str, Any]],
        role: MemoryRole,
    ) -> list[MemoryEntry]:
        """Filter memory entries by role."""
        return filter_memory_by_role(memory, role)

    @staticmethod
    def last_n_entries(
        memory: list[MemoryEntry] | list[dict[str, Any]],
        n: int,
    ) -> list[MemoryEntry]:
        """Get the last N entries from memory."""
        return get_last_n_entries(memory, n)

    @staticmethod
    def memory_size(memory: list[MemoryEntry] | list[dict[str, Any]]) -> int:
        """Calculate the total character count of memory."""
        return calculate_memory_size(memory)

    @staticmethod
    def truncate_memory(
        memory: list[MemoryEntry] | list[dict[str, Any]],
        max_size: int,
    ) -> list[MemoryEntry]:
        """Truncate memory to fit within a character limit."""
        return truncate_memory(memory, max_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Output Formatting
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def json_output(
        options: JsonOutputOptions | dict[str, Any] | None = None,
    ) -> str:
        """Generate instructions for requesting JSON output.

        Args:
            options: Options for JSON formatting (strict mode, schema, example).

        Returns:
            Instructions string for the model.

        Example:
            >>> Format.json_output({"strict": True})
            'Respond with valid JSON only...'
        """
        return format_json_output(options)

    @staticmethod
    def structured_output(
        format_type: OutputFormat,
        options: StructuredOutputOptions | dict[str, Any] | None = None,
    ) -> str:
        """Generate instructions for requesting specific output formats.

        Args:
            format_type: The output format - "json", "yaml", "xml", "markdown", "plain".
            options: Options for formatting (strict mode, schema, example).

        Returns:
            Instructions string for the model.
        """
        return format_structured_output(format_type, options)

    @staticmethod
    def output_constraints(
        constraints: OutputConstraints | dict[str, Any],
    ) -> str:
        """Generate instructions for output constraints.

        Args:
            constraints: The output constraints to apply.

        Returns:
            Instructions string describing the constraints.
        """
        return format_output_constraints(constraints)

    @staticmethod
    def output_section(
        format_type: OutputFormat,
        options: OutputFormatSectionOptions | dict[str, Any] | None = None,
    ) -> str:
        """Create a complete output format section.

        Args:
            format_type: The output format type.
            options: Section options including format options and constraints.

        Returns:
            A complete output format section string.
        """
        return create_output_format_section(format_type, options)

    @staticmethod
    def extract_json(output: str) -> str:
        """Extract JSON from model output that may contain extra text.

        Handles JSON in code blocks, with surrounding text, or raw JSON.
        """
        return extract_json_from_output(output)

    @staticmethod
    def clean_output(output: str) -> str:
        """Clean common prefixes and wrappers from model output."""
        return clean_output(output)

    @staticmethod
    def validate_json(output: str) -> tuple[bool, str | None]:
        """Validate that output is valid JSON.

        Returns:
            A tuple of (is_valid, error_message).
        """
        return validate_json_output(output)

    @staticmethod
    def wrap_output(instruction: str) -> str:
        """Wrap output instruction in clear delimiter.

        Args:
            instruction: The output format instruction to wrap.

        Returns:
            The instruction wrapped in <output_format> tags.

        Example:
            >>> Format.wrap_output("Respond with valid JSON only.")
            '<output_format>\\nRespond with valid JSON only.\\n</output_format>'
        """
        return wrap_output_instruction(instruction)

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Formatting
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def tool(
        tool: Tool,
        options: ToolFormatOptions | dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Format a single tool definition.

        Args:
            tool: The tool to format.
            options: Formatting options (style, include_description).

        Returns:
            The formatted tool (string or dict depending on style).
        """
        return format_tool(tool, options)

    @staticmethod
    def tools(
        tools: list[Tool],
        options: ToolFormatOptions | dict[str, Any] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Format multiple tool definitions.

        Args:
            tools: The tools to format.
            options: Formatting options.

        Returns:
            The formatted tools (list for json-schema, string for others).
        """
        return format_tools(tools, options)

    @staticmethod
    def create_tool(
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
    ) -> Tool:
        """Create a tool definition.

        Args:
            name: The tool name.
            description: Description of what the tool does.
            parameters: List of parameter definitions.

        Returns:
            A Tool object.
        """
        return create_tool(name, description, parameters)

    @staticmethod
    def parameter(
        name: str,
        param_type: ParameterType,
        description: str = "",
        required: bool = False,
        *,
        enum: list[str] | None = None,
        default: Any = None,
        items: dict[str, Any] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> ToolParameter:
        """Create a parameter definition.

        Args:
            name: The parameter name.
            param_type: The parameter type.
            description: Description of the parameter.
            required: Whether the parameter is required.
            enum: List of allowed values (for string types).
            default: Default value if not provided.
            items: Item schema for array types.
            properties: Property schema for object types.

        Returns:
            A ToolParameter object.
        """
        return create_parameter(
            name,
            param_type,
            description,
            required,
            enum=enum,
            default=default,
            items=items,
            properties=properties,
        )

    @staticmethod
    def validate_tool(tool: Tool) -> list[str]:
        """Validate a tool definition.

        Returns:
            A list of validation error messages. Empty if valid.
        """
        return validate_tool(tool)

    @staticmethod
    def parse_function_call(output: str) -> FunctionCall | None:
        """Parse a function call from model output.

        Returns:
            A FunctionCall object if found, None otherwise.
        """
        return parse_function_call(output)

    @staticmethod
    def function_arguments(
        arguments: dict[str, Any],
        pretty: bool = False,
    ) -> str:
        """Format function arguments as JSON."""
        return format_function_arguments(arguments, pretty)

    # ─────────────────────────────────────────────────────────────────────────
    # String Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def trim(s: str) -> str:
        """Trim whitespace from both ends of a string."""
        return trim(s)

    @staticmethod
    def escape(s: str) -> str:
        """Escape special characters (newlines, tabs, quotes)."""
        return escape(s)

    @staticmethod
    def unescape(s: str) -> str:
        """Unescape special characters."""
        return unescape(s)

    @staticmethod
    def escape_html(s: str) -> str:
        """Escape HTML entities in a string."""
        return escape_html(s)

    @staticmethod
    def unescape_html(s: str) -> str:
        """Unescape HTML entities in a string."""
        return unescape_html(s)

    @staticmethod
    def escape_regex(s: str) -> str:
        """Escape regex special characters in a string."""
        return escape_regex(s)

    @staticmethod
    def sanitize(s: str) -> str:
        """Remove control characters from a string."""
        return sanitize(s)

    @staticmethod
    def truncate(s: str, max_length: int, suffix: str = "...") -> str:
        """Truncate a string to a maximum length with a suffix."""
        return truncate(s, max_length, suffix)

    @staticmethod
    def truncate_words(s: str, max_length: int, suffix: str = "...") -> str:
        """Truncate a string at word boundaries."""
        return truncate_words(s, max_length, suffix)

    @staticmethod
    def wrap(s: str, width: int) -> str:
        """Wrap text to a specified width."""
        return wrap(s, width)

    @staticmethod
    def pad(
        s: str,
        length: int,
        char: str = " ",
        align: Alignment = "left",
    ) -> str:
        """Pad a string to a specified length."""
        return pad(s, length, char, align)

    @staticmethod
    def remove_ansi(s: str) -> str:
        """Remove ANSI escape codes from a string."""
        return remove_ansi(s)
