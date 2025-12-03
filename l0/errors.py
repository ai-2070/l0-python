from __future__ import annotations

import re

from .types import ErrorCategory

# Network error patterns (matches TS implementation)
NETWORK_PATTERNS = [
    r"connection.*reset",
    r"connection.*refused",
    r"connection.*timeout",
    r"timed?\s*out",
    r"dns.*failed",
    r"name.*resolution",
    r"socket.*error",
    r"ssl.*error",
    r"eof.*occurred",
    r"broken.*pipe",
    r"network.*unreachable",
    r"host.*unreachable",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in NETWORK_PATTERNS]


def is_network_error(error: Exception) -> bool:
    """Check if error matches network error patterns."""
    msg = str(error).lower()
    return any(p.search(msg) for p in _compiled_patterns)


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize error for retry decisions."""
    msg = str(error).lower()

    # Check network patterns
    if is_network_error(error):
        return ErrorCategory.NETWORK

    # Check HTTP status if available
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status:
        if status == 429:
            return ErrorCategory.TRANSIENT
        if status in (401, 403):
            return ErrorCategory.FATAL
        if 500 <= status < 600:
            return ErrorCategory.TRANSIENT

    # Check for rate limit in message
    if "rate" in msg and "limit" in msg:
        return ErrorCategory.TRANSIENT

    return ErrorCategory.MODEL


def is_retryable(error: Exception) -> bool:
    """Determine if error should trigger retry."""
    category = categorize_error(error)
    return category not in (ErrorCategory.FATAL, ErrorCategory.INTERNAL)
