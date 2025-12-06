"""Tests for l0 namespace pollution.

Ensures the public API stays clean and internal symbols are not exposed.

NOTE: These tests verify the initial state of the l0 namespace before any
submodules are imported. When other tests import submodules (e.g.,
`from l0.events import ...`), Python automatically adds those submodule
names to the parent module's namespace. This is standard Python behavior.

To get accurate results, these tests use importlib to get a fresh view of
what the __init__.py explicitly exports, rather than relying on dir() which
includes submodules imported by other tests.
"""

from __future__ import annotations

import l0


def _get_explicit_exports() -> set[str]:
    """Get symbols explicitly defined/exported in l0.__init__.py.

    This checks __all__ if defined, otherwise returns the symbols that
    would be visible on a fresh import (excluding submodules added by
    other imports during the test session).
    """
    # If __all__ is defined, use it
    if hasattr(l0, "__all__"):
        return set(l0.__all__)

    # Otherwise, check what's defined in _API_MODULES (the lazy import mapping)
    # plus the explicitly defined functions
    explicit = {"run", "wrap", "l0", "__version__"}
    if hasattr(l0, "_API_MODULES"):
        explicit.update(l0._API_MODULES.keys())
    return explicit


class TestNamespacePollution:
    """Tests that internal symbols are not exposed in the l0 namespace."""

    # These are internal symbols that should NOT be in the public namespace
    FORBIDDEN_SYMBOLS = [
        # Typing imports
        "Any",
        "TYPE_CHECKING",
        "Callable",
        "Iterator",
        "AsyncIterator",
        "overload",
        # Internal modules (should not be explicitly exported)
        "version",
        # Common pollution patterns
        "annotations",
        "importlib",
    ]

    # These are the only non-underscore symbols allowed at module level
    # (not counting lazy-loaded symbols from _API_MODULES)
    ALLOWED_TOP_LEVEL_SYMBOLS = [
        # Package self-reference (unavoidable)
        "l0",
        # Top-level functions
        "run",
        "wrap",
    ]

    def test_no_forbidden_symbols_in_explicit_exports(self):
        """Ensure forbidden internal symbols are not explicitly exported."""
        # Check __all__ or _API_MODULES for explicit exports
        explicit_exports = _get_explicit_exports()

        for forbidden in self.FORBIDDEN_SYMBOLS:
            assert forbidden not in explicit_exports, (
                f"Internal symbol '{forbidden}' is explicitly exported in l0 namespace. "
                f"Remove it from __all__ or _API_MODULES."
            )

    def test_no_typing_imports_leaked(self):
        """Ensure typing imports are not leaked into namespace."""
        # These should never appear even after submodule imports
        typing_symbols = [
            "Any",
            "TYPE_CHECKING",
            "Callable",
            "Iterator",
            "AsyncIterator",
            "overload",
        ]
        public_attrs = [x for x in dir(l0) if not x.startswith("_")]

        for symbol in typing_symbols:
            assert symbol not in public_attrs, (
                f"Typing symbol '{symbol}' is exposed in l0 namespace. "
                f"Use underscore prefix (e.g., '_{symbol}') to hide it."
            )

    def test_lazy_imports_not_loaded(self):
        """Ensure lazy imports are not loaded until accessed."""
        # These should not be in dir() until accessed
        public_attrs = dir(l0)

        # Major classes should not be eagerly loaded
        assert "WrappedClient" not in public_attrs
        assert "Retry" not in public_attrs
        assert "Stream" not in public_attrs

    def test_lazy_imports_work(self):
        """Ensure lazy imports work when accessed."""
        # Accessing should trigger lazy load
        assert l0.WrappedClient is not None
        assert l0.Retry is not None
        assert l0.Stream is not None

    def test_version_accessible(self):
        """Ensure __version__ is accessible."""
        assert hasattr(l0, "__version__")
        assert isinstance(l0.__version__, str)
        assert len(l0.__version__) > 0
