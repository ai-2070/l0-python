"""Tests for l0 namespace pollution.

Ensures the public API stays clean and internal symbols are not exposed.
"""

from __future__ import annotations

import l0


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
        # Internal modules
        "version",
        "api",
        # Common pollution patterns
        "annotations",
        "importlib",
    ]

    # These are the only non-underscore symbols allowed
    ALLOWED_SYMBOLS = [
        # Package self-reference (unavoidable)
        "l0",
        # Top-level functions
        "run",
        "wrap",
    ]

    def test_no_forbidden_symbols(self):
        """Ensure forbidden internal symbols are not exposed."""
        public_attrs = [x for x in dir(l0) if not x.startswith("_")]

        for forbidden in self.FORBIDDEN_SYMBOLS:
            assert forbidden not in public_attrs, (
                f"Internal symbol '{forbidden}' is exposed in l0 namespace. "
                f"Use underscore prefix (e.g., '_{forbidden}') to hide it."
            )

    def test_only_allowed_non_underscore_symbols(self):
        """Ensure only expected symbols are in namespace without underscore."""
        public_attrs = set(x for x in dir(l0) if not x.startswith("_"))
        allowed = set(self.ALLOWED_SYMBOLS)

        unexpected = public_attrs - allowed
        assert not unexpected, (
            f"Unexpected symbols in l0 namespace: {unexpected}. "
            f"Either add to ALLOWED_SYMBOLS or prefix with underscore."
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
