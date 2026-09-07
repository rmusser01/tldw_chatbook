"""Regression guard for the production_path marker (Task 4 review I-1).

History: the first implementation of the marker deleted only
PYTEST_CURRENT_TEST and patched the SHIM module's is_test_mode. Both were
no-ops for the engine:

* pytest's runner re-sets PYTEST_CURRENT_TEST at the start of each phase
  (runner.py: pytest_runtest_call -> _update_current_test_var), undoing any
  setup-phase delenv before the test body runs; and the engine reads the var
  inline (engine/chunker.py:1374), not through the shim.
* engine/chunker.py:21 imports is_test_mode DIRECTLY into its namespace at
  import time, so patching tldw_chatbook.Chunking._shims.testing.is_test_mode
  left the engine's bound reference untouched — the engine kept returning
  True from is_test_mode(), and sanitization stayed relaxed.
* the root Tests/conftest.py autouse isolate_test_environment fixture sets
  TLDW_TEST_MODE=1 for EVERY test, and the shim's is_test_mode reads it.

The fixed mechanism (Tests/Chunking/conftest.py::_production_sanitization)
patches os.getenv so PYTEST_CURRENT_TEST reads empty AND the engine-BOUND
is_test_mode, and deletes TLDW_TEST_MODE. These tests pin that behavior to
the engine's actual sanitization outcome — null bytes REPLACED under the
marker (production) and PRESERVED without it (test mode) — so the marker can
never silently rot again. If the engine's detection logic ever moves, these
fail loudly and the fixture must be re-derived, not deleted.
"""
import os

import pytest

from tldw_chatbook.Chunking.engine import Chunker, ChunkerConfig


def _chunk(text: str) -> str:
    ck = Chunker(ChunkerConfig())
    return " ".join(ck.chunk_text(text, method="words", max_size=10, overlap=0))


@pytest.mark.production_path
def test_marker_null_bytes_are_sanitized():
    """Under @pytest.mark.production_path the engine must take the production
    sanitization branch: null bytes replaced with spaces."""
    out = _chunk("a\x00b c\x00d")
    assert "\x00" not in out, (
        "production_path marker is NOT disabling engine test mode — null bytes "
        "survived sanitization. The _production_sanitization fixture in "
        "Tests/Chunking/conftest.py has rotted; see this file's docstring."
    )
    assert "a b" in out.replace("  ", " ")


@pytest.mark.production_path
def test_marker_engine_bound_is_test_mode_is_false():
    """The engine's OWN bound is_test_mode (the name engine/chunker.py
    imported at module scope) must report False under the marker."""
    import tldw_chatbook.Chunking.engine.chunker as chunker_module

    assert chunker_module.is_test_mode() is False


@pytest.mark.production_path
def test_marker_pytest_current_test_hidden_from_engine():
    """The engine's inline `os.getenv("PYTEST_CURRENT_TEST")` check must see
    an empty value, even though pytest sets the real env var at call-phase
    start."""
    assert os.getenv("PYTEST_CURRENT_TEST", "") == ""


@pytest.mark.production_path
def test_marker_tldw_test_mode_removed():
    """The root conftest sets TLDW_TEST_MODE=1 for every test; the marker's
    fixture must remove it so the shim cannot re-enable test mode."""
    assert os.environ.get("TLDW_TEST_MODE", "") == ""


def test_without_marker_test_mode_relaxation_active():
    """Control: withOUT the marker the engine stays in test mode (relaxed
    sanitization, null bytes preserved). This is the upstream-intended
    behavior for ordinary ported tests and must not regress either."""
    out = _chunk("a\x00b")
    assert "\x00" in out, (
        "Expected relaxed (test-mode) sanitization to PRESERVE null bytes "
        "without the production_path marker — the default path changed."
    )


def test_marker_scoped_env_wrapper_leaves_other_vars_alone():
    """The production fixture's os.getenv wrapper is surgical: it only hides
    PYTEST_CURRENT_TEST. Other env reads pass through (proves the wrapper
    doesn't nuke the environment)."""
    os.environ["TLDW_PROBE_VAR"] = "probe-value"
    try:
        assert os.getenv("TLDW_PROBE_VAR") == "probe-value"
        assert os.getenv("MISSING_VAR_XYZ", "dflt") == "dflt"
    finally:
        os.environ.pop("TLDW_PROBE_VAR", None)
