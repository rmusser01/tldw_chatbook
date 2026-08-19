"""Shared fixtures for the ported upstream chunking suite (Task 4).

Ported from tldw_Server_API/tests/Chunking/ @ 385afa95 with the same import
rewrite as Helper_Scripts/sync_chunking_engine.py. Engine-only tests run
against the vendored tree directly; server-fixture files (endpoints, AuthNZ,
templates, async_chunker, auto_planner, server Metrics) are skipped at module
level with documented reasons.
"""
import os
from pathlib import Path

import pytest


def requires_cached_hf_tokenizer():
    """Skip-mark helper for ported tests that load a real HF tokenizer.

    Several upstream tests exercise the tokens strategy with its default
    'gpt2' tokenizer, which transformers downloads from huggingface.co on
    first use. chatbook's repo-wide network guard (Tests/conftest.py) blocks
    that download, and the engine (correctly) surfaces the failure as
    TokenizerError. These tests are environment-dependent, not parity gaps:
    skip unless the tokenizer is already in a cache the test process can see.

    NOTE: Tests/conftest.py's `isolate_test_environment` redirects HOME to a
    temp dir, so the default ~/.cache/huggingface location is invisible under
    pytest. Resolve the real cache from the pre-redirect HOME (or explicit
    HF_HOME/HF_HUB_CACHE) so the skip decision matches what the test process
    will actually be able to load.

    Apply as: @pytest.mark.skipif(not requires_cached_hf_tokenizer(), reason=...)
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import HF_HUB_CACHE, default_cache_path

        probe_cache = HF_HUB_CACHE
        if Path(probe_cache).resolve() == Path(default_cache_path).resolve():
            # default location: re-resolve against the original HOME, not the
            # pytest-redirected one
            real_home = os.environ.get("TLDW_REAL_HOME") or _ORIG_HOME
            probe_cache = str(Path(real_home) / ".cache" / "huggingface" / "hub")
        p = try_to_load_from_cache("gpt2", "config.json", cache_dir=probe_cache)
        return p is not None and not str(p).endswith(".no_exist")
    except Exception:
        return False


def _detect_orig_home() -> str:
    """Capture HOME before pytest's conftest chain redirects it."""
    return os.environ.get("HOME") or str(Path.home())


_ORIG_HOME = _detect_orig_home()

# Upstream's Chunking/__init__.py exports DEFAULT_CHUNK_OPTIONS, ChunkMetadata
# and ChunkResult at the package root; chatbook's engine package init is
# authored (spec §5.1) and deliberately omits them (chatbook's import-time
# consumer surface lives in the Chunk_Lib compat shim). Inject them onto the
# engine package so ported tests importing from the package root behave as
# upstream (test-side compatibility only; the engine itself is untouched).
import tldw_chatbook.Chunking.engine as _engine_pkg  # noqa: E402

if not hasattr(_engine_pkg, "DEFAULT_CHUNK_OPTIONS"):
    from tldw_chatbook.Chunking.Chunk_Lib import DEFAULT_CHUNK_OPTIONS as _DCO  # noqa: E402

    _engine_pkg.DEFAULT_CHUNK_OPTIONS = _DCO
if not hasattr(_engine_pkg, "ChunkMetadata"):
    from tldw_chatbook.Chunking.engine.base import (  # noqa: E402
        ChunkMetadata as _CM,
        ChunkResult as _CR,
    )

    _engine_pkg.ChunkMetadata = _CM
    _engine_pkg.ChunkResult = _CR


@pytest.fixture(autouse=True)
def _production_sanitization(request, monkeypatch):
    """Disable the engine's test-mode relaxation for tests that opt in.

    Spec §10.2: sanitization relaxes under PYTEST_CURRENT_TEST/is_test_mode,
    so production-path evidence requires explicitly disabling test mode.
    Tests marked with @pytest.mark.production_path get test mode off.
    """
    if request.node.get_closest_marker("production_path"):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("TLDW_DISABLE_TEST_MODE", "1")
        monkeypatch.setattr(
            "tldw_chatbook.Chunking._shims.testing.is_test_mode", lambda: False
        )
