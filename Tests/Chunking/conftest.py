"""Shared fixtures for the ported upstream chunking suite (Task 4).

Ported from tldw_Server_API/tests/Chunking/ @ 385afa95 with the same import
rewrite as Helper_Scripts/sync_chunking_engine.py. Engine-only tests run
against the vendored tree directly; server-fixture files (endpoints, AuthNZ,
templates, async_chunker, auto_planner, server Metrics) are skipped at module
level with documented reasons.
"""
import os
import pwd
from pathlib import Path

import pytest


def _real_home() -> str:
    """The user's real home directory, immune to pytest's HOME sandbox.

    Tests/conftest.py redirects HOME (and XDG_*) into a per-test temp sandbox
    at conftest IMPORT time — before this module loads — so os.environ["HOME"]
    and Path.home() here both point at the sandbox. The robust source is the
    passwd entry for the current uid (pwd.getpwuid), which never lies. Fall
    back to the env only if pwd is unavailable.
    """
    try:
        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError, OSError):
        return os.environ.get("HOME") or str(Path.home())


def real_hf_hub_cache() -> str:
    """Path of the real HuggingFace hub cache (outside the pytest sandbox).

    Resolution order mirrors huggingface_hub's own precedence, but anchored at
    the REAL home: an explicit HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE / HF_HOME
    from the pre-sandbox environment (Tests/conftest.py snapshots the original
    HOME into _PREVIOUS_TEST_ENV, which we reuse for the override vars), else
    <real home>/.cache/huggingface/hub.
    """
    prev = getattr(_root_conftest(), "_PREVIOUS_TEST_ENV", {})
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        val = prev.get(var) or os.environ.get(var)
        if val:
            return str(Path(val).expanduser())
    hf_home = prev.get("HF_HOME") or os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home).expanduser() / "hub")
    return str(Path(_real_home()) / ".cache" / "huggingface" / "hub")


def _root_conftest():
    """The repo-root Tests/conftest.py module, if loaded (it always is under
    pytest; guarded for standalone imports of this file)."""
    import sys

    return sys.modules.get("Tests.conftest")


def tokenizer_cached(cache_dir: str) -> bool:
    """True if the gpt2 tokenizer files needed by the ported tests are
    present in the given HF hub cache (and not marked .no_exist)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        for fname in ("config.json", "tokenizer.json", "vocab.json"):
            p = try_to_load_from_cache("gpt2", fname, cache_dir=cache_dir)
            if p is None or str(p).endswith(".no_exist"):
                return False
        return True
    except Exception:
        return False


def requires_cached_hf_tokenizer() -> bool:
    """Skip-mark helper for ported tests that load the real gpt2 tokenizer.

    Several upstream tests exercise the tokens strategy with its default
    'gpt2' tokenizer. Chatbook's repo-wide network guard (Tests/conftest.py,
    task-15111) blocks the HF-hub download, and the engine surfaces that as
    TokenizerError — so those tests can only run where the tokenizer is
    already cached on this machine. Probes the REAL cache (outside the pytest
    HOME sandbox) via `real_hf_hub_cache()`.

    Apply as: @pytest.mark.skipif(not requires_cached_hf_tokenizer(), reason=...)
    """
    return tokenizer_cached(real_hf_hub_cache())


@pytest.fixture
def real_hf_cache(monkeypatch):
    """Make the REAL HuggingFace hub cache visible to this test, offline.

    Why this exists: Tests/conftest.py sandboxes HOME (and XDG_*) per test, so
    the default ~/.cache/huggingface location is invisible under pytest, and
    the network guard forbids re-downloading. Tests that need the real gpt2
    tokenizer use this fixture to point the HF stack at the real cache with
    offline mode forced — a pure read of local files, no network.

    Mechanics (both required):
    * huggingface_hub.constants.HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE —
      transformers' cached-file resolution reads `constants.HF_HUB_CACHE` at
      CALL time (transformers/utils/hub.py), so patching the module attribute
      works even after import.
    * HF_HUB_OFFLINE=True on huggingface_hub.constants — skips every hub
      HEAD call, so the network guard never sees an attempt and resolution
      is a pure cache lookup.
    Also pre-imports transformers under the corrected env so its own module
    init (which snapshots cache paths at import time in some versions) is
    consistent.

    Skips (with a true reason) if the real cache genuinely lacks gpt2.
    """
    cache = real_hf_hub_cache()
    if not tokenizer_cached(cache):
        pytest.skip(
            f"gpt2 tokenizer not cached at {cache} (network downloads are "
            "blocked by the repo network guard; pre-cache gpt2 to run this)"
        )
    import huggingface_hub.constants as hf_constants

    monkeypatch.setenv("HF_HOME", str(Path(cache).parent))
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", cache)
    monkeypatch.setattr(hf_constants, "HUGGINGFACE_HUB_CACHE", cache)
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    import transformers  # noqa: F401  (pre-import under corrected env)
    return cache


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
    """Force the engine's PRODUCTION sanitization path for opted-in tests.

    Spec §10.2: ``Chunker._sanitize_input`` relaxes itself (keeps null bytes
    and control characters) when it believes it runs under a test. The
    detection at engine/chunker.py:1373 is::

        _os.getenv("PYTEST_CURRENT_TEST", "") != "" or is_test_mode()

    Two independent problems defeat a naive "just delete the env var" fix:

    1. ``PYTEST_CURRENT_TEST`` is RE-SET by pytest's runner at the start of
       every phase (``pytest_runtest_call`` calls ``_update_current_test_var``),
       so a setup-phase ``monkeypatch.delenv`` is undone before the test body
       runs. The engine also reads the var INLINE (``_os.getenv``), not via
       the shim.
    2. ``is_test_mode`` is imported into the engine module's namespace at
       import time (engine/chunker.py:21), so patching the shim module's
       attribute does not affect the engine's bound reference. Separately,
       Tests/conftest.py's autouse ``isolate_test_environment`` sets
       ``TLDW_TEST_MODE=1`` for every test, which the shim's ``is_test_mode``
       reads.

    The fix patches BOTH detection inputs for the duration of the test:

    * ``os.getenv`` (the stdlib module object — the engine's function-local
      ``import os as _os`` binds the same module) is wrapped so reads of
      ``PYTEST_CURRENT_TEST`` return "". This covers the engine's inline
      check (and ``_sanitize_input``'s ``import os as _os`` re-import).
    * the ENGINE-BOUND ``is_test_mode`` (``engine.chunker.is_test_mode``) is
      patched to ``lambda: False`` — the engine's own name, not the shim's.
    * ``TLDW_TEST_MODE`` is deleted from the environment (the shim would
      otherwise see the root conftest's "1").

    The wrapper is scoped: every other env read passes through unchanged, so
    this stays a surgical "production path" switch rather than an env wipe.

    Regression-protected by test_production_path_marker.py, which asserts a
    null-byte input is SANITIZED (replaced) under this fixture and preserved
    without it — if this mechanism ever rots, that file fails loudly.
    """
    if request.node.get_closest_marker("production_path"):
        monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
        real_getenv = os.getenv

        def _getenv_production(key, default=None):
            if key == "PYTEST_CURRENT_TEST":
                return ""
            return real_getenv(key, default)

        monkeypatch.setattr(os, "getenv", _getenv_production)
        monkeypatch.setattr(
            "tldw_chatbook.Chunking.engine.chunker.is_test_mode", lambda: False
        )
