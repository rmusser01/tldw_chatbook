"""Deps-absent surface of the lazy RAG_Search facade (TASK-21102 review round).

The eager ``RAG_Search/__init__`` had an all-or-nothing ImportError fallback
that defined exactly four stub names -- ``EmbeddingsService``,
``ChunkingService``, ``IndexingService`` and the ``RAGService`` alias -- whose
constructors raise ``ImportError``; every OTHER re-export (``RAGConfig``,
``create_rag_service``, ...) was simply undefined, so from-importing it
raised ``ImportError``. Feature-detection code depends on that split:
``Tests/RAG/test_rag_dependencies.py``'s ``check_rag_services`` treats a
successful ``from tldw_chatbook.RAG_Search import create_rag_service`` as
"simplified RAG available". The first lazy-facade cut stubbed all ten names,
silently flipping that probe to True on deps-absent installs; this file pins
the base surface.

Subprocess-isolated (pattern of ``Tests/Packaging/test_chunking_import_
closure.py``): the deps-absent simulation blocks the facade's backing
submodules with a meta-path finder, which must be installed before anything
imports them, and must not leak into other tests' interpreters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            imports can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_DEPS_ABSENT_SNIPPET = """
import importlib.abc
import sys

BLOCKED_PREFIXES = (
    "tldw_chatbook.RAG_Search.simplified",
    "tldw_chatbook.RAG_Search.chunking_service",
)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(BLOCKED_PREFIXES):
            raise ImportError(f"blocked by test: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())

# The four names the eager fallback defined: importable, stub semantics.
from tldw_chatbook.RAG_Search import (
    ChunkingService,
    EmbeddingsService,
    IndexingService,
    RAGService,
)

for stub in (ChunkingService, EmbeddingsService, IndexingService, RAGService):
    try:
        stub()
    except ImportError as exc:
        assert "RAG services not available" in str(exc), (stub, str(exc))
    else:
        raise AssertionError(f"{stub.__name__} stub constructor did not raise")

# Every other re-export: NOT defined on deps-absent installs -- the
# from-import must raise ImportError, exactly as when the eager fallback
# left the name undefined. check_rag_services() feature-detects on this.
for absent_name in (
    "RAGConfig",
    "SearchResult",
    "SearchResultWithCitations",
    "create_rag_service",
    "create_config_for_collection",
    "create_config_for_testing",
):
    try:
        exec(f"from tldw_chatbook.RAG_Search import {absent_name}")
    except ImportError:
        pass
    else:
        raise AssertionError(
            f"deps-absent from-import of {absent_name} succeeded; base raised "
            "ImportError and feature detection keys off that"
        )

print("DEPS_ABSENT_SURFACE_OK")
"""


def test_deps_absent_surface_matches_eager_fallback(tmp_path: Path) -> None:
    """Deps-absent: 4 stub names importable, the other 6 raise ImportError.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _DEPS_ABSENT_SNIPPET)
    assert result.returncode == 0, (
        f"deps-absent facade probe failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "DEPS_ABSENT_SURFACE_OK" in result.stdout


_HEALTHY_SURFACE_SNIPPET = """
from tldw_chatbook.RAG_Search import (
    ChunkingService,
    IndexingService,
    RAGConfig,
    RAGService,
    SearchResult,
    SearchResultWithCitations,
    create_config_for_collection,
    create_config_for_testing,
    create_rag_service,
)
from tldw_chatbook.RAG_Search.simplified import RAGService as direct_rag_service

assert RAGService is direct_rag_service
assert IndexingService is RAGService  # backward-compat alias
assert callable(create_rag_service)

from tldw_chatbook.RAG_Search.chunking_service import (
    ChunkingService as direct_chunking,
)

assert ChunkingService is direct_chunking

import tldw_chatbook.RAG_Search as rag_search

try:
    rag_search.definitely_not_an_export
except AttributeError:
    pass
else:
    raise AssertionError("unknown attribute did not raise AttributeError")

print("HEALTHY_SURFACE_OK")
"""


def test_healthy_surface_resolves_real_objects(tmp_path: Path) -> None:
    """With deps present, every re-export is the real object, lazily resolved.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _HEALTHY_SURFACE_SNIPPET)
    assert result.returncode == 0, (
        f"healthy facade probe failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "HEALTHY_SURFACE_OK" in result.stdout
