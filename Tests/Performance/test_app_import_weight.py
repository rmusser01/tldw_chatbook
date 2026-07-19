"""Regression guard for `tldw_chatbook.app`'s import-time dependency weight
(backlog task 163: "slim eager heavy imports at app startup").

Before this task, `import tldw_chatbook.app` eagerly pulled in torch and
transformers (`Embeddings/Embeddings_Lib.py` did module-scope
`get_safe_import('torch')` / `get_safe_import('transformers')` /
`get_safe_import('numpy')`, reachable via
`RAG_Admin/local_rag_admin_service.py` -> `Embeddings/Chroma_Lib.py` ->
`Embeddings/Embeddings_Lib.py`) plus several redundant module-scope
`from ...LLM_Calls.Summarization_General_Lib import analyze` /
`from ...Chunking.Chunk_Lib import chunk_for_embedding` imports (in
`Web_Scraping/Article_Extractor_Lib.py`, `Web_Scraping/WebSearch_APIs.py`,
and `Embeddings/Chroma_Lib.py`) and `Article_Extractor_Lib.py`'s own
`import pandas` fallback block. Combined, these pulled in roughly 1,800
extra modules and multiple seconds of import time at every app start, even
though none of it is needed until a user actually creates an embedding or
requests summarization.

torch/transformers are now imported lazily (see
`Embeddings/Embeddings_Lib.py`'s `_ensure_torch()` / `_ensure_transformers()`
/ `_ensure_numpy()` helpers) and the `analyze`/`chunk_for_embedding`/pandas
imports above were moved into the functions that actually use them.
(`Embeddings/Chroma_Lib.py` itself was later removed entirely by task-248 —
the RAG_Search vector store is the sole Chroma stack now — so the historical
import chains above no longer exist at all.)

nltk/scipy/sklearn/pandas were a second chain, pulled in via
`app.py` -> `RAG_Admin/local_rag_admin_service.py:11
from ..Chunking.chunking_interop_library import get_chunking_service` ->
`Chunking/__init__.py from .Chunk_Lib import (...)` (package `__init__` runs
for *any* import under `tldw_chatbook.Chunking`) -> `Chunking/Chunk_Lib.py`'s
module-scope `import nltk` (nltk transitively imports scipy from
`nltk/metrics/association.py`, sklearn from `nltk/classify/scikitlearn.py`,
and pandas). `Chunk_Lib.py`'s `import nltk` was deferred behind an
`_ensure_nltk()` helper + a `find_spec`-based `NLTK_AVAILABLE` probe, and the
module-scope `ensure_nltk_data()` call (which did a punkt *network download*
at import time) was removed and made lazy/idempotent -- so nltk, and with it
scipy/sklearn/pandas, no longer load at boot. This is the full task 163 guard
set (`HEAVY_MODULES` below), now asserted as a hard requirement by
`test_app_import_does_not_load_full_heavy_dependency_set`.

numpy is intentionally NOT in the guard set: it is pulled by chromadb (and
pymupdf), is comparatively light, and is a legitimate boot-time dependency.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# The full heavy-dependency guard set from the task 163 plan. NONE of these
# may be resident after a plain `import tldw_chatbook.app` (numpy is
# deliberately excluded -- see module docstring).
HEAVY_MODULES = (
    "torch",
    "transformers",
    "nltk",
    "scipy",
    "sklearn",
    "pandas",
    "docling",
    "torchvision",
)

# A focused subset kept as its own named test so a torch/transformers
# regression produces an obviously-scoped failure message.
ELIMINATED_MODULES = ("torch", "transformers")

# Generous catastrophic-regression bounds. Original pre-fix baseline measured
# ~4.8s-5.7s / 6,518-6,519 modules; after the torch/transformers deferral it
# was ~4,659 modules; after the nltk (scipy/sklearn/pandas) deferral it is
# ~1.5s-2s / ~3,291 modules (wall time is noisy -- cold-cache subprocess boot
# on a fresh isolated HOME can vary a lot -- so the module count, which is
# deterministic module-for-module, is the primary signal here; time is just a
# loose sanity check for a hang/runaway import, not a tight perf assertion).
MAX_IMPORT_SECONDS = 8.0
MAX_MODULE_COUNT = 4000


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    A fresh interpreter is required because `sys.modules` is process-global --
    once torch/nltk are imported by anything else (e.g. an earlier test in
    the same pytest session), they would stay cached in-process and this
    guard would give a false pass.
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

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_MEASURE_SNIPPET = """
import json
import sys
import time

t0 = time.perf_counter()
import tldw_chatbook.app
elapsed = time.perf_counter() - t0

loaded = sorted(m for m in {heavy_modules!r} if m in sys.modules)
print(json.dumps({{
    "elapsed": elapsed,
    "module_count": len(sys.modules),
    "loaded_heavy": loaded,
}}))
""".format(heavy_modules=HEAVY_MODULES)


def _measure_app_import(tmp_path: Path) -> dict:
    result = _run_isolated_python(tmp_path, _MEASURE_SNIPPET)
    assert result.returncode == 0, (
        f"import tldw_chatbook.app failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # The snippet's JSON is the last line of stdout (app boot logs to stderr
    # and, in some configurations, to stdout before the print()).
    last_line = result.stdout.strip().splitlines()[-1]
    return json.loads(last_line)


def test_app_import_does_not_load_torch_or_transformers(tmp_path: Path) -> None:
    """Plain `import tldw_chatbook.app` must never pull in torch/transformers.

    This is the core fix for task 163: EmbeddingFactory's torch/transformers
    resolution is now lazy (Embeddings_Lib._ensure_torch()/_ensure_transformers()),
    so neither module should be loaded until an embedding is actually built.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    payload = _measure_app_import(tmp_path)
    loaded_eliminated = [m for m in payload["loaded_heavy"] if m in ELIMINATED_MODULES]
    assert loaded_eliminated == [], (
        f"import tldw_chatbook.app eagerly loaded {loaded_eliminated}; "
        f"full heavy set loaded: {payload['loaded_heavy']}"
    )


def test_app_import_stays_well_under_pre_fix_baseline(tmp_path: Path) -> None:
    """Catastrophic-regression tripwire on wall time and module count.

    Not a tight perf assertion (machines vary) -- just a guard against
    accidentally reintroducing the whole torch/transformers/nltk stack at
    boot. Original pre-fix baseline: ~4.8s-5.7s / 6,518-6,519 modules;
    post-fix: ~1.5s-2s / ~3,291 modules.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    payload = _measure_app_import(tmp_path)
    assert payload["elapsed"] < MAX_IMPORT_SECONDS, (
        f"import tldw_chatbook.app took {payload['elapsed']:.2f}s "
        f"(limit {MAX_IMPORT_SECONDS}s); heavy modules loaded: {payload['loaded_heavy']}"
    )
    assert payload["module_count"] < MAX_MODULE_COUNT, (
        f"import tldw_chatbook.app loaded {payload['module_count']} modules "
        f"(limit {MAX_MODULE_COUNT}); heavy modules loaded: {payload['loaded_heavy']}"
    )


def test_app_import_does_not_load_full_heavy_dependency_set(tmp_path: Path) -> None:
    """The full task-163-plan guard: no heavy module at all should load at boot.

    Covers torch/transformers (deferred in Embeddings_Lib) AND
    nltk/scipy/sklearn/pandas (deferred by making Chunk_Lib's `import nltk`
    lazy via `_ensure_nltk()` -- nltk transitively pulls scipy/sklearn/pandas,
    so deferring nltk removes all four). numpy is intentionally excluded from
    HEAVY_MODULES (legit chromadb/pymupdf dependency).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    payload = _measure_app_import(tmp_path)
    assert payload["loaded_heavy"] == [], (
        f"import tldw_chatbook.app eagerly loaded heavy modules: "
        f"{payload['loaded_heavy']}"
    )


def test_ensure_torch_resolves_real_torch_when_installed() -> None:
    """The lazy accessor must still return the real module once called.

    Guards against the deferral breaking availability: a genuine
    embedding/torch use must still resolve torch, just later than at import
    time.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed in this environment")

    from tldw_chatbook.Embeddings import Embeddings_Lib

    assert Embeddings_Lib.torch is None or Embeddings_Lib.torch.__name__ == "torch"
    resolved = Embeddings_Lib._ensure_torch()
    assert resolved is not None
    assert resolved.__name__ == "torch"
    assert Embeddings_Lib.torch is resolved


def test_ensure_transformers_resolves_real_transformers_when_installed() -> None:
    """`_ensure_transformers()` must resolve transformers (and AutoModel/
    AutoTokenizer) once called, and no-op afterwards."""
    if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
        pytest.skip("torch/transformers not installed in this environment")

    from tldw_chatbook.Embeddings import Embeddings_Lib

    resolved = Embeddings_Lib._ensure_transformers()
    assert resolved is not None
    assert resolved.__name__ == "transformers"
    assert Embeddings_Lib.AutoModel is resolved.AutoModel
    assert Embeddings_Lib.AutoTokenizer is resolved.AutoTokenizer


def test_ensure_numpy_resolves_real_numpy_when_installed() -> None:
    """`_ensure_numpy()` must resolve numpy once called, and no-op afterwards."""
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy not installed in this environment")

    from tldw_chatbook.Embeddings import Embeddings_Lib

    resolved = Embeddings_Lib._ensure_numpy()
    assert resolved is not None
    assert resolved.__name__ == "numpy"
    assert Embeddings_Lib.np is resolved
