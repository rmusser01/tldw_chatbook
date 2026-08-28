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

# Budgets. Original pre-fix baseline measured ~4.8s-5.7s / 6,518-6,519
# modules; after the torch/transformers deferral it was ~4,659 modules; after
# the nltk (scipy/sklearn/pandas) deferral it was ~1.5s-2s / ~3,291 modules.
#
# TASK-21108 re-measured on 2026-08-23 (Python 3.12.11, macOS, isolated
# HOME/XDG, base install without the `subscriptions`/`all-tools` extras):
#
#   total sys.modules        1,700 before the diet -> 1,665 after
#   tldw_chatbook.* modules    664 before the diet ->   630 after
#   wall time                2.55s -> 2.44s with no repo .pyc,
#                            0.75s -> 0.74s warm, 5.6s worst observed on a
#                            fully cold interpreter+filesystem cache
#
# The old 4,000-module ceiling sat 2.4x above reality and could not see a
# 35-module regression, which is what the 21108 diet removed. It is tightened
# below, but on two axes rather than one, because the two axes are not
# equally deterministic:
#
# * MAX_TLDW_MODULE_COUNT is the drift signal. Only this repo's own modules
#   count toward it, so it moves when -- and essentially only when -- the
#   boot import graph changes. It is set just above the measured 630.
#
#   RATCHET (TASK-23029 / ADR-097,
#   backlog/decisions/097-boot-budget-ratchets.md): this constant never
#   rises. A breach means the cost defers or is shed elsewhere in the same
#   PR; the only other path is an explicit owner exception recorded in the
#   ADR's exception ledger. When a diet drops the measured number well below
#   the limit, LOWER the limit to measured + standard slack (ADR-097's
#   tightening convention) in that same PR. The pinned module-name snapshot
#   lives in boot_budget_snapshots/boot_import_modules.txt; refresh it only
#   via `scripts/update_boot_budget_snapshots.py`.
#
#   STANDING BREACH: dev b5eaa9cf64 measures 666 (2026-08-28) -- this guard
#   is red on dev until the 17 modules named in its failure message are
#   deferred (ADR-097 "Standing breach at adoption"; repayment tracked as
#   task-23112). The snapshot is deliberately pinned at c6218918d1's 657
#   in-budget set so the breach keeps naming the culprits.
#
#   Know what this does NOT catch. Measured by reverting each 21108 deferral
#   on its own and re-running this probe: panel only -> 649, notes-sync chain
#   only -> 645. Both PASS. Only the combined 34-module regression trips 660.
#   The per-deferral guard is
#   `Tests/Packaging/test_app_import_diet_closure.py`, which names each
#   module; this budget is the coarse net under it, not a substitute.
# * MAX_MODULE_COUNT stays a catastrophic-regression tripwire with real
#   slack, because the TOTAL closure varies with what is installed. The one
#   case reachable through a DECLARED extra: `Subscriptions/security.py:40`
#   attempts `cryptography` on the boot path, and `cryptography` ships in the
#   `subscriptions` / `all-tools` extras (tens of modules, with cffi). The
#   boot path also probes python-frontmatter, tokenizers and datasets
#   (`Prompts_Interop`, `Utils/custom_tokenizers`, `Evals/task_loader`), but
#   none of those three is declared in any extra or in core, so no supported
#   install pulls them -- and `datasets` would red HEAVY_MODULES via pandas
#   long before it moved this number. Pinning the total near 1,665 would fail
#   on an all-tools dev box for a reason unrelated to boot-path drift.
# * MAX_IMPORT_SECONDS deliberately stays at 8.0s: it is a hang tripwire, not
#   a perf assertion. A genuinely cold run (no .pyc anywhere, cold FS cache)
#   was measured at 5.6s on the machine above, so a "tightened" time bound
#   would buy noise-driven flakes, not signal.
#
# TASK-21731 re-measured on 2026-08-24, after this budget had caught its
# first real regression: `tldw_chatbook.*` had reached 703 (the whole
# Chunking engine + the RAG_Search.simplified tree + Internal_Prompts, all
# pulled by one module-scope import in
# `Library/library_local_rag_search_service.py`). Deferring it returned the
# count to 637 -- the drift signal worked exactly as designed, and the
# budget was NOT relaxed to accommodate the regression. The 637 is the 630
# above plus unrelated growth since, plus the one-module stdlib-only
# `RAG_Search/search_modes.py` that replaced the heavy import. Note what
# this axis still cannot see: the same modules were also being imported
# during the initial Chat screen mount, so removing them from the app
# import alone left time-to-interactive unchanged -- that leg is guarded by
# `Tests/Packaging/test_rag_boot_import_closure.py`.
MAX_IMPORT_SECONDS = 8.0
MAX_MODULE_COUNT = 2200
MAX_TLDW_MODULE_COUNT = 660


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
tldw_modules = sorted(
    m for m in sys.modules
    if m.startswith("tldw_chatbook") and sys.modules[m] is not None
)
print(json.dumps({{
    "elapsed": elapsed,
    "module_count": len(sys.modules),
    "tldw_module_count": len(tldw_modules),
    "tldw_modules": tldw_modules,
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
    """Catastrophic-regression tripwire on wall time and total module count.

    Not a tight perf assertion (machines and installed extras vary) -- just a
    guard against accidentally reintroducing the whole torch/transformers/nltk
    stack at boot. Original pre-fix baseline: ~4.8s-5.7s / 6,518-6,519
    modules; measured 2026-08-23: 1,665 modules. See the budget block at the
    top of this module for why this axis keeps slack and
    ``test_app_import_own_module_count_stays_at_the_post_diet_size`` is the
    drift signal.

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


def test_app_import_own_module_count_stays_at_the_post_diet_size(
    tmp_path: Path, ratchet
) -> None:
    """This repo's own boot import graph must stay at its post-diet size.

    The tight axis (TASK-21108). ``tldw_chatbook.*`` module residency after
    ``import tldw_chatbook.app`` depends only on this repo's import graph, not
    on which optional third-party extras happen to be installed, so it is the
    axis that can sit just above reality: 630 measured, 660 allowed.

    It catches a regression the SIZE of the whole 21108 diet (34 modules), not
    any single piece of it: reverting the panel deferral alone measures 649 and
    the notes-sync chain alone 645, both of which PASS here. Per-deferral
    coverage is ``Tests/Packaging/test_app_import_diet_closure.py``.

    ``MAX_TLDW_MODULE_COUNT`` is a RATCHET (TASK-23029 / ADR-097): it never
    rises. A breach diffs the live module set against the pinned snapshot
    (``boot_budget_snapshots/boot_import_modules.txt``) so the failure names
    the modules that consumed the headroom; on a pass the guard emits one
    ``boot-import-weight: used/limit`` headroom line.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
        ratchet: shared ratchet helper (see ``conftest.py``).
    """
    payload = _measure_app_import(tmp_path)
    count = payload["tldw_module_count"]
    modules = payload["tldw_modules"]
    assert count < MAX_TLDW_MODULE_COUNT, (
        f"import tldw_chatbook.app loaded {count} tldw_chatbook modules "
        f"(ratchet limit {MAX_TLDW_MODULE_COUNT}). Something new is eager on "
        "the boot path.\n"
        f"{ratchet.format_module_diff(modules, 'boot-import-weight')}\n"
        f"{ratchet.ratchet_policy('MAX_TLDW_MODULE_COUNT')}\n"
        f"Deliberate snapshot refresh: `{ratchet.SNAPSHOT_REFRESH}`"
    )
    ratchet.emit_headroom(
        ratchet.headroom_line(
            "boot-import-weight", [("modules", count, MAX_TLDW_MODULE_COUNT)]
        )
        + ratchet.snapshot_drift_suffix(modules, "boot-import-weight")
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
    if (
        importlib.util.find_spec("transformers") is None
        or importlib.util.find_spec("torch") is None
    ):
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
