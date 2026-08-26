"""Import-order regression guard for the reranker <-> simplified cycle.

The second edge of the cycle `Tests/RAG/test_config_profiles_import_order.py`
guards, found by a review of PR2075 and reproduced here before anything was
changed:

    import tldw_chatbook.RAG_Search.reranker
      -> reranker imports `.simplified.vector_store`
      -> which executes `simplified/__init__`
      -> which eagerly imports `enhanced_rag_service_v2`
      -> which imported `create_reranker_from_config` from the STILL
         partially-initialized `reranker` module
      => ImportError: cannot import name 'create_reranker_from_config' from
         partially initialized module

Same shape as TASK-21160, opposite direction, and latent for the same reason:
the eager `RAG_Search/__init__` used to front-load `simplified` in the safe
order until TASK-21102 made that facade lazy. A test module that imported
`reranker` first therefore could not be collected on its own, and full-suite
import order masked it.

Fixed by deferring the import to first use in `enhanced_rag_service_v2`,
keeping `create_reranker_from_config` a module-level name because it is a
monkeypatch seam (see `Tests/RAG_Search/test_reranker_construction.py`).

Each test runs in a fresh subprocess so this file's own import order cannot
mask a regression.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _fresh_import(statement: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )


def test_reranker_first_import_succeeds():
    """The order that could not be collected: reranker before simplified."""
    result = _fresh_import(
        "from tldw_chatbook.RAG_Search import reranker as r; "
        "from tldw_chatbook.RAG_Search.reranker import PointwiseReranker; "
        "assert callable(r.create_reranker_from_config)"
    )
    assert result.returncode == 0, (
        "reranker-first import failed -- the reranker<->simplified cycle is "
        f"back:\n{result.stderr[-2000:]}"
    )


def test_simplified_first_import_still_succeeds():
    """The historically-safe order must keep working after the deferral."""
    result = _fresh_import(
        "import tldw_chatbook.RAG_Search.simplified as s; "
        "from tldw_chatbook.RAG_Search import reranker as r; "
        "assert s.EnhancedRAGServiceV2 is not None; "
        "assert callable(r.create_reranker_from_config)"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_reranker_is_not_on_the_eager_simplified_import_graph():
    """Behavioural edge census, self-maintaining where a static one is not.

    A static sweep for module-level `..reranker` imports under `simplified/`
    would flag `active_config.py`, which is NOT executed by
    `simplified/__init__` and so cannot close the cycle. Asserting on the
    loaded-module set instead catches any new eager back-edge from any
    module the package actually executes, and stays quiet about the lazy ones.
    """
    result = _fresh_import(
        "import sys; import tldw_chatbook.RAG_Search.simplified; "
        "print('reranker' if 'tldw_chatbook.RAG_Search.reranker' in sys.modules "
        "else 'clean')"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("clean"), (
        "importing `simplified` pulled in `RAG_Search.reranker` eagerly, which "
        "re-creates the cycle for a reranker-first import order"
    )


def test_v2_keeps_create_reranker_from_config_as_a_patchable_module_attribute():
    """The deferral must not break the construction-failure monkeypatch seam.

    `Tests/RAG_Search/test_reranker_construction.py` replaces
    `enhanced_rag_service_v2.create_reranker_from_config` eight times. A
    module-accessor deferral (the shape used for `config_profiles`) would have
    removed the name those tests patch, so the wrapper keeps it.
    """
    result = _fresh_import(
        "from tldw_chatbook.RAG_Search.simplified import enhanced_rag_service_v2 as v2; "
        "assert callable(v2.create_reranker_from_config); "
        "sentinel = object(); "
        "v2.create_reranker_from_config = lambda config: sentinel; "
        "assert v2.create_reranker_from_config(None) is sentinel; "
        "print('patchable')"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("patchable")


def test_deferred_wrapper_actually_builds_a_reranker():
    """The wrapper must delegate, not just exist.

    A wrapper that returned None would satisfy every assertion above while
    silently disabling reranking, so build a real one through it.
    """
    result = _fresh_import(
        "from tldw_chatbook.RAG_Search.simplified import enhanced_rag_service_v2 as v2; "
        "from tldw_chatbook.RAG_Search.reranker import RerankingConfig, PointwiseReranker; "
        "built = v2.create_reranker_from_config(RerankingConfig(strategy='pointwise')); "
        "assert isinstance(built, PointwiseReranker), type(built); "
        "print('built')"
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("built")
