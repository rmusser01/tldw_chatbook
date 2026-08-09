# Tests/RAG_Eval/conftest.py
"""Fixtures for the retrieval eval harness directory.

Only one thing lives here, and it is deliberately *not* a skip: the
directory-wide autouse fixture below no-ops unless `RAG_EVAL=1`, so the
always-on modules in this directory (metrics, regression gating, fixture
integrity) keep running untouched with no env var set. The env gate itself
is per-module `pytestmark` — see `harness/environment.py:harness_gate`.

What the fixture does do is undo one piece of the suite's own isolation for
the duration of a harness test. `Tests/conftest.py` repoints `HOME` and
`XDG_DATA_HOME` at a throwaway sandbox, which drags
`config.get_model_cache_dir()` — and therefore the embedding model cache —
into a directory that is empty by construction. Left alone, every harness
run would re-download the model into a temp dir and delete it afterwards.
"""
from __future__ import annotations

import os

import pytest

from Tests.RAG_Eval.harness.environment import RAG_EVAL_ENV_VAR, model_cache_dir


@pytest.fixture(autouse=True)
def rag_eval_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the app's model cache at the real one, and forbid downloads.

    The offline flags are what make "the harness never downloads a model" a
    property of the run rather than a claim in a docstring: with them set,
    a cache miss raises instead of quietly fetching 87 MB. The gate in
    `environment.skip_reason` checks the same directory this points at, so
    a miss here means the gate was wrong, which is worth failing over.
    """
    if os.environ.get(RAG_EVAL_ENV_VAR) != "1":
        return

    from tldw_chatbook import config

    cache_dir = model_cache_dir()
    monkeypatch.setattr(config, "get_model_cache_dir", lambda: cache_dir)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
