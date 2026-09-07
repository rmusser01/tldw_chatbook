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

    Forcing `huggingface_hub.constants.HF_HUB_OFFLINE` — rather than only
    setting the env var — is what makes "the harness never downloads a
    model" a property of the run instead of a claim in a docstring. That
    constant is frozen at huggingface_hub import time, so an env var written
    here, at test setup, arrives far too late to matter; the constant itself
    is read through `is_offline_mode()` on every request, by hf_hub and by
    transformers alike, so writing it here does take effect. `environment.py`
    sets the env var at module top for the case where huggingface_hub has
    not been imported yet; this covers the case where it has. See that
    module's offline-latch comment for the measurement.

    `monkeypatch.setattr` rather than a bare assignment so the constant is
    restored afterwards: a harness run that left the whole process offline
    would break any unrelated test later in the session.

    The gate in `environment.skip_reason` checks the same directory this
    points at, so a cache miss here means the gate was wrong — which, with
    downloads genuinely blocked, now fails loudly instead of silently
    fetching 87 MB into the user's real cache.
    """
    if os.environ.get(RAG_EVAL_ENV_VAR) != "1":
        return

    from huggingface_hub import constants

    from tldw_chatbook import config

    cache_dir = model_cache_dir()
    monkeypatch.setattr(config, "get_model_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
