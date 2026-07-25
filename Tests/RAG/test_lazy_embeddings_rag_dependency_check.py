"""Regression coverage for task-628: lazy embeddings_rag dependency check.

Live UAT (see backlog/tasks/task-628) found that on the default lazy
dependency-checking configuration, ``DEPENDENCIES_AVAILABLE["embeddings_rag"]``
never gets populated -- ``ensure_dependencies_checked()`` /
``check_embeddings_rag_deps()`` have no call site anywhere the app actually
reaches during normal use -- so it stays at its pristine ``False`` default
for the app's entire lifetime even when torch/transformers/numpy/chromadb/
sentence_transformers are all genuinely importable. ``EmbeddingFactory``
(tldw_chatbook/Embeddings/Embeddings_Lib.py) then refuses every
construction, and RAG Backfill (which routes through
``ingestion_indexing.get_shared_rag_service`` ->
``simplified.create_rag_service`` -> ``EmbeddingsServiceWrapper`` ->
``EmbeddingFactory``) fails with a misleading "install the dependencies"
error even on a fully-provisioned install.

These tests reset the registry to that exact pristine "never checked" state
and confirm construction now succeeds when the real packages ARE importable
-- using the cheap ``embeddings_rag_deps_installed()`` find_spec probe (not
the buggy stale flag) to decide whether to skip, so these tests are honest
about needing the real extras installed.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils import optional_deps
from tldw_chatbook.Utils.optional_deps import (
    DEPENDENCIES_AVAILABLE,
    embeddings_rag_deps_installed,
    reset_dependency_checks,
)

pytestmark = pytest.mark.skipif(
    not embeddings_rag_deps_installed(),
    reason=(
        "These tests specifically prove the lazy-check gate resolves to "
        "available when the embeddings_rag extras ARE installed; without "
        "them, there's nothing to distinguish this bug from a real absence."
    ),
)


@pytest.fixture(autouse=True)
def _pristine_registry():
    """Reset the shared registry to its never-checked default before/after.

    Mirrors what a fresh app process looks like under the default lazy
    dependency-checking mode: nothing has called check_embeddings_rag_deps()
    yet, so the flag sits at its pristine False default even though the
    packages are importable.
    """
    reset_dependency_checks()
    try:
        yield
    finally:
        reset_dependency_checks()


def test_registry_starts_pristine_false_before_any_lazy_check():
    """Sanity check: confirms the test fixture actually reproduces the bug's
    starting condition (nothing has run check_embeddings_rag_deps() yet)."""
    assert DEPENDENCIES_AVAILABLE.get("embeddings_rag", False) is False


def test_embedding_factory_construction_succeeds_on_first_use_without_eager_check():
    """The UAT-reported symptom, isolated to the EmbeddingFactory gate.

    With the registry still pristine (as it always is under the default lazy
    mode, since nothing ever calls the lazy-check function), constructing an
    EmbeddingFactory must succeed when the real deps are importable instead
    of raising "EmbeddingFactory requires embeddings/RAG dependencies."
    """
    from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory

    # Empty `models` is a structurally valid EmbeddingConfigSchema -- this
    # test only needs to reach past the dependency gate, not load a model.
    factory = EmbeddingFactory({"models": {}})
    assert factory is not None

    # The registry should now correctly reflect reality instead of staying
    # stuck at its pristine False default.
    assert DEPENDENCIES_AVAILABLE.get("embeddings_rag", False) is True


def test_create_rag_service_succeeds_for_backfill_without_a_prior_eager_check(
    monkeypatch,
):
    """UAT symptom path: ``simplified.create_rag_service`` for Backfill.

    ``ingestion_indexing.get_shared_rag_service`` (Settings > RAG's "Backfill
    now" seam) calls this exact function
    (``create_rag_service(profile_name=..., config=...)``) to build the
    shared RAG runtime; the live UAT's "Failed to create shared RAG service"
    error bottomed out in this call raising ImportError from the
    EmbeddingFactory gate. A real (non-mock) embedding model + in-memory
    vector store is used so the dependency gate is genuinely exercised
    end-to-end -- the embedding-dimension self-probe EnhancedRAGServiceV2
    performs during construction is stubbed out since it would otherwise
    try to download the real model over the network, which is unrelated to
    what this test verifies (the earlier, previously-broken dependency
    gate).
    """
    from tldw_chatbook.RAG_Search.simplified import (
        embeddings_wrapper as embeddings_wrapper_module,
    )
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service

    monkeypatch.setattr(
        embeddings_wrapper_module.EmbeddingsServiceWrapper,
        "get_embedding_dimension",
        lambda self: 768,
    )

    cfg = RAGConfig()
    cfg.embedding.model = "mxbai-embed-large-v1"  # real (non-mock) model id
    cfg.embedding.device = "cpu"
    cfg.vector_store.type = "memory"
    cfg.vector_store.persist_directory = None

    service = create_rag_service(profile_name="hybrid_basic", config=cfg)

    assert service is not None
    assert DEPENDENCIES_AVAILABLE.get("embeddings_rag", False) is True


def test_manually_forced_unavailable_is_still_honored():
    """A caller that explicitly marks embeddings_rag unavailable (e.g. a real
    failed probe, or force_recheck_embeddings() after uninstalling) must
    still be refused -- the lazy check only fills in a never-checked True
    default, it must not silently override an explicit False determination
    reached by actually running the check.
    """
    from tldw_chatbook.Embeddings.Embeddings_Lib import EmbeddingFactory

    # Simulate "the real probe already ran and found deps missing" by
    # patching the underlying checker itself, not just the flag -- our lazy
    # gate re-runs the checker whenever it reads False, so merely poking the
    # flag would be silently overwritten by a truthful re-check.
    original_check = optional_deps.check_embeddings_rag_deps
    optional_deps.DEPENDENCIES_AVAILABLE["embeddings_rag"] = False
    optional_deps.check_embeddings_rag_deps = lambda: False
    try:
        with pytest.raises(ImportError, match="embeddings/RAG dependencies"):
            EmbeddingFactory({"models": {}})
    finally:
        optional_deps.check_embeddings_rag_deps = original_check
