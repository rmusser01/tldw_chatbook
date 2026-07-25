"""Locks the public `create_rag_service` seam (task-625).

`tldw_chatbook.RAG_Search.simplified.rag_service` used to define its own
`create_rag_service(embedding_model=None, vector_store="chroma", ...)`, but
`simplified/__init__.py` never imported it -- it only imports the
same-named `rag_factory.create_rag_service(profile_name="hybrid_basic", ...)`,
which is what every real caller (production and tests) actually reaches
through the public package seam. The `rag_service.py` version was therefore
unreachable dead code with a different, misleading signature. These tests
pin the public seam's identity and confirm the shadowed function is gone.

No optional deps (embeddings_rag) are required to import either module, so
this file is intentionally NOT gated behind DEPENDENCIES_AVAILABLE.
"""

import inspect


def test_public_seam_resolves_to_rag_factorys_create_rag_service():
    from tldw_chatbook.RAG_Search.simplified import create_rag_service
    from tldw_chatbook.RAG_Search.simplified import rag_factory

    assert create_rag_service is rag_factory.create_rag_service

    # Sanity: it's rag_factory's profile-based signature, not a
    # vector_store/persist_dir one.
    params = list(inspect.signature(create_rag_service).parameters)
    assert params[0] == "profile_name"


def test_rag_service_module_has_no_shadowed_create_rag_service():
    from tldw_chatbook.RAG_Search.simplified import rag_service

    assert not hasattr(rag_service, "create_rag_service"), (
        "rag_service.py must not define its own create_rag_service -- it was "
        "dead code shadowed by rag_factory.create_rag_service at the public "
        "package seam (task-625)."
    )
