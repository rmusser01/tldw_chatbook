"""The RAG search-mode vocabulary and its normalizer (stdlib only).

Split out of ``RAG_Search/simplified/active_config.py`` by TASK-21731 for the
same reason ``tldw_chatbook/chunking_engine_version.py`` was split out of
``Chunking/`` by TASK-21102: a boot-path module needed one pure helper, and
importing the module that held it executed a service tree.

``Library/library_local_rag_search_service.py`` is on the app's import path
and calls ``normalize_rag_search_mode`` to read the active profile's mode.
Importing it from ``active_config`` pulled ``simplified.rag_service`` ->
``chunking_service`` -> the whole ``Chunking`` engine -> ``Internal_Prompts``:
67 modules of this repo executed before first paint, for a function that
compares a string against a three-element frozenset.

``active_config`` re-imports both names from here, so there is exactly one
``_RAG_SEARCH_MODES`` object and one ``normalize_rag_search_mode`` function in
the process — no second copy that can drift from the vocabulary
``SearchConfig.default_search_mode`` validates against.

This module must stay import-cheap: **standard library only**, no
``tldw_chatbook`` imports. Guarded by
``Tests/Packaging/test_rag_boot_import_closure.py``.
"""

from __future__ import annotations

#: The exact vocabulary ``SearchConfig.default_search_mode`` supports
#: (``RAG_Search/simplified/config.py``). Anything else -- a hand-edited
#: TOML, a future mode this build does not know -- normalizes to
#: ``"semantic"``, the historical behavior.
RAG_SEARCH_MODES = frozenset({"plain", "semantic", "hybrid"})


def normalize_rag_search_mode(value: object) -> str:
    """Return a supported exact search mode.

    Args:
        value: Candidate search mode.

    Returns:
        ``value`` when it is a supported mode; otherwise ``"semantic"``.
    """
    return (
        value if isinstance(value, str) and value in RAG_SEARCH_MODES else "semantic"
    )
