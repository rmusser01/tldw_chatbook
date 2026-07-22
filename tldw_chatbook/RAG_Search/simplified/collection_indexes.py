"""Chroma-only collection lifecycle: legacy adoption migration + index admin.

Persistent-backend only. See
Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from .config import RAGConfig
from .collection_fingerprint import fingerprinted_collection_name, collection_provenance

_LEGACY_BASE = "default"  # the pre-fingerprint canonical collection name


def _client(persist_directory) -> Any:
    import chromadb
    from chromadb.config import Settings
    # NOTE: chromadb (1.5.8) caches one client instance per persist path
    # within a process (SharedSystemClient) and raises ValueError on any
    # later PersistentClient(...) call at the same path whose Settings
    # aren't == the ones first registered ("An instance of Chroma already
    # exists ... with different settings"). This function runs immediately
    # before EnhancedRAGServiceV2 constructs its own Chroma client at this
    # same persist_directory (see vector_store.py's `client` property), so
    # these Settings MUST match that construction exactly (currently
    # anonymized_telemetry=False, allow_reset=True) or real service creation
    # would break the moment this migration runs.
    return chromadb.PersistentClient(
        path=str(persist_directory),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _is_persistent_chroma(config: RAGConfig) -> bool:
    return (
        str(config.vector_store.type) == "chroma"
        and config.vector_store.persist_directory is not None
        and config.vector_store.collection_name == _LEGACY_BASE
    )


def adopt_legacy_collection(
    persist_directory, legacy_name: str, target_name: str, provenance: dict
) -> bool:
    """Rename ``legacy_name`` -> ``target_name`` and stamp provenance, once.

    Idempotent and race-safe: returns False (no-op) when the target already
    exists, the legacy collection is absent, or a concurrent adopter won the
    rename. Never deletes data on a lost race.

    Note on the collection's distance metric ("hnsw:space"): chromadb (as of
    1.5.8) permanently binds a collection's distance metric at creation time.
    ``Collection.modify()`` raises ``ValueError`` if the new metadata dict
    contains an "hnsw:space" key AT ALL -- even to restate the existing
    value -- so it can never be part of the metadata we hand to ``modify()``.
    We rely on that same immutability to our advantage: simply omitting the
    key from the rename's metadata leaves the legacy collection's actual
    index configuration (readable via ``collection.configuration_json``)
    completely untouched, which is exactly the "preserve the original
    metric" behavior we want. We still read the legacy metadata first purely
    to log a diagnostic if it disagrees with the metric recorded in
    ``provenance`` (which documents the *config* we adopted the collection
    under, not the collection's real index metric -- those are allowed to
    differ, e.g. after a config edit, and we must never paper over that by
    relabeling the index).
    """
    try:
        client = _client(persist_directory)
        existing = {c.name for c in client.list_collections()}
        if target_name in existing or legacy_name not in existing:
            return False
        col = client.get_collection(legacy_name)

        legacy_space = (col.metadata or {}).get("hnsw:space")
        adopted_metric = provenance.get("distance_metric")
        if legacy_space and adopted_metric and legacy_space != adopted_metric:
            logger.warning(
                f"Legacy collection '{legacy_name}' was built with "
                f"hnsw:space={legacy_space!r}, but the config it's being "
                f"adopted under records distance_metric={adopted_metric!r}; "
                f"the collection's actual index metric is left untouched."
            )

        # Never include "hnsw:space" in the metadata passed to modify() --
        # see the docstring above.
        new_metadata = {k: v for k, v in provenance.items() if k != "hnsw:space"}
        col.modify(name=target_name, metadata=new_metadata)
        logger.info(f"Adopted legacy collection '{legacy_name}' -> '{target_name}'")
        return True
    except Exception as e:
        # Lost race / Chroma error: the other actor did (or is doing) the rename.
        logger.debug(f"Legacy collection adoption no-op: {e}")
        return False


def maybe_adopt_legacy_collection(config: RAGConfig) -> None:
    """Adopt the legacy ``default`` collection under ``config``'s fingerprint.

    No-op unless ``config`` is a persistent Chroma store still pointing at the
    canonical ``default`` base. Adopts under the CONFIG ACTIVE AT FIRST
    PERSISTENT CONSTRUCTION (marked provenance ``legacy-adopted`` /
    ``verified=False``), never the shipping default — see spec §4.
    """
    if not _is_persistent_chroma(config):
        return
    target = fingerprinted_collection_name(config)
    provenance = collection_provenance(config, source="legacy-adopted", verified=False)
    adopt_legacy_collection(
        config.vector_store.persist_directory, _LEGACY_BASE, target, provenance
    )
