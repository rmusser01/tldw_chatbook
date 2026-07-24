"""Chroma-only collection lifecycle: legacy adoption migration + index admin.

Persistent-backend only. See
Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.
"""
from __future__ import annotations

from pathlib import Path
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
    # NOTE (SP1 scope, spec §4): only the canonical "default" legacy
    # collection is eligible for auto-adoption. A user who set a custom
    # [rag.vector_store].collection_name never had a "default" collection to
    # adopt from, so their existing data is left alone on disk under its old
    # name and they get a fresh, empty, fingerprinted collection instead --
    # they must re-index. Generalizing adoption to arbitrary legacy names is
    # out of scope here.
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
        # Expected benign cause: a lost race, where a concurrent adopter
        # already did (or is doing) the rename between our existence check
        # and `col.modify()`. Anything else here is a genuine bug (e.g. a
        # Chroma client/Settings collision) that would otherwise be nearly
        # invisible -- `maybe_adopt_legacy_collection` discards this return
        # value and `rag_factory` wraps the whole call in its own broad
        # `except Exception: logger.debug(...)` -- so log at warning here,
        # not debug.
        logger.warning(f"Legacy collection adoption no-op (assumed lost race): {e}")
        return False


def maybe_adopt_legacy_collection(config: RAGConfig) -> None:
    """Adopt the legacy ``default`` collection under ``config``'s fingerprint.

    No-op unless ``config`` is a persistent Chroma store still pointing at the
    canonical ``default`` base. Adopts under the CONFIG ACTIVE AT FIRST
    PERSISTENT CONSTRUCTION (marked provenance ``legacy-adopted`` /
    ``verified=False``), never the shipping default — see spec §4.

    LIMITATION (out of scope for SP1, see spec §4): a user who had already
    set a custom ``collection_name`` before this migration existed is not
    covered -- ``_is_persistent_chroma`` only matches the literal
    ``"default"`` base, so their pre-existing collection is left on disk
    untouched and they get a fresh, empty, fingerprinted collection under
    their custom name instead. They must re-index.
    """
    if not _is_persistent_chroma(config):
        return
    # A legacy collection cannot exist before Chroma has created its database.
    # Avoid constructing a migration-only PersistentClient for fresh profiles:
    # older Chroma releases keep that client's SQLite handle open for the
    # process lifetime, which prevents temporary directories from being
    # removed on Windows.
    persist_directory = Path(config.vector_store.persist_directory)
    if not (persist_directory / "chroma.sqlite3").is_file():
        return
    # Race-safety here (see adopt_legacy_collection's docstring) assumes the
    # race is between two concurrent first-runs of the SAME active config
    # (same fingerprint). Two DIFFERENTLY-fingerprinted configs that still
    # both carry literal collection_name=="default" racing to adopt is out
    # of scope: last writer's target simply wins the rename and no data is
    # destroyed, but the loser's config won't see its own adoption this run.
    target = fingerprinted_collection_name(config)
    provenance = collection_provenance(config, source="legacy-adopted", verified=False)
    adopt_legacy_collection(
        config.vector_store.persist_directory, _LEGACY_BASE, target, provenance
    )


def list_indexes(persist_directory) -> list[dict]:
    """List on-disk collections with provenance + document count."""
    out: list[dict] = []
    try:
        client = _client(persist_directory)
        for col in client.list_collections():
            meta = dict(col.metadata or {})
            out.append({
                "name": col.name,
                "fp": meta.get("fp"),
                "provenance": meta,
                "count": col.count(),
            })
    except Exception as e:
        logger.error(f"list_indexes failed: {e}")
    return out


def delete_index(persist_directory, name: str) -> bool:
    """Delete the collection ``name``. False when absent or on error."""
    try:
        client = _client(persist_directory)
        if name not in {c.name for c in client.list_collections()}:
            return False
        client.delete_collection(name)
        logger.info(f"Deleted index collection '{name}'")
        return True
    except Exception as e:
        logger.error(f"delete_index failed: {e}")
        return False


def index_status(config: RAGConfig) -> dict:
    """Resolved-collection state for ``config``: absent | empty | built."""
    if not (str(config.vector_store.type) == "chroma"
            and config.vector_store.persist_directory is not None):
        return {"state": "absent", "count": 0, "provenance": {}}
    target = fingerprinted_collection_name(config)
    for entry in list_indexes(config.vector_store.persist_directory):
        if entry["name"] == target:
            state = "built" if entry["count"] > 0 else "empty"
            return {"state": state, "count": entry["count"],
                    "provenance": entry["provenance"]}
    return {"state": "absent", "count": 0, "provenance": {}}
