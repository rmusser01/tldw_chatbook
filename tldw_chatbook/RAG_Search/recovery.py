"""Fresh owner observations for projection recovery; persisted records are hints.

A result is valid only for its observed sources/index. Later generation binding
must call recheck_projection under its held scope; this module grants no runtime
or restored-generation authority and never trusts imported collection metadata.
"""

import hashlib
import json
import math
import re
import struct
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Backup_Recovery.rag_inventory import recovery_adapters
from tldw_chatbook.Backup_Recovery.rag_projection_lifetime import participant

_LIMIT = 100_000
_RECORDING = ContextVar("rag_recovery_recording", default=None)


def projection_ready(
    *,
    source_digest: str,
    indexed_source_digest: str,
    compatible: bool,
    reconciled: bool,
) -> bool:
    """Combine owner observations; callers must not use imported flags as inputs."""
    return (
        bool(source_digest)
        and source_digest == indexed_source_digest
        and compatible
        and reconciled
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _rows(ids, documents, metadatas, embeddings):
    """Canonicalize actual stored float32 vectors and installed metadata coercion."""
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise ValueError("projection_rows_incomplete")
    result = {}
    for key, text, meta, vector in zip(ids, documents, metadatas, embeddings):
        clean = {
            k: v if isinstance(v, (str, int, float)) else str(v)
            for k, v in (meta or {}).items()
            if v is not None
        }
        values = [struct.unpack("<f", struct.pack("<f", float(v)))[0] for v in vector]
        if not values or len(values) > 65536 or key in result:
            raise ValueError("projection_vectors_invalid")
        result[key] = (
            clean.get("doc_id"),
            len(values),
            _digest((key, text, clean, values)),
        )
    return result


def _written_rows(batch, ids, documents, metadatas, embeddings, metric):
    """Verify only initial installed float32/cosine conversion, then hash exactly."""
    expected = _rows(ids, documents, metadatas, embeddings)
    if len(batch["ids"]) != len(ids) or set(batch["ids"]) != set(ids):
        raise ValueError("projection_build_membership_invalid")
    positions = {key: index for index, key in enumerate(ids)}
    for offset, key in enumerate(batch["ids"]):
        index = positions[key]
        generated = [
            struct.unpack("<f", struct.pack("<f", float(v)))[0]
            for v in embeddings[index]
        ]
        stored = batch["embeddings"][offset]
        # Reuse the exact document/metadata canonicalization with generated vectors.
        observed = _rows(
            [key],
            [batch["documents"][offset]],
            [batch["metadatas"][offset]],
            [generated],
        )
        if observed[key] != expected[key] or len(stored) != len(generated):
            raise ValueError("projection_build_membership_invalid")
        if metric == "cosine":
            norm = math.sqrt(sum(value * value for value in generated))
            if norm:
                generated = [value / norm for value in generated]
            # Chroma 1.5.8 Rust cosine storage normalizes float32. This bounded
            # initial conversion allowance is never used by later reconciliation.
            valid = all(
                math.isfinite(float(actual)) and abs(actual - value) <= 1e-6
                for actual, value in zip(stored, generated)
            )
        elif metric in {"l2", "ip"}:
            valid = list(stored) == generated
        else:
            valid = False
        if not valid:
            raise ValueError("projection_stored_vector_mismatch")
    return _rows(
        batch["ids"], batch["documents"], batch["metadatas"], batch["embeddings"]
    )


def _configuration(service):
    from .simplified.collection_fingerprint import FINGERPRINT_VERSION, _index_fields

    return _digest((FINGERPRINT_VERSION, _index_fields(service.config)))


def _model(service):
    """Inspect an already-loaded installed backend, without loading or embedding."""
    from .simplified.embeddings_wrapper import (
        EmbeddingsServiceWrapper,
        _DeterministicEmbeddingFactory,
    )

    wrapper = service.embeddings
    if type(wrapper) is not EmbeddingsServiceWrapper:
        raise ValueError("projection_model_identity_unavailable")
    factory = wrapper.factory
    if type(factory) is _DeterministicEmbeddingFactory:
        return _digest(("installed-deterministic-v1", factory.dimension)), (
            id(factory),
        )
    # Existing factory cache exposes the actual loaded model/tokenizer owner.
    from tldw_chatbook.Embeddings.Embeddings_Lib import (
        EmbeddingFactory,
        _HuggingFaceEmbedder,
        _masked_mean,
    )

    if type(factory) is not EmbeddingFactory:
        raise ValueError("projection_model_identity_unavailable")
    with factory._lock:
        key = factory.config.default_model_id
        record = factory._cache.get(key)
        backend = getattr(record["embed"], "__self__", None) if record else None
        spec = factory.config.models.get(key)
        if (
            type(backend) is not _HuggingFaceEmbedder
            or spec is None
            or spec.trust_remote_code
            or backend._pool is not _masked_mean
        ):
            raise ValueError("projection_model_identity_unavailable")
        model, tokenizer = backend._model, backend._tok
        revision = getattr(model.config, "_commit_hash", None)
        tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash")
        if (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or tokenizer_revision != revision
        ):
            raise ValueError("projection_model_revision_unavailable")
        identity = _digest(
            (
                "installed-hf-masked-mean-v1",
                revision,
                model.config.to_dict(),
                backend._max_len,
                str(backend._dtype),
            )
        )
        versions = tuple(
            (name, tuple(value.shape), str(value.dtype), value._version)
            for name, value in model.named_parameters()
        )
        return identity, (id(backend), id(model), id(tokenizer), versions)


def _failure(error):
    """Keep fixed owner prerequisites while suppressing arbitrary backend text."""
    known = {
        "projection_model_identity_unavailable",
        "projection_model_revision_unavailable",
        "projection_source_scope_unavailable",
        "projection_source_owner_unavailable",
        "projection_source_schema_unavailable",
        "projection_source_limit",
        "projection_backend_unavailable",
        "projection_tracking_unavailable",
        "projection_shared_source_scope_required",
        "projection_build_incomplete",
        "projection_build_membership_invalid",
        "projection_identity_changed",
        "projection_source_changed_during_build",
        "projection_final_reconciliation_failed",
        "projection_native_format_unavailable",
        "projection_record_limit",
        "projection_legacy_source_scope_required",
        "projection_stored_vector_mismatch",
    }
    return (
        str(error)
        if isinstance(error, ValueError) and str(error) in known
        else "projection_owner_observation_unavailable"
    )


def _sources(*, media_db=None, chachanotes_db=None, item_types):
    """Read current active documents through their real installed owner builders."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    from .ingestion_indexing import (
        _iter_conversation_entries,
        _iter_media_entries,
        _iter_note_entries,
    )

    selected = tuple(sorted(set(item_types)))
    if not selected or not set(selected) <= {"media", "note", "conversation"}:
        raise ValueError("projection_source_scope_unavailable")
    entries, observations = {}, []
    for kind in selected:
        database = media_db if kind == "media" else chachanotes_db
        expected = MediaDatabase if kind == "media" else CharactersRAGDB
        if type(database) is not expected or database.is_memory_db:
            raise ValueError("projection_source_owner_unavailable")
        path = database.db_path
        info = path.stat()
        query = (
            "SELECT version FROM schema_version LIMIT 1"
            if kind == "media"
            else "SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'"
        )
        version = database.execute_query(query).fetchone()[0]
        if version != expected._CURRENT_SCHEMA_VERSION:
            raise ValueError("projection_source_schema_unavailable")
        schema = [
            tuple(row)
            for row in database.execute_query(
                "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"
            ).fetchmany(2001)
        ]
        if len(schema) > 2000:
            raise ValueError("projection_source_limit")
        observations.append(
            (kind, str(path), info.st_dev, info.st_ino, version, _digest(schema))
        )
        if kind == "media":
            iterator = _iter_media_entries(database, 100)
        elif kind == "note":
            iterator = _iter_note_entries(database, 100)
        else:
            iterator = _iter_conversation_entries(
                database, 100, messages_per_conversation=None
            )
        for entry in iterator:
            key = entry.document["id"]
            if key in entries or len(entries) >= _LIMIT:
                raise ValueError("projection_source_limit")
            entries[key] = entry
    documents = tuple(
        (key, _digest(entry.document)) for key, entry in sorted(entries.items())
    )
    return entries, tuple(observations), _digest((observations, documents))


@dataclass
class _Recording:
    service: Any
    rows: dict = field(default_factory=dict)
    model: Any = None


async def record_stored_chunks(service, ids, embeddings, documents, metadatas):
    """Observe completed real storage only inside explicit recovery indexing."""
    recording = _RECORDING.get()
    if recording is None or recording.service is not service:
        return
    model = _model(service)
    if recording.model is not None and recording.model != model:
        raise ValueError("projection_model_changed")
    recording.model = model
    import asyncio

    native = await asyncio.to_thread(
        service.vector_store.recovery_snapshot,
        written=(ids, documents, metadatas, embeddings),
    )
    rows = native["rows"]
    if set(rows) & set(recording.rows) or len(rows) + len(recording.rows) > _LIMIT:
        raise ValueError("projection_build_membership_invalid")
    recording.rows.update(rows)


@dataclass(frozen=True)
class _Build:
    source_digest: str
    sources: tuple
    configuration: str
    model: tuple
    root: tuple
    collection: str
    index_digest: str
    dimensions: tuple
    record_key: str


@dataclass(frozen=True)
class ProjectionObservation:
    """Point-in-time owner result; local generation binding must recheck it."""

    source_digest: str = ""
    indexed_source_digest: str = ""
    index_digest: str = ""
    identity_digest: str = ""
    record_key: str = ""
    issues: tuple[str, ...] = ()

    @property
    def ready(self):
        return projection_ready(
            source_digest=self.source_digest,
            indexed_source_digest=self.indexed_source_digest,
            compatible=not self.issues,
            reconciled=not self.issues,
        )


def _qualified_service(service):
    from .simplified.rag_service import RAGService
    from .simplified.vector_store import ChromaVectorStore

    if (
        type(service) is not RAGService
        or type(service.vector_store) is not ChromaVectorStore
    ):
        raise ValueError("projection_backend_unavailable")


@participant.async_operation
async def reconcile_projection(
    service,
    *,
    media_db=None,
    chachanotes_db=None,
    item_types=("media", "note", "conversation"),
):
    """Reconcile without writes, model acquisition, re-embedding or auto-rebuild."""
    import asyncio

    try:
        _qualified_service(service)
        build = getattr(service, "_recovery_projection_build", None)
        if type(build) is not _Build:
            return ProjectionObservation(
                issues=("projection_build_provenance_missing",)
            )
        _, sources, source_digest = _sources(
            media_db=media_db, chachanotes_db=chachanotes_db, item_types=item_types
        )
        native = await asyncio.to_thread(service.vector_store.recovery_snapshot)
        model = _model(service)
        configuration = _configuration(service)
        issues = []
        if source_digest != build.source_digest or sources != build.sources:
            issues.append("projection_source_changed")
        if (native["root"], native["collection"], configuration, model) != (
            build.root,
            build.collection,
            build.configuration,
            build.model,
        ):
            issues.append("projection_identity_incompatible")
        index_digest = _digest(native["rows"])
        if index_digest != build.index_digest:
            issues.append("projection_membership_or_content_changed")
        if native["metric"] != service.config.distance_metric:
            issues.append("projection_metric_incompatible")
        # Repeat owner observations around verification; later binding rechecks again.
        if _sources(
            media_db=media_db, chachanotes_db=chachanotes_db, item_types=item_types
        )[1:] != (sources, source_digest):
            issues.append("projection_source_changed_during_verification")
        if await asyncio.to_thread(service.vector_store.recovery_snapshot) != native:
            issues.append("projection_changed_during_verification")
        if _model(service) != model or _configuration(service) != configuration:
            issues.append("projection_identity_changed_during_verification")
        identity = _digest(
            (sources, native["root"], native["collection"], configuration, model)
        )
        return ProjectionObservation(
            source_digest,
            build.source_digest,
            index_digest,
            identity,
            build.record_key,
            tuple(issues),
        )
    except Exception as error:  # noqa: BLE001 -- native/owner failures return no authority or backend text.
        return ProjectionObservation(issues=(_failure(error),))


async def recheck_projection(observation, service, **sources):
    """Recompute exact observations; an earlier ready result is never a lease."""
    if type(observation) is not ProjectionObservation or not observation.ready:
        return False
    current = await reconcile_projection(service, **sources)
    return current.ready and current == observation


@participant.async_operation
async def rebuild_projection(
    service,
    indexing_db,
    *,
    media_db=None,
    chachanotes_db=None,
    item_types,
    batch_size=16,
):
    """Explicit Backfill recovery path; ordinary incremental Backfill is unchanged."""
    import asyncio

    from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB

    from .ingestion_indexing import _clear_service_search_cache

    summary = {
        "status": "partial",
        "indexed": 0,
        "removed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
        "by_type": {},
    }
    recording = _Recording(service)
    token = _RECORDING.set(recording)
    previous = getattr(service, "_recovery_projection_build", None)
    service._recovery_projection_build = None
    try:
        _qualified_service(service)
        if type(indexing_db) is not RAGIndexingDB or batch_size < 1:
            raise ValueError("projection_tracking_unavailable")
        entries, sources, source_digest = _sources(
            media_db=media_db, chachanotes_db=chachanotes_db, item_types=item_types
        )
        configuration = _configuration(service)
        recording.model = _model(service)
        # Creating an empty collection is allowed only by this explicit rebuild.
        _ = service.vector_store.collection
        old = await asyncio.to_thread(service.vector_store.recovery_snapshot)
        # A document prefix is not a database namespace. A populated legacy
        # collection needs the existing user-reviewed whole-collection clear;
        # only this service's prior complete build can establish repair scope.
        if old["rows"] and (
            type(previous) is not _Build
            or (previous.sources, previous.root, previous.collection)
            != (sources, old["root"], old["collection"])
        ):
            raise ValueError("projection_legacy_source_scope_required")
        # Later ordinary writes or direct native additions can belong to another
        # database, even with the same source-type prefix. Only the unchanged
        # verified native rows retain the previous build's deletion authority.
        if old["rows"] and _digest(old["rows"]) != previous.index_digest:
            raise ValueError("projection_shared_source_scope_required")
        for doc_id in sorted({row[0] for row in old["rows"].values()}):
            await asyncio.to_thread(service.vector_store.delete_document, doc_id)
            if doc_id not in entries:
                summary["removed"] += 1
        values = list(entries.values())
        for offset in range(0, len(values), batch_size):
            batch = values[offset : offset + batch_size]
            results = await service.index_batch_optimized(
                [e.document for e in batch], show_progress=False
            )
            if len(results) != len(batch) or any(
                not result.success or result.error for result in results
            ):
                raise ValueError("projection_build_incomplete")
            if {r.doc_id for r in results} != {e.document["id"] for e in batch}:
                raise ValueError("projection_build_incomplete")
            await _clear_service_search_cache(service)
            counts = {r.doc_id: r.chunks_created for r in results}
            indexing_db.mark_items_indexed(
                [
                    (e.item_id, e.item_type, e.last_modified, counts[e.document["id"]])
                    for e in batch
                ]
            )
            summary["indexed"] += len(batch)
        await _clear_service_search_cache(service)
        native = await asyncio.to_thread(service.vector_store.recovery_snapshot)
        model = recording.model or _model(service)
        if recording.rows != native["rows"] or {
            row[0] for row in recording.rows.values()
        } != set(entries):
            raise ValueError("projection_build_membership_invalid")
        if (native["root"], native["collection"], native["metric"]) != (
            old["root"],
            old["collection"],
            service.config.distance_metric,
        ):
            raise ValueError("projection_identity_changed")
        if _model(service) != model or _configuration(service) != configuration:
            raise ValueError("projection_identity_changed")
        if _sources(
            media_db=media_db, chachanotes_db=chachanotes_db, item_types=item_types
        )[1:] != (sources, source_digest):
            raise ValueError("projection_source_changed_during_build")
        record_key = "projection-v1:" + _digest(
            (native["root"][0], native["collection"])
        )
        build = _Build(
            source_digest,
            sources,
            configuration,
            model,
            native["root"],
            native["collection"],
            _digest(native["rows"]),
            tuple(sorted({r[1] for r in native["rows"].values()})),
            record_key,
        )
        indexing_db.update_collection_state(
            record_key,
            len(entries),
            len(entries),
            {
                "version": 1,
                "source_digest": source_digest,
                "sources": sources,
                "index_digest": build.index_digest,
                "configuration": configuration,
                "model": model[0],
                "dimensions": build.dimensions,
                "root": native["root"],
                "collection": native["collection"],
            },
        )
        service._recovery_projection_build = build
        result = await reconcile_projection(
            service,
            media_db=media_db,
            chachanotes_db=chachanotes_db,
            item_types=item_types,
        )
        if not result.ready:
            raise ValueError("projection_final_reconciliation_failed")
        summary.update(status="ok", projection=result)
    except Exception as error:  # noqa: BLE001 -- native/tracking failures never certify a partial build.
        service._recovery_projection_build = None
        summary["failed"] += 1
        summary["errors"].append(_failure(error))
        summary["projection"] = ProjectionObservation(issues=(_failure(error),))
    finally:
        _RECORDING.reset(token)
    return summary


__all__ = [
    "projection_ready",
    "recheck_projection",
    "reconcile_projection",
    "recovery_adapters",
]
