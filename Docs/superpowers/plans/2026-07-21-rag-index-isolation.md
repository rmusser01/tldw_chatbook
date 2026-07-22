# SP1 — Per-Embedding-Config Index Isolation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make vector collections keyed by the config that built them, so changing embedding model / chunking / distance metric points at a distinct index instead of silently corrupting a shared one.

**Architecture:** A pure `collection_fingerprint.py` computes a versioned fingerprint over index-determining config fields and the fingerprinted collection name + provenance. The single store-construction seam (`RAGService.__init__`, `rag_service.py:139`, inherited by `EnhancedRAGServiceV2`) resolves `collection_name` through it, so ingestion, backfill, and search all target the same collection. A `collection_indexes.py` module handles Chroma-only lifecycle: a provenance-aware, idempotent, race-safe legacy-collection adoption migration (wired into `create_rag_service`), plus `list_indexes`/`delete_index`/`index_status` over the existing store CRUD. Empty-index honesty reuses the existing `semantic_index_is_empty` probe — a fresh fingerprinted collection has count 0 and auto-triggers it.

**Tech Stack:** Python 3.11+, ChromaDB (persistent), `hashlib`, existing `RAGConfig` dataclasses, pytest with the `Tests/RAG/simplified/conftest.py` fixtures (`mock_embeddings`, `chroma_persist_dir`, `test_rag_config`, `memory_rag_config`).

## Global Constraints

- **Spec:** `Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md`; program overview `Docs/superpowers/specs/2026-07-21-rag-settings-profiles-overview.md`. Plan **only SP1**.
- **Fingerprint input set (index-determining, verbatim):** `embedding.model`, `embedding.max_length`; **all** `ChunkingConfig` fields (`chunk_size`, `chunk_overlap`, `chunking_method`, `min_chunk_size`, `max_chunk_size`, `enable_parent_retrieval`, `parent_size_multiplier`, `preserve_structure`, `clean_artifacts`, `preserve_tables`); `vector_store.distance_metric`. **Excluded** (query-time / non-output): everything in `SearchConfig`, `embedding.batch_size/device/cache_size/api_key/base_url`, reranking, citations, `vector_store.type/persist_directory`.
- **Fingerprint is a persistent contract:** one function, a `FINGERPRINT_VERSION` constant folded into the hash, inputs normalized before hashing (`"400"` str and `400` int MUST hash identically). Any future change to inputs/algorithm is a migration, never a silent tweak.
- **Chroma name rules:** output name 3–63 chars, charset `[a-zA-Z0-9._-]`, starts/ends alphanumeric, no `..`. The name builder guarantees this (truncate the base, keep the suffix).
- **Persistent-backend only:** migration, provenance stamping, and index admin apply to the Chroma store. The in-memory store rebuilds per session — fingerprinting its name is harmless but there is nothing to migrate/preserve.
- **Migration invariant (cross-SP):** the fingerprint the legacy `default` collection is adopted under MUST equal the fingerprint SP2's "Imported settings" profile will produce (overview §4.1). SP1 adopts under **the config active at first persistent construction**; SP2's import must resolve to that same config. This plan documents the contract; the end-to-end test lands in SP2.
- **Naming (match neighbors):** new modules live in `tldw_chatbook/RAG_Search/simplified/` beside `config.py`/`vector_store.py`. snake_case functions, Google-style docstrings, type hints on public APIs.
- **Tests:** real in-memory SQLite + mock embeddings; Chroma tests use the `chroma_persist_dir` fixture and the `requires_chromadb` marker. Run in the venv.

---

## File Structure

- **Create** `tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py` — pure: `FINGERPRINT_VERSION`, `fingerprint_collection`, `fingerprinted_collection_name`, `collection_provenance`, name-safety helpers. No IO.
- **Create** `tldw_chatbook/RAG_Search/simplified/collection_indexes.py` — Chroma-only lifecycle: `adopt_legacy_collection`, `maybe_adopt_legacy_collection`, `list_indexes`, `delete_index`, `index_status`.
- **Modify** `tldw_chatbook/RAG_Search/simplified/vector_store.py` — thread `collection_metadata` into `ChromaVectorStore.__init__`, `InMemoryVectorStore.__init__`, and `create_vector_store`; merge into `get_or_create_collection` at creation.
- **Modify** `tldw_chatbook/RAG_Search/simplified/rag_service.py:139-144` — resolve `collection_name` via `fingerprinted_collection_name(self.config)` and pass `collection_metadata=collection_provenance(self.config)`.
- **Modify** `tldw_chatbook/RAG_Search/simplified/rag_factory.py` — call `maybe_adopt_legacy_collection(rag_config)` before constructing the service.
- **Create tests:** `Tests/RAG/simplified/test_collection_fingerprint.py`, `Tests/RAG/simplified/test_collection_indexes.py`, `Tests/RAG/simplified/test_index_isolation_integration.py`.

---

## Task 1: Pure fingerprint module

**Files:**
- Create: `tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py`
- Test: `Tests/RAG/simplified/test_collection_fingerprint.py`

**Interfaces:**
- Consumes: `RAGConfig` (and its `.embedding`, `.chunking`, `.vector_store`) from `tldw_chatbook/RAG_Search/simplified/config.py`.
- Produces:
  - `FINGERPRINT_VERSION: int`
  - `fingerprint_collection(config: RAGConfig) -> str` — 12-hex-char digest.
  - `fingerprinted_collection_name(config: RAGConfig) -> str` — `"{safe_base}__{fingerprint}"`, always a valid Chroma name.
  - `collection_provenance(config: RAGConfig, *, source: str = "built", verified: bool = True) -> dict` — metadata dict stamped on collections.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/simplified/test_collection_fingerprint.py
import re
from dataclasses import replace

from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    FINGERPRINT_VERSION,
    fingerprint_collection,
    fingerprinted_collection_name,
    collection_provenance,
)

def _cfg(**vs):
    return RAGConfig(
        embedding=EmbeddingConfig(model="modelA", max_length=512),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100, chunking_method="words"),
        vector_store=VectorStoreConfig(type="memory", collection_name="default",
                                       distance_metric="cosine", **vs),
    )

def test_fingerprint_is_deterministic():
    assert fingerprint_collection(_cfg()) == fingerprint_collection(_cfg())

def test_str_and_int_chunk_size_hash_identically():
    a = _cfg()
    b = _cfg()
    b.chunking.chunk_size = "400"  # str from TOML
    assert fingerprint_collection(a) == fingerprint_collection(b)

def test_query_only_diff_shares_fingerprint():
    a = _cfg()
    b = _cfg()
    b.search.default_top_k = 999          # query-time, excluded
    b.search.enable_reranking = True      # query-time, excluded
    assert fingerprint_collection(a) == fingerprint_collection(b)

def test_index_field_diffs_change_fingerprint():
    base = fingerprint_collection(_cfg())
    m = _cfg(); m.embedding.model = "modelB"
    ch = _cfg(); ch.chunking.chunk_size = 401
    mx = _cfg(); mx.chunking.max_chunk_size = 1001
    metric = _cfg(); metric.vector_store.distance_metric = "l2"
    ml = _cfg(); ml.embedding.max_length = 256
    for c in (m, ch, mx, metric, ml):
        assert fingerprint_collection(c) != base

def test_name_is_valid_chroma_name():
    name = fingerprinted_collection_name(_cfg())
    assert name.startswith("default__")
    assert 3 <= len(name) <= 63
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]", name)
    assert ".." not in name

def test_long_base_truncated_but_suffix_preserved():
    c = _cfg()
    c.vector_store.collection_name = "x" * 200
    name = fingerprinted_collection_name(c)
    assert len(name) <= 63
    assert name.endswith("__" + fingerprint_collection(c))

def test_provenance_carries_version_and_fields():
    p = collection_provenance(_cfg(), source="legacy-adopted", verified=False)
    assert p["fp_version"] == FINGERPRINT_VERSION
    assert p["fp"] == fingerprint_collection(_cfg())
    assert p["embedding_model"] == "modelA"
    assert p["distance_metric"] == "cosine"
    assert p["source"] == "legacy-adopted"
    assert p["verified"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/simplified/test_collection_fingerprint.py -q`
Expected: FAIL — `ModuleNotFoundError: ...collection_fingerprint`.

- [ ] **Step 3: Write minimal implementation**

```python
# tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py
"""Deterministic, versioned fingerprint of index-determining RAG config.

The fingerprint keys a vector collection to the config that built it: two
configs that differ only in query-time settings share a collection; two that
differ in embedding model, any chunking field, or distance metric get
separate collections. See
Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.

This module is PURE (no IO). The fingerprint is a persistent contract:
changing FINGERPRINT_VERSION or the input set re-points every collection, so
treat any such change as a migration.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, List, Tuple

from .config import RAGConfig

#: Bump ONLY with a migration mapping old->new collection names.
FINGERPRINT_VERSION: int = 1

_MAX_CHROMA_NAME = 63
_MIN_CHROMA_NAME = 3
_FP_LEN = 12  # hex chars of the digest kept


def _index_fields(config: RAGConfig) -> List[Tuple[str, Any]]:
    """Ordered (name, value) pairs of every index-determining field.

    Values are normalized so that equal logical configs hash identically
    regardless of source type (TOML strings vs ints/bools).
    """
    e = config.embedding
    c = config.chunking
    v = config.vector_store
    raw: List[Tuple[str, Any]] = [
        ("embedding.model", e.model),
        ("embedding.max_length", e.max_length),
        ("chunking.chunk_size", c.chunk_size),
        ("chunking.chunk_overlap", c.chunk_overlap),
        ("chunking.chunking_method", c.chunking_method),
        ("chunking.min_chunk_size", c.min_chunk_size),
        ("chunking.max_chunk_size", c.max_chunk_size),
        ("chunking.enable_parent_retrieval", c.enable_parent_retrieval),
        ("chunking.parent_size_multiplier", c.parent_size_multiplier),
        ("chunking.preserve_structure", c.preserve_structure),
        ("chunking.clean_artifacts", c.clean_artifacts),
        ("chunking.preserve_tables", c.preserve_tables),
        ("vector_store.distance_metric", v.distance_metric),
    ]
    return [(k, _normalize(val)) for k, val in raw]


def _normalize(value: Any) -> Any:
    """Coerce a config value to a canonical, hash-stable form."""
    if isinstance(value, bool):
        return bool(value)               # keep bools distinct from ints
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # ints-as-floats (400.0) normalize to int; real floats keep precision
        return int(value) if float(value).is_integer() else float(value)
    if value is None:
        return None
    s = str(value).strip()
    # numeric strings ("400") normalize to int so they match int configs
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def fingerprint_collection(config: RAGConfig) -> str:
    """Return the 12-hex-char fingerprint of ``config``'s index-determining fields."""
    payload = {
        "version": FINGERPRINT_VERSION,
        "fields": _index_fields(config),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:_FP_LEN]


def _safe_base(name: str, reserved: int) -> str:
    """A Chroma-safe base name leaving ``reserved`` chars for the suffix."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in str(name))
    cleaned = cleaned.strip("._-") or "rag"
    budget = max(1, _MAX_CHROMA_NAME - reserved)
    cleaned = cleaned[:budget].strip("._-") or "rag"
    return cleaned


def fingerprinted_collection_name(config: RAGConfig) -> str:
    """Return ``"{base}__{fingerprint}"`` — always a valid Chroma name."""
    fp = fingerprint_collection(config)
    base = _safe_base(config.collection_name, reserved=len(fp) + 2)  # "__" + fp
    name = f"{base}__{fp}"
    if len(name) < _MIN_CHROMA_NAME:  # pragma: no cover - fp alone is 12 chars
        name = f"rag__{fp}"
    return name


def collection_provenance(
    config: RAGConfig, *, source: str = "built", verified: bool = True
) -> dict:
    """Metadata stamped on a collection at creation, read back by the UI (SP3)."""
    return {
        "fp_version": FINGERPRINT_VERSION,
        "fp": fingerprint_collection(config),
        "embedding_model": str(config.embedding.model),
        "chunk_size": int(config.chunking.chunk_size),
        "chunk_overlap": int(config.chunking.chunk_overlap),
        "chunking_method": str(config.chunking.chunking_method),
        "distance_metric": str(config.vector_store.distance_metric),
        "source": source,
        "verified": bool(verified),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/simplified/test_collection_fingerprint.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py \
        Tests/RAG/simplified/test_collection_fingerprint.py
git commit -m "feat(rag): versioned collection fingerprint over index-determining config"
```

---

## Task 2: Apply the fingerprint at the store seam + thread provenance

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/vector_store.py` (`ChromaVectorStore.__init__` ~182, `collection` property ~292, `InMemoryVectorStore.__init__` ~788, `create_vector_store` ~1502)
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py:139-144`
- Test: `Tests/RAG/simplified/test_index_isolation_integration.py`

**Interfaces:**
- Consumes: `fingerprinted_collection_name`, `collection_provenance` (Task 1).
- Produces: services now open the collection named `fingerprinted_collection_name(self.config)`; created Chroma collections carry `collection_provenance(...)` in their metadata alongside `hnsw:space`.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/simplified/test_index_isolation_integration.py
import pytest
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig, SearchConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    fingerprinted_collection_name,
)
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService


def _chroma_cfg(persist_dir, **over):
    vs = dict(type="chroma", persist_directory=persist_dir,
              collection_name="default", distance_metric="cosine")
    vs.update(over.pop("vector_store", {}))
    return RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100, chunking_method="words"),
        vector_store=VectorStoreConfig(**vs),
        search=SearchConfig(enable_cache=False),
    )


@pytest.mark.requires_chromadb
def test_service_opens_fingerprinted_collection(chroma_persist_dir):
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    expected = fingerprinted_collection_name(cfg)
    assert svc.vector_store.collection_name == expected
    assert expected != "default"


@pytest.mark.requires_chromadb
def test_query_only_diff_shares_collection(chroma_persist_dir):
    a = _chroma_cfg(chroma_persist_dir)
    b = _chroma_cfg(chroma_persist_dir)
    b.search.default_top_k = 42
    assert (RAGService(a).vector_store.collection_name
            == RAGService(b).vector_store.collection_name)


@pytest.mark.requires_chromadb
def test_metric_and_model_diffs_fork_collection(chroma_persist_dir):
    base = RAGService(_chroma_cfg(chroma_persist_dir)).vector_store.collection_name
    metric = _chroma_cfg(chroma_persist_dir, vector_store={"distance_metric": "l2"})
    model = _chroma_cfg(chroma_persist_dir)
    model.embedding.model = "other-model"
    assert RAGService(metric).vector_store.collection_name != base
    assert RAGService(model).vector_store.collection_name != base


@pytest.mark.requires_chromadb
def test_created_collection_carries_provenance(chroma_persist_dir):
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    meta = svc.vector_store.collection.metadata  # forces get_or_create
    assert meta.get("fp") is not None
    assert meta.get("embedding_model") == "mock"
    assert meta.get("hnsw:space") == "cosine"  # existing key preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/simplified/test_index_isolation_integration.py -q`
Expected: FAIL — `collection_name` is still `"default"` (no fingerprint), and metadata lacks `fp`.

- [ ] **Step 3a: Thread `collection_metadata` through the store**

In `tldw_chatbook/RAG_Search/simplified/vector_store.py`, `ChromaVectorStore.__init__` — add the parameter and store it:

```python
    def __init__(
        self,
        persist_directory: Union[str, Path],
        collection_name: str = "default",
        distance_metric: str = "cosine",
        collection_metadata: Optional[dict] = None,
    ):
```
Immediately after `self.distance_metric = distance_metric` add:
```python
        self.collection_metadata = dict(collection_metadata or {})
```
In the `collection` property, merge provenance into the create metadata (keep `hnsw:space`):
```python
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": metric_map.get(self.distance_metric, "cosine"),
                    **self.collection_metadata,
                },
            )
```

In `InMemoryVectorStore.__init__`, accept and ignore the same kwarg so the factory can pass it uniformly:
```python
        collection_metadata: Optional[dict] = None,
```
(add to the signature; no body change needed).

In `create_vector_store`, add the pass-through:
```python
def create_vector_store(
    store_type: str,
    persist_directory: Optional[Union[str, Path]] = None,
    collection_name: str = "default",
    distance_metric: str = "cosine",
    collection_metadata: Optional[dict] = None,
    **kwargs,
) -> VectorStore:
```
and forward `collection_metadata=collection_metadata` to both the `ChromaVectorStore(...)` and `InMemoryVectorStore(...)` constructions inside it.

- [ ] **Step 3b: Resolve the fingerprinted name at the seam**

In `tldw_chatbook/RAG_Search/simplified/rag_service.py`, add near the top-of-file imports:
```python
from .collection_fingerprint import fingerprinted_collection_name, collection_provenance
```
Replace the store construction at `:139-144`:
```python
        self.vector_store = create_vector_store(
            store_type=self.config.vector_store_type,
            persist_directory=self.config.persist_directory,
            collection_name=fingerprinted_collection_name(self.config),
            distance_metric=self.config.distance_metric,
            collection_metadata=collection_provenance(self.config),
        )
```

- [ ] **Step 4: Run the new test + the existing simplified suite**

Run: `pytest Tests/RAG/simplified/test_index_isolation_integration.py -q`
Expected: PASS (4 tests).

Run: `pytest Tests/RAG/simplified/ -q`
Expected: PASS. **Watch for** tests asserting a raw collection name on a service's store (e.g. `assert store.collection_name == "default"` / a unique test name). The correct fix is to expect `fingerprinted_collection_name(config)`, not to revert the seam — fold that change into this task if any surface.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/vector_store.py \
        tldw_chatbook/RAG_Search/simplified/rag_service.py \
        Tests/RAG/simplified/test_index_isolation_integration.py
git commit -m "feat(rag): resolve store collection via fingerprint + stamp provenance"
```

---

## Task 3: Legacy-collection adoption migration

**Files:**
- Create: `tldw_chatbook/RAG_Search/simplified/collection_indexes.py`
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_factory.py`
- Test: `Tests/RAG/simplified/test_collection_indexes.py`

**Interfaces:**
- Consumes: `fingerprinted_collection_name`, `collection_provenance` (Task 1); `RAGConfig`.
- Produces:
  - `adopt_legacy_collection(persist_directory, legacy_name, target_name, provenance) -> bool`
  - `maybe_adopt_legacy_collection(config: RAGConfig) -> None` — guarded no-op unless persistent Chroma + legacy `default` present + target absent.

- [ ] **Step 1: Write the failing test**

```python
# Tests/RAG/simplified/test_collection_indexes.py
import pytest

from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    fingerprinted_collection_name,
)
from tldw_chatbook.RAG_Search.simplified.collection_indexes import (
    adopt_legacy_collection,
    maybe_adopt_legacy_collection,
)


def _cfg(persist_dir):
    return RAGConfig(
        embedding=EmbeddingConfig(model="mock", device="cpu"),
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=100),
        vector_store=VectorStoreConfig(
            type="chroma", persist_directory=persist_dir,
            collection_name="default", distance_metric="cosine"),
    )


def _seed_legacy(persist_dir, name="default", n=3):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(persist_dir),
                                       settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"id{i}" for i in range(n)],
            embeddings=[[float(i)] * 8 for i in range(n)],
            documents=[f"doc {i}" for i in range(n)])
    return client


@pytest.mark.requires_chromadb
def test_adopt_moves_docs_and_removes_legacy(chroma_persist_dir):
    _seed_legacy(chroma_persist_dir)
    cfg = _cfg(chroma_persist_dir)
    target = fingerprinted_collection_name(cfg)
    maybe_adopt_legacy_collection(cfg)

    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False))
    names = [c.name for c in client.list_collections()]
    assert target in names and "default" not in names
    assert client.get_collection(target).count() == 3
    assert client.get_collection(target).metadata.get("source") == "legacy-adopted"
    assert client.get_collection(target).metadata.get("verified") is False


@pytest.mark.requires_chromadb
def test_adopt_is_idempotent(chroma_persist_dir):
    _seed_legacy(chroma_persist_dir)
    cfg = _cfg(chroma_persist_dir)
    maybe_adopt_legacy_collection(cfg)
    maybe_adopt_legacy_collection(cfg)  # second run must not raise or duplicate
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(chroma_persist_dir),
                                       settings=Settings(anonymized_telemetry=False))
    assert client.get_collection(fingerprinted_collection_name(cfg)).count() == 3


@pytest.mark.requires_chromadb
def test_no_legacy_is_noop(chroma_persist_dir):
    cfg = _cfg(chroma_persist_dir)
    maybe_adopt_legacy_collection(cfg)  # nothing to adopt
    assert adopt_legacy_collection(
        chroma_persist_dir, "default", fingerprinted_collection_name(cfg), {}
    ) is False


def test_memory_type_is_noop():
    cfg = RAGConfig(vector_store=VectorStoreConfig(type="memory", collection_name="default"))
    maybe_adopt_legacy_collection(cfg)  # must not touch disk / raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/simplified/test_collection_indexes.py -q`
Expected: FAIL — `ModuleNotFoundError: ...collection_indexes`.

- [ ] **Step 3: Write minimal implementation**

```python
# tldw_chatbook/RAG_Search/simplified/collection_indexes.py
"""Chroma-only collection lifecycle: legacy adoption migration + index admin.

Persistent-backend only. See
Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .config import RAGConfig
from .collection_fingerprint import fingerprinted_collection_name, collection_provenance

_LEGACY_BASE = "default"  # the pre-fingerprint canonical collection name


def _client(persist_directory) -> Any:
    import chromadb
    from chromadb.config import Settings
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
    """
    try:
        client = _client(persist_directory)
        existing = {c.name for c in client.list_collections()}
        if target_name in existing or legacy_name not in existing:
            return False
        col = client.get_collection(legacy_name)
        col.modify(name=target_name, metadata={"hnsw:space": provenance.get(
            "distance_metric", "cosine"), **provenance})
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
```

Wire it into `tldw_chatbook/RAG_Search/simplified/rag_factory.py`. Add the import:
```python
from .collection_indexes import maybe_adopt_legacy_collection
```
In `create_rag_service`, right after the `if config: rag_config = config` block and before the `return EnhancedRAGServiceV2(...)`:
```python
    # Adopt a pre-fingerprint 'default' collection under this config's
    # fingerprint on first persistent construction (idempotent, race-safe).
    try:
        maybe_adopt_legacy_collection(rag_config)
    except Exception as e:  # never block service creation on migration
        logger.debug(f"Legacy collection adoption skipped: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/simplified/test_collection_indexes.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/collection_indexes.py \
        tldw_chatbook/RAG_Search/simplified/rag_factory.py \
        Tests/RAG/simplified/test_collection_indexes.py
git commit -m "feat(rag): provenance-aware legacy collection adoption migration"
```

---

## Task 4: Index admin API — list / delete / status

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/collection_indexes.py`
- Test: `Tests/RAG/simplified/test_collection_indexes.py` (append)

**Interfaces:**
- Produces:
  - `list_indexes(persist_directory) -> list[dict]` — one entry per collection: `{"name", "fp", "provenance", "count"}`.
  - `delete_index(persist_directory, name) -> bool`.
  - `index_status(config: RAGConfig) -> dict` — `{"state": "built"|"empty"|"absent", "count", "provenance"}` for the config's resolved collection.

- [ ] **Step 1: Write the failing test (append)**

```python
@pytest.mark.requires_chromadb
def test_list_and_delete_indexes(chroma_persist_dir):
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
    from tldw_chatbook.RAG_Search.simplified.collection_indexes import (
        list_indexes, delete_index, index_status,
    )
    a = _cfg(chroma_persist_dir)
    b = _cfg(chroma_persist_dir); b.chunking.chunk_size = 512  # different fp
    RAGService(a).vector_store.collection            # force-create collection a
    RAGService(b).vector_store.collection            # force-create collection b

    idx = list_indexes(chroma_persist_dir)
    names = {i["name"] for i in idx}
    assert fingerprinted_collection_name(a) in names
    assert fingerprinted_collection_name(b) in names
    assert all("provenance" in i and "count" in i for i in idx)

    assert delete_index(chroma_persist_dir, fingerprinted_collection_name(b)) is True
    names_after = {i["name"] for i in list_indexes(chroma_persist_dir)}
    assert fingerprinted_collection_name(b) not in names_after
    assert delete_index(chroma_persist_dir, "does-not-exist") is False


@pytest.mark.requires_chromadb
def test_index_status_absent_then_built(chroma_persist_dir):
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService
    from tldw_chatbook.RAG_Search.simplified.collection_indexes import index_status
    cfg = _cfg(chroma_persist_dir)
    assert index_status(cfg)["state"] == "absent"
    RAGService(cfg).vector_store.collection           # create, still empty
    assert index_status(cfg)["state"] == "empty"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Tests/RAG/simplified/test_collection_indexes.py -q`
Expected: FAIL — `ImportError: cannot import name 'list_indexes'`.

- [ ] **Step 3: Implement (append to `collection_indexes.py`)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest Tests/RAG/simplified/test_collection_indexes.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/RAG_Search/simplified/collection_indexes.py \
        Tests/RAG/simplified/test_collection_indexes.py
git commit -m "feat(rag): index admin API (list/delete/status) over fingerprinted collections"
```

---

## Task 5: Empty-until-backfill honesty (reuse existing probe) + parity

**Files:**
- Test: `Tests/RAG/simplified/test_index_isolation_integration.py` (append)

**Interfaces:**
- Consumes: `semantic_index_is_empty` from `tldw_chatbook.RAG_Search.semantic_availability`; `RAGService`; Task 1/4 helpers. No new production code — this task locks the contract that a fresh fingerprinted collection auto-triggers the existing honest empty state, and that ingest and query resolve the same collection.

- [ ] **Step 1: Write the failing test (append)**

```python
import asyncio

@pytest.mark.requires_chromadb
def test_fresh_fingerprinted_collection_reads_as_empty(chroma_persist_dir):
    from tldw_chatbook.RAG_Search.semantic_availability import semantic_index_is_empty
    cfg = _chroma_cfg(chroma_persist_dir)
    svc = RAGService(cfg)
    svc.vector_store.collection  # create, no docs
    assert asyncio.get_event_loop().run_until_complete(
        semantic_index_is_empty(svc)) is True


@pytest.mark.requires_chromadb
def test_ingest_and_query_resolve_same_collection(chroma_persist_dir):
    # Two services built from the SAME config must open the SAME collection —
    # the embed/query parity guard against divergent-model index misses.
    cfg = _chroma_cfg(chroma_persist_dir)
    ingest_side = RAGService(cfg).vector_store.collection_name
    query_side = RAGService(cfg).vector_store.collection_name
    assert ingest_side == query_side == fingerprinted_collection_name(cfg)
```

- [ ] **Step 2: Run test to verify it fails (or passes green — confirm the contract)**

Run: `pytest Tests/RAG/simplified/test_index_isolation_integration.py -q -k "empty or same_collection"`
Expected: PASS if Task 2 wired correctly (this is a contract lock, not new behavior). If `test_fresh_fingerprinted_collection_reads_as_empty` FAILS, `get_collection_stats()` is not returning a clean integer `0` for a fresh collection — investigate the store's stats seam before proceeding (do NOT weaken the probe).

- [ ] **Step 3: Minimal implementation**

No production change expected. If Step 2 revealed the stats seam returns a non-integer count for an empty collection, fix `ChromaVectorStore.get_collection_stats` to return an integer `count` for the resolved collection (the honest-state probe rejects `0.0`/`"0"`).

- [ ] **Step 4: Run full RAG suite (regression gate)**

Run: `pytest Tests/RAG/ -q`
Expected: PASS at or above the recorded baseline (note any pre-existing failures unrelated to this change).

- [ ] **Step 5: Commit**

```bash
git add Tests/RAG/simplified/test_index_isolation_integration.py \
        tldw_chatbook/RAG_Search/simplified/vector_store.py
git commit -m "test(rag): lock empty-index honesty + ingest/query collection parity"
```

---

## Task 6: Docs — config example + fingerprint contract note

**Files:**
- Modify: `tldw_chatbook/RAG_Search/rag_config_example.toml` (comment only)
- Modify: `tldw_chatbook/RAG_Search/README_enhanced_services.md` (short section)

**Interfaces:** none (documentation).

- [ ] **Step 1: Add a note to `rag_config_example.toml`**

Under the `[rag.embeddings]` / `[rag.chunking]` sections, add a comment:
```toml
# NOTE: embedding model, max_length, every chunking setting, and the vector
# store distance_metric are "index-determining": changing any of them makes
# the RAG engine use a SEPARATE vector collection (fingerprinted), which is
# empty until re-indexed. Query-time settings (top_k, hybrid_alpha,
# reranking, citations) share one index. See
# Docs/superpowers/specs/2026-07-21-rag-index-isolation-design.md
```

- [ ] **Step 2: Add a "Collection fingerprinting" subsection to `README_enhanced_services.md`**

```markdown
### Collection fingerprinting (index isolation)

Each vector collection is named `<base>__<fingerprint>` where the fingerprint
is a versioned hash of the index-determining config (embedding model +
max_length, all chunking fields, distance metric). Ingestion and search
resolve to the same fingerprinted collection, so changing an index-determining
setting points at a fresh (empty-until-backfilled) collection rather than
mixing incompatible vectors. Admin helpers: `collection_indexes.list_indexes`,
`delete_index`, `index_status`. Legacy pre-fingerprint `default` collections
are adopted on first run (`maybe_adopt_legacy_collection`).
```

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/RAG_Search/rag_config_example.toml \
        tldw_chatbook/RAG_Search/README_enhanced_services.md
git commit -m "docs(rag): document collection fingerprinting / index isolation"
```

---

## Self-Review

**1. Spec coverage** (`2026-07-21-rag-index-isolation-design.md`):
- §2 fingerprint (index-determining set incl. distance_metric; excludes query-time; normalized; versioned; valid Chroma name) → Task 1. ✓
- §3 single resolution seam (ingest/backfill/search share it) → Task 2 (`rag_service.py:139`) + Task 5 parity test. ✓
- §4 legacy migration (adopt under active-time config, provenance `legacy/unverified`, idempotent, race-safe, Chroma-only) → Task 3. ✓
- §5 provenance stamped at creation + honest empty state → Task 2 (stamp) + Task 5 (empty reuse). ✓
- §6 proliferation cleanup (list/delete) → Task 4. ✓
- §7 testing (determinism, normalization, query-vs-index diffs, migration idempotency/race, parity, empty honesty) → Tasks 1–5. ✓
- §8 plan-time verifications: field set enumerated (Global Constraints); single seam confirmed (`RAGService.__init__:139`, inherited); backfill resolves via shared service (unchanged — same seam). Rename-vs-alias: uses Chroma `collection.modify(name=...)`; if unavailable at execution, fall back to copy-into-new + delete-legacy in `adopt_legacy_collection` (same signature). ✓

**2. Placeholder scan:** no TBD/TODO; every code step shows full code; every test shows assertions. ✓

**3. Type consistency:** `fingerprint_collection`/`fingerprinted_collection_name`/`collection_provenance` signatures identical across Tasks 1–5; `collection_metadata` param name consistent across store constructors and `create_vector_store`; `maybe_adopt_legacy_collection(config)` / `adopt_legacy_collection(persist_directory, legacy_name, target_name, provenance)` consistent between Task 3 def and Task 3 wiring. ✓

**Cross-SP note (not a gap):** the migration-invariant end-to-end test (adopted legacy fp == imported-profile fp) belongs to SP2, which introduces the import; SP1 documents the contract in Global Constraints and stamps `verified=False` so SP2/SP3 can surface unverified legacy indexes.
