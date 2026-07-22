# SP1 — Per-Embedding-Config Index Isolation (Foundation) — Design

**Date:** 2026-07-21 · **Status:** Draft (owner review pending)
**Part of:** `2026-07-21-rag-settings-profiles-overview.md` (sub-project 1 of 3)
**Goal:** make vector collections keyed by the config that built them, so that changing embedding model or chunking (as profiles will) points at a distinct index rather than silently corrupting a shared one. Prerequisite for SP2/SP3.

## 1. Problem

Today the persistent store uses a single collection name (`self.config.collection_name`, default `"default"`, `rag_service.py:142`); content types are separated by `source_type` metadata, not by collection. If a profile changes the embedding model or chunk parameters, new vectors land in the same collection alongside incompatible old ones. The store raises on embedding-dimension mismatch (`vector_store.py:352`) and Chroma raises on query mismatch — so the failure is loud (a crash) rather than silent corruption, but it is still a failure. Chunk-size changes are worse: same dimensionality, so no crash, just a collection of mixed-granularity chunks that silently degrade retrieval.

## 2. The fingerprint

A single pure function, `fingerprint_collection(config) -> str`, hashing only **index-determining** fields available without loading a model:

- **Embedding identity:** `embedding.model` (its dimension is derived only after load — `embeddings_wrapper.py:253` caches `factory.dimension` on first use — so the model *name* is the proxy; two models that happen to share a dimension still get separate collections, which is correct). Any additional embedding field that alters the produced vector for a given model (e.g. an instruction/query prefix, normalization) is included; `batch_size` is **excluded** (throughput, not output).
- **Chunking identity:** *every* `ChunkingConfig` field that alters chunk output — `chunk_size`, `chunk_overlap`, `chunking_method`, `min_chunk_size`, `max_chunk_size`, `parent_size_multiplier`, and structural-chunking settings — enumerated explicitly from the dataclass at plan time. Cherry-picking a subset creates the surprise "I changed max_chunk_size and my index didn't rebuild."
- **Excluded (query-time, must NOT fork the index):** `top_k`, `hybrid_alpha`, `score_threshold`, reranking config, citation settings. Two profiles differing only in `top_k` share one index; two differing in `chunk_size` get separate indexes.

**Output:** `f"{base}__{short_hash}"`, e.g. `default__a3f1c9`. Constraints the function guarantees: a valid Chroma collection name (3–63 chars, charset `[a-zA-Z0-9._-]`, no consecutive dots, not an IPv4), and **input normalization before hashing** — TOML `"400"` (str) and int `400` must fingerprint identically (the same boundary-coercion discipline as the RAG-scope program).

**The fingerprint is a persistent contract.** Once shipped, changing its input set or hash algorithm re-points every collection at once (all read empty). Therefore: it lives in exactly one function, its field list is **versioned**, and any future change to its inputs is treated as a schema migration (ship an old→new mapping), never a silent tweak.

## 3. Resolution seam

Fingerprinting is applied at the single seam where `collection_name` is finalized — inside the config resolution that feeds the shared RAG service. Because ingestion, `backfill_semantic_index` (`ingestion_indexing.py:958`), and service-path search all resolve through `get_shared_rag_service()`, putting the fingerprint there makes **all three target the same collection automatically**. This is exactly why the overview's single-pointer seam is a correctness prerequisite: if ingestion and search resolved different active-profile pointers, they would fingerprint differently and queries would miss the index.

Scope: fingerprinting applies uniformly, but the migration and "empty until backfill" logic below are **persistent-backend (Chroma) only** — the in-memory store rebuilds per session, so there is nothing to migrate or preserve there.

## 4. Legacy-collection migration (must ship in SP1)

Existing users hold vectors in the un-suffixed `default` collection. The moment SP1 ships, the active config resolves to `default__<fp>` = empty, silently blanking every existing index. To prevent this:

- On first resolution against the persistent store, if the fingerprinted collection is absent but the legacy `default` collection exists, **adopt the legacy collection in place** by renaming it to the fingerprinted name.
- **Adopt it under the fingerprint of the profile active at migration time** — NOT the shipping default config. The legacy collection was built with whatever profile the user last had active (they may have set `[rag.service].profile` to a non-default embedding model); labeling it as the default config's fingerprint would claim model-X while it holds model-Y vectors, producing a dimension-mismatch crash on first query. Stamp its provenance as `legacy/unverified`; the dimension-mismatch raise remains the backstop if the guess is wrong.
- **Idempotent and race-safe.** `PersistentClient` runs against a shared `persist_directory` and the environment has concurrent sessions; two first-runs must both converge without destroying data — create-if-absent semantics, tolerate "already migrated," never delete on a lost race.

**Cross-SP invariant:** this adopted fingerprint MUST equal the fingerprint of SP2's "Imported settings" first-run profile, so the upgraded user's existing index is owned by their active profile (overview §4.1).

## 5. Provenance + honest empty state

- On collection creation, stamp its build config (embedding model, chunk params, fingerprint version) as collection metadata. SP3's screen reads this to show "index built with model X / chunk 400·100."
- Switching to a profile whose fingerprinted collection does not exist yields an **empty** collection — no auto-reindex. Retrieval surfaces this through the honest semantic-empty states the RAG-audit program already built ("index empty for this config; run backfill"), distinct from "0 results found."
- **Re-embedding cost is surfaced, never silent** (no-silent-caps rule): switching to a new embedding config means re-embedding the whole corpus (potentially long); the empty state and SP3's backfill action must communicate this.

## 6. Index proliferation + cleanup

Every distinct (model, chunking) combination leaves a full persistent copy of the corpus embedded — unbounded disk if unmanaged. The cleanup seams already exist: `delete_collection(name)` and `list_collections()` (`vector_store.py:632/765`). SP1 exposes a small internal API over them (`list_indexes()` returning fingerprint + provenance + count, `delete_index(fingerprint)`); SP2 calls `delete_index` when a profile is deleted (with confirmation), and SP3's follow-up can present the full list. The accumulated-index cost is thus visible and manageable, not silent.

## 7. Testing

- **Fingerprint:** determinism; input normalization (str vs int coerce identically); query-only diff → same collection; any index-field diff (incl. `max_chunk_size`, `parent_size_multiplier`) → different collection; output is a valid Chroma name at boundary lengths; version bump changes output.
- **Migration:** legacy `default` adopted under the *active* profile's fingerprint; idempotent (second run is a no-op); race-safe (two concurrent first-runs converge, no data loss); provenance stamped `legacy/unverified`.
- **Resolution:** ingestion-then-search hit the same collection for one active profile; backfill targets the resolved collection; empty-state honesty on an unbuilt config; provenance round-trips.
- **Cross-SP:** with SP2 present, the adopted legacy fingerprint == the imported profile's fingerprint (the upgrade invariant); embed-config == query-config for an active profile (parity).
- Real in-memory SQLite + mock embeddings per repo convention; persistent-backend tests use a temp `persist_directory`.

## 8. Plan-time verifications

1. Enumerate the exact `EmbeddingConfig`/`ChunkingConfig` fields that affect produced output (the fingerprint input set) from the real dataclasses.
2. Confirm `collection_name` has no other assignment site than `self.config.collection_name` (grep showed only `rag_service.py:142`).
3. Confirm Chroma's rename/adopt path is available and atomic-enough for the idempotent migration; if rename is unavailable, fall back to a stable alias map.
4. Confirm `backfill_semantic_index` writes through the resolved collection with no independent collection-name derivation.
