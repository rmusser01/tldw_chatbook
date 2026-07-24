# SP2 — Profile System — Design

**Date:** 2026-07-21 · **Status:** Draft (owner review pending)
**Part of:** `2026-07-21-rag-settings-profiles-overview.md` (sub-project 2 of 3)
**Goal:** turn the existing code-only `ConfigProfileManager` scaffold into a real, file-backed, user-facing profile system with one active profile that IS the engine config — and unify config resolution so every runtime reader agrees on that profile. This is the hardest sub-project; the program's correctness lives here.

## 1. The profile unit

A profile is a `ProfileConfig` (`config_profiles.py:73`), which already wraps `rag_config: RAGConfig` **plus** `reranking_config` and `processing_config` and metadata. This is already the shape `create_rag_service` consumes, so "full engine config" maps to `ProfileConfig` — no new type. The reranking/processing configs are genuinely consumed (`enhanced_rag_service_v2.py:73–74, 96–98`), so they belong in the profile, not as decoration.

## 2. Storage: file-per-profile

- The manager already uses `get_user_data_dir()/"rag_profiles"` (`config_profiles.py:158`) — the chosen dir — but today writes *all* custom profiles into a single `custom_profiles.json` blob (`:451/:535`). SP2 switches to **one JSON file per user profile**, reusing the existing `ProfileConfig.to_dict`/`from_dict`.
- **Builtins stay code-defined seeds**, never written to disk.
- **Migration:** on first load, split any legacy `custom_profiles.json` blob into per-file profiles (idempotent; tolerate concurrent first-runs; leave the blob in place or archive it, never lose a profile).
- **Identity vs. rename.** Each file stores the profile's display name; the filename is a slug; the active pointer stores a **stable id** (not the display name), so renaming rewrites the file's name field without breaking the pointer or orphaning the file. Slug collisions (`"My Config"` vs `"my config"`) are disambiguated by the stable id.

## 3. Read-only builtins

There are ~13 builtins (`bm25_only`, `vector_only`, `hybrid_basic`, `hybrid_enhanced`, `hybrid_full`, `fast_search`, `high_accuracy`, `balanced`, `long_context`, `technical_docs`, `research_papers`, `code_search`), not four. SP2 marks all immutable: `save`/`delete`/`rename` on a builtin is refused at the manager layer. "Clone to edit" writes a new user-profile file seeded from the builtin's `ProfileConfig`. (Presenting 13 read-only seeds is an SP3 UX concern — grouping/curation.)

## 4. Single active pointer

Reuse `[rag.service].profile` (already read by ingestion `ingestion_indexing.py:128` and defaulted by the factory). Extend it to name user profiles as well as builtins. **No second pointer key** — a separate `active_profile` would let ingestion and search resolve different profiles → different embedding models → index miss / dimension crash (overview §3).

## 5. Config resolution unification (the load-bearing work)

Two config paths exist today and read different sources:

- **Path A — structured service** (`create_rag_service` → `get_profile_manager().get_profile()` → `profile.rag_config`/`reranking_config`, `rag_factory.py:33–34`): reads the profile object directly. Ingestion, backfill, and service-path search already flow through here — profile-driven.
- **Path B — `config.py` loader** (`config.py:457–524`): re-derives a `RAGConfig` from `AppRAGSearchConfig.rag.*` + `embedding_config.default_model` + env, with a priority chain `override > env > rag.embedding > embedding_config default > class default` (`:473`). The flat readers (`fusion.py` `resolve_hybrid_alpha`, `chat_rag_events`) also read `AppRAGSearchConfig.rag.*`.

**Projecting the profile into `AppRAGSearchConfig` is unsound** against Path B's priority chain: a projected model value can be overridden by a higher-priority env var or `embedding_config` default and silently lose, so Path A (ingestion) embeds with model X while Path B (some search) queries with model Y.

**Resolution:** introduce one `resolve_active_rag_config() -> ProfileConfig` that reads the pointer, loads the profile, and is consumed by **both** paths:

- Path A already calls `get_profile(active)`; it routes through the new resolver (no behavior change).
- Path B's loader sources the active profile's `rag_config` as its **base**, keeping env only as a deliberate explicit-override layer on top. The scattered `AppRAGSearchConfig.rag.*` **value** keys are deprecated (the narrow screen that wrote them is absorbed by SP3).
- A plan-time map enumerates every reader of `AppRAGSearchConfig.rag.*` / `embedding_config` / top-level `[rag.*]` and points it at the resolver or documents why env-override is intentional there.

**Safety net — parity test:** for a given active profile, the config used to embed (ingestion path) == the config used to query (search path). This is the guard against the divergent-model crash and mirrors the RAG-scope backend-parity contract.

## 6. First-run import (no lost hand-tuning)

Unifying onto the profile means an upgrader's existing config stops being an ad-hoc scattered source. On first run after SP2b ships, snapshot the **currently-resolved active config** (`resolve_active_rag_config()` — the active profile, built-in `hybrid_basic` on a true first run, plus any `RAG_*` env overrides) into an **"Imported settings" user profile** and set it active.

**Correction (SP2b implementation, 2026-07-23):** the snapshot captures the **active-profile resolution**, NOT the legacy `AppRAGSearchConfig.rag.*` / `embedding_config` values (those keys are deprecated by SP2b). This is deliberate and load-bearing for the index invariant: **pre-SP2b ingestion built the vector collection from the built-in profile** (`create_rag_service(profile_name=…)`, no config), so SP1's `maybe_adopt_legacy_collection` adopted the legacy `default` collection under the *built-in* fingerprint — which the built-in-based snapshot matches. Capturing the old `from_settings` default (`mxbai-embed-large-v1`) as the earlier draft mandated would instead have *broken* the fingerprint invariant for default users. So the imported profile's fingerprint **equals** SP1's adopted legacy-collection fingerprint (overview §4.1) by resolving through the same `resolve_active_rag_config()` both use — verified e2e (incl. an env-divergence case).

**Known limitations (follow-ups, not silent):** (a) hand-tuned *query-time* legacy keys (`default_top_k`, `score_threshold`, `include_citations`, reranking) are not merged into the snapshot — a non-fingerprint-affecting enrichment deferred to a follow-up. (b) A user who set a *fingerprint-affecting* env var (`RAG_EMBEDDING_MODEL`/`RAG_CHUNK_SIZE`/`RAG_CHUNK_OVERLAP`) was already in a pre-SP2b divergent state (ingestion used the built-in model, search applied env); post-SP2b both apply env consistently, so their built-in-embedded legacy collection becomes stale and SP1's honest empty-index state prompts a re-index — expected, not silent blanking. The user-facing profile description is worded to not overclaim.

## 7. Switch mechanics (set-active)

Set-active is a deliberate action, not a live keystroke:

1. Write the pointer (`[rag.service].profile`).
2. Recompute the SP1 fingerprint; if it differs from the current index's, mark the target index built / empty-needs-backfill for SP3 to surface.
3. **Reset the shared RAG service singleton** so the next resolution rebuilds on the new config and SP1's fingerprinted collection. This reset seam **does not exist today** — `get_shared_rag_service` caches `_shared_service` under `_shared_service_lock` (`ingestion_indexing.py:121–122`) with no way to clear it. SP2 adds `reset_shared_rag_service()` (clear the global under the lock; the next call rebuilds). Set-active cannot work without it.
4. The embedding-model reload runs **off-thread** (a model load is expensive); the singleton reset must not yank the service out from under an in-flight ingestion/search worker (running ops keep their reference; new ops get the new service).

Deleting the active profile falls back to a builtin default (and offers to delete its fingerprinted index via SP1's `delete_index`). Note `get_profile_manager()` currently re-instantiates per call (`:769`, not a singleton) — acceptable for stateless reads since active state lives in config, not memory; SP2 need not change that, but must not assume in-memory caching of the active profile.

## 8. Testing

- File round-trip (`to_dict`/`from_dict`); builtin immutability (save/delete/rename refused); clone → editable copy seeded from builtin; stable-id survives rename; slug-collision disambiguation.
- Legacy-blob → per-file migration idempotent and concurrency-safe.
- `resolve_active_rag_config` consumed by both paths; **parity test** (embed-config == query-config for the active profile); env-override layer still wins where documented.
- Set-active: pointer written, service reset, fingerprint-change detected, off-thread reload does not block, in-flight worker unaffected; delete-active fallback.
- First-run import: existing config captured into a profile, set active, fingerprint == SP1 adopted legacy fingerprint (the upgrade invariant).
- **Singleton reset must be test-isolated.** `reset_shared_rag_service()` mutates a module global, and there is a documented cross-file RAG cache-singleton test-pollution issue (backlog task-408). Every test that resets or builds the shared service resets it again in teardown (fixture), or the pollution leaks into unrelated suites.
- Real in-memory SQLite + mock embeddings; temp profiles dir.

## 9. Plan-time verifications

1. The complete list of `AppRAGSearchConfig.rag.*` / `embedding_config` / `[rag.*]` readers (the resolution map) — "plumbing exists ≠ works."
2. That routing Path B through the resolver preserves intended env-override behavior (no regression for users who set env vars).
3. That `create_rag_service`'s fallback-when-profile-missing path (`rag_factory.py:38`) still behaves when the pointer names a deleted user profile.
4. No DB schema migration is required (confirm profiles + pointer are the only persistence).
