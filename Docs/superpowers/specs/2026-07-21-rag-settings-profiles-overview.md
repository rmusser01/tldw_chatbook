# RAG Settings Screen + Profiles — Program Overview

**Date:** 2026-07-21 · **Status:** Draft (owner review pending)
**Owner ask:** a dedicated RAG settings screen in the Settings area covering the full RAG engine config, plus named **profiles** — saved collections of RAG settings the user can swap between without manual re-tweaking.

## 1. Decisions (owner-confirmed)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Config scope | **Full RAG engine config** — everything in `ProfileConfig` (`RAGConfig`: search, embeddings, chunking, vector store, retriever; plus reranking + processing). Absorbs and replaces the narrow "Library RAG" settings category. |
| D2 | Active-profile model | **Profile IS the config.** The active profile's values ARE what the RAG engine reads at runtime. One active profile at a time; switching re-points the engine; the screen edits the active profile directly; "Save as new profile" clones. |
| D3 | Builtins | **Read-only seeds.** The existing ~13 code-defined builtins (`hybrid_basic`, `high_accuracy`, `balanced`, `code_search`, …) are immutable; the user selects them active or clones them to edit. Users cannot corrupt baselines. |
| D4 | Storage | **One file per profile** (JSON) under `rag_profiles/` in the user data dir. The single active-profile name lives in `config.toml` at the existing `[rag.service].profile` pointer. NOT a growing list in config.toml (rail_state precedent). |
| D5 | Index handling on config change | **Separate index per embedding config.** Vector collections carry an embedding-config fingerprint; switching to a config whose collection is absent yields an empty collection (built lazily by backfill), surfaced with honest empty-index states. |
| D6 | Sequencing | **Design all three now, build in order.** One overview spec + three sub-specs; foundation → profile system → screen, each an independent user-gated PR. |

## 2. Sub-projects and build order

The program decomposes into three sub-projects built in dependency order. Each gets its own spec, plan, and user-gated PR.

1. **SP1 — Per-embedding-config index isolation** (`2026-07-21-rag-index-isolation-design.md`).
   Vector collection names carry a deterministic fingerprint of the *index-determining* config (embedding model + all chunk-output-affecting fields). Ingestion, backfill, and search resolve to the collection matching the active config, so embedding/chunk changes are safe rather than silently corrupting a shared index. Ships a provenance-aware legacy-collection migration so existing users keep their index on upgrade.

2. **SP2 — Profile system** (`2026-07-21-rag-profile-system-design.md`).
   Extends the existing `ConfigProfileManager` (already wired into `create_rag_service`) with file-per-profile storage, read-only builtins, a single active-profile pointer, and — the load-bearing correctness work — **unification of config resolution onto the active profile** so every runtime reader (structured service path AND the `config.py` loader / flat readers) resolves from one source. Includes a first-run import of the user's existing config into an "Imported settings" profile.

3. **SP3 — RAG settings screen** (`2026-07-21-rag-settings-screen-design.md`).
   A new "RAG" settings category (repurposing the `LIBRARY_RAG` category) with two regions: a profile manager (list / set-active / clone / rename / delete) and a full-`ProfileConfig` editor (collapsible groups) that edits the active profile. Saves write the profile file (not the deprecated config keys) and trigger SP2's project + service reset.

**Why this order:** SP1 makes embedding/chunk fields *safe* to vary, which is the precondition for profiles that change them (SP2), which is the precondition for a screen that edits them (SP3). SP2 is the hardest sub-project — the correctness lives there — while SP3 is comparatively mechanical.

## 3. Load-bearing seams (shared across sub-projects)

- **Single active-profile pointer.** `[rag.service].profile` (already read by ingestion `ingestion_indexing.py:128` and the factory) is THE pointer; no second key is introduced. If ingestion and search resolved different pointers they would fingerprint to different collections and query vectors would miss the index — a correctness requirement, not tidiness.
- **One collection, fingerprinted.** The store uses a single `collection_name` (`rag_service.py:142`, default `"default"`); content types are separated by `source_type` metadata (the same field the RAG-scope program filters on). SP1 fingerprints that one name (`default` → `default__<fp>`); the vestigial per-type `*_collection` fields in `VectorStoreConfig` are not used.
- **Config resolution unification.** Two config paths exist today: the structured service path (`create_rag_service` → `get_profile_manager().get_profile()` → `profile.rag_config`, already profile-driven) and the `config.py` loader path (`config.py:457–524`, re-derives a `RAGConfig` from `AppRAGSearchConfig.rag.*` + `embedding_config` + env, with a priority chain). SP2 makes both resolve from the active profile; the flat `AppRAGSearchConfig.rag.*` *value* keys are deprecated.

## 4. Cross-SP invariants (tested end-to-end)

1. **Fingerprint agreement on upgrade.** SP1's legacy-collection adoption and SP2's "Imported settings" first-run profile MUST resolve to the same fingerprint, so an upgraded user's existing index remains owned by the active profile — no surprise empty index, no forced backfill. This is the single most important upgrade-path guarantee and is tested end-to-end (upgrade → existing index still queried by the active profile).
2. **Embed-config == query-config for the active profile.** For any active profile, the config used to embed at ingestion equals the config used to query at search time. This is the guard against divergent-model index misses / dimension-mismatch crashes, mirroring the RAG-scope backend-parity contract test.
3. **No silent index blanking.** Shipping SP1 must never leave an existing user's index unreadable without a surfaced, actionable "empty / needs backfill" state.

## 5. Out of scope (follow-up candidates)

- Full multi-index management UI (list/delete every on-disk fingerprinted collection with sizes) — SP3 ships only the active profile's index status + backfill; the `list_collections`/`delete_collection` seams exist (`vector_store.py:632/765`) for a later follow-up.
- Per-conversation / per-workspace profile overrides (profiles are global in v1).
- Profile import/export/sharing between instances.
- Dynamic profile auto-selection by query type (`create_rag_service_from_config` already has heuristic auto-detection; not surfaced to users here).

## 6. Delivery

Three phased PRs (SP1 → SP2 → SP3). Every PR: independent merge-gate review, live QA captures where UI is involved, and **user-approved before merge** — subagents never merge (2026-07-21 incident rule). Backlog task IDs assigned at branch time past all open-branch claims and re-verified at merge (collision history). No DB schema migration (profiles are files; the pointer is config.toml).
