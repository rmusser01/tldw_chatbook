# RAG Scope Narrowing — Design

**Date:** 2026-07-21 · **Status:** Approved in brainstorm; pending spec review
**Owner ask:** a button + modal to narrow RAG search to selected media/notes for the current conversation (sortable, keyword/tag filterable), and the same mechanism at workspace level ("sales reports" / "hunt X" workspaces whose in-scope set bounds all retrieval inside them).

## 1. Decisions (owner-confirmed)

| # | Decision | Choice |
|---|---|---|
| D1 | Scope semantics | **Hard filter on all retrieval** (semantic + FTS legs + hybrid). Staged sources remain a separate attach mechanism; no interaction — staging an out-of-scope item is allowed. |
| D2 | v1 reach | **Retrieval only**: RAG answers, Run Library RAG, hybrid/chat RAG. Library browsing stays unfiltered (browse-filtering = possible v2). |
| D3 | Layering | **Conversation narrows within workspace** (intersection). Conversation picker inside a scoped workspace offers only the workspace's items. |
| D4 | Scope content | **Static item lists**. Tag/keyword filtering is picker convenience; no dynamic rules in v1 (v2 candidate). |
| D5 | Conversations FTS leg | **Excluded while scope active** in v1; `ScopeItem` vocabulary admits `source_type="conversation"` for future scoping-in. Exclusion is diagnosed, not silent. |
| D6 | Entry points | **Inspector "Retrieval scope" row** (below the Sources tray) + **effective-count header chip**; workspace scope from the Session/workspace area. One reusable modal for both targets. |

## 2. Data model & resolution — `Chat/rag_scope.py` (pure logic, no UI imports)

- `ScopeItem = (source_type: "media"|"note", source_id: str)`. `source_id` is **exactly the identifier the RAG index stamps as `source_id`** (task-247 ingestion metadata), string-coerced at the boundary. No mapping layer. (Plan-time: confirm media `id` vs `uuid` from `ingestion_indexing.py`.)
- Stored shape: `{"version": 1, "items": [...], "updated_at": ...}`.
  - Conversation: `conversations.metadata["rag_scope"]` (schema-v20 JSON column; same guarded-read discipline as chat-dictionaries attach: missing/None/malformed → unscoped, never a crash).
  - Workspace: scope record keyed by `workspace_id`, stored **in the DB** alongside/adjacent to the workspace registry — never in config.toml (rail_state growth precedent). Workspace deletion drops the record; resolver tolerates orphans.
- `resolve_effective_scope(conversation_id, workspace_id) -> UNSCOPED | SCOPED(allowlist) | EMPTY(cause)`:
  - Intersection when both levels set. Dangling ids (deleted content) drop lazily at resolution; if that empties the result → `EMPTY` with a named cause (deletions vs no-overlap-with-workspace — the fork-into-workspace path makes no-overlap reachable and it must read as diagnosis).
  - Cached per session, keyed on both levels' `updated_at` stamps — stamp comparison per query, no invalidation fan-out. Concurrent edits: last-write-wins; stamp key makes the loser visible on next query.
- Scope is inert while RAG is off (config-state chip semantics, like other header chips).
- Export/import: scope travels as inert conversation metadata; unknown ids in the target instance degrade via dangling-drop → `EMPTY(cause)`.

## 3. Enforcement — one resolver, two backend adapters, legs self-enforce

Shared `build_scope_filter()` helpers translate one allowlist into each backend's native form.

**Backend A — pipeline legs (chat RAG, all pipeline shapes):** scope enters via `PipelineContext`; each leg self-enforces so builtin, custom-TOML, and retrieve-step pipelines inherit enforcement identically (the task-250 lesson — caller-side leg-skipping leaks through custom pipelines):
- `search_semantic` → `RAGService.search(filter_metadata={"source_id": {"$in": [...]}})` — the parameter and chunk metadata already exist (tasks 246/247); no re-indexing.
- media/notes FTS legs → id-allowlist predicates via `json_each` or chunked binds (SQLite param limits).
- conversations leg → returns `[]` and records `excluded by scope` in `PipelineContext['diagnostics']` (task-250 machinery).
- `EMPTY` → short-circuit before any leg runs; diagnostics carry the cause; hybrid fuses only pre-filtered legs so RRF cannot leak.

**Backend B — `LibraryLocalRagSearchService` (Console "Run Library RAG"):** semantic delegate gets the same `filter_metadata` (free); its keyword seams (`notes_scope_service.search_notes`, `media_reading_scope_service.search_media`) gain id-allowlist parameters. Same resolver, same helpers.

Scoped runs surface in the Inspector run-recipe line (`… / scope 8 items`). Scoped-but-zero-results renders "No results within scope (N items searched)" — distinct from `EMPTY` (task-250 count-marker pattern).

## 4. UI

**`ConsoleScopePickerModal`** (new; `console_prompt_picker_modal` conventions — focus trap, Esc):
- Title names the target: "Narrow RAG scope — this conversation" / "— workspace 'hunt X'".
- Filter row: type tabs (All/Media/Notes) · text filter · **searchable tag selector** (top-N most-used chips + autocomplete; multi-tag OR, AND with text filter) fed by both Keywords tables.
- Sort: Recent / Title / Type. List: paginated checkbox rows with type glyphs, loaded off the UI loop via read-only Library seams. (Plan-time: seams must return the full universe regardless of Library-side workspace visibility filtering — the `scope_type="all"` hide-bug class.)
- **All / Selected view toggle** — selection survives filter changes; opens in *Selected* view when a scope exists, *All* when unscoped.
- **"Select all matching"** operates on the full filtered set via id-only query, with count confirmation. "Clear shown" complements.
- Footer: live "N selected of M" · Save · Clear scope · Cancel. **Save with zero selected = clear scope** (intentionally-empty scopes cannot be created; `EMPTY` only arises from intersection/deletions and is always diagnosed).
- Conversation target inside a scoped workspace: item universe = the workspace's scope list (D3 enforced at pick time).

**Inspector**: compact "Retrieval scope" row directly below the Sources tray — "Scope: everything · Narrow…" unscoped; "Scope: 8 items · Edit · Clear" scoped. A separate row, not a button inside the tray: staged-vs-scope mechanism boundary stays visible.
**Header chip**: effective (post-intersection) count — "Scope: 5"; tooltip "conversation 8 ∩ workspace 40 → 5"; `EMPTY` uses action-required styling. Hidden when unscoped.
**Workspace entry**: Scope button beside the workspace row in the Session area; same modal, workspace target.
Modal save → write storage, bump `updated_at`, refresh row/chip/run-recipe.

## 5. Testing

- **Unit**: resolver state machine (all three states, intersection, stamp caching, dangling-drop, orphan tolerance); `build_scope_filter` per backend; boundary coercion.
- **Contract**: identical scoped search against **both** `ChromaVectorStore` and `InMemoryVectorStore` — parity or loud failure (the `$in`-support divergence risk), never silent.
- **Integration**: index→scope→retrieve returns only in-scope on both backends; conversations-leg exclusion diagnosed; `EMPTY` short-circuit; custom-TOML pipeline inherits enforcement (self-enforcement proof); **one end-to-end chat-send test** asserting injected context is scope-only; **large-scope test** (~1k items) exercising `json_each` chunking and `$in` behavior.
- **UI**: modal select/filter/sort, select-all-matching, Selected view default, zero-selection save clears; Inspector row and chip states incl. `EMPTY`.
- Real in-memory SQLite + mock embeddings per repo convention; ~33-failure UI baseline discipline; QA captures via the textual-serve rig (modal, Selected view, chip states) for owner screen approval.

## 6. Plan-time verifications (before implementation)

1. `filter_metadata` is actually applied in every `EnhancedRAGServiceV2` search path (incl. parallel/rerank branches) — "plumbing exists" ≠ "plumbing works" (`RAGConfig.validate()` zero-callers lesson).
2. The concrete `source_id` identity stamped by `ingestion_indexing.py` (media `id` vs `uuid`), and that re-ingest/overwrite in `add_media_with_keywords` preserves it (a URL re-ingest minting a new row would orphan scopes).
3. Chroma `$in` practical size limits at the assumed 1–2k ceiling.
4. Native-Console send path routes RAG context through `get_rag_context_for_chat` (enforcement point coverage).
5. Workspace registry storage location → exact placement of the workspace scope record.
6. Library seam queries used by the picker ignore Library-side visibility filters.

## 7. Out of scope (v2 candidates)

Dynamic tag rules (D4), scoping-in conversations (D5), Library browse filtering / workspace materials shelf (D2), per-message scope overrides.

## 8. Delivery

Phased PRs (plan to define; likely: core module + enforcement → modal + Inspector/chip → workspace level). Every PR: user-approved before merge after QA captures; **subagents never merge** (2026-07-21 incident rule). Backlog task IDs assigned at branch time past all open-branch claims, re-verified at merge (collision history).
