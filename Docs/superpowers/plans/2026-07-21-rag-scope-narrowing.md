# RAG Scope Narrowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users hard-limit RAG retrieval to a selected set of media/notes per conversation and per workspace, with a sortable/tag-filterable picker modal, an Inspector row, and an effective-count header chip.

**Architecture:** A pure `Chat/rag_scope.py` resolver (UNSCOPED/SCOPED/EMPTY with workspace∩conversation intersection) feeds two enforcement backends: pipeline legs self-enforce via `PipelineContext`, and `LibraryLocalRagSearchService` takes caller-passed scope. Store-level allowlist filtering is added to both vector stores (verification found `filter_metadata` is an equality post-filter that never reaches the store). Spec: `Docs/superpowers/specs/2026-07-21-rag-scope-narrowing-design.md`.

**Tech Stack:** Python 3.11, Textual, SQLite (ChaChaNotes + Media DBs), ChromaDB/in-memory vector stores, pytest.

## Global Constraints

- Spec decisions D1–D6 are fixed (hard filter; retrieval-only reach; intersection layering; static lists; conversations-leg exclusion diagnosed; Inspector row + chip entry).
- `ScopeItem.source_id` is EXACTLY what the indexer stamps: `str(media_row_id)` for media, `str(note_id)` for notes (verified `ingestion_indexing.py:258,303`); source types match `ITEM_TYPE_MEDIA`/`ITEM_TYPE_NOTE`.
- Conversation identity comes from the active session's `persisted_conversation_id` — never `app.current_chat_*` reactives.
- No DB reads on compose/recompose paths; scope resolution is cached keyed `(conversation_id, workspace_id, conv_updated_at, ws_updated_at)`.
- Scope payloads never live in config.toml.
- Guarded reads everywhere: missing/None/malformed/newer-version scope payloads → UNSCOPED + logged warning, never a crash.
- Worker discipline: no `run_worker(exclusive=True)` without `group=`; sync DB work off the UI loop; exceptions caught in workers.
- Every phase = one PR, user-approved after QA captures; subagents never merge. Backlog task IDs chosen past every open branch's claims, re-verified at merge.
- Tests: real in-memory SQLite via tmp_path, mock embeddings (pattern: `Tests/RAG/test_ingestion_indexing.py`); run with venv `python -m pytest` from worktree root; ~33 pre-existing UI-failure baseline — compare against clean origin/dev when unsure.
- Known limitation (verified): the native Console send path performs no RAG injection today; scope governs the legacy chat send path, the standalone Search window, and Run Library RAG. File a follow-up backlog task, do not fix here.

---

# Phase 1 — Core module + store filtering + pipeline enforcement (PR 1)

### Task 1: `Chat/rag_scope.py` — model + storage codecs

**Files:**
- Create: `tldw_chatbook/Chat/rag_scope.py`
- Test: `Tests/Chat/test_rag_scope.py`

**Interfaces:**
- Produces: `ScopeItem(source_type: str, source_id: str)` (frozen dataclass, `eq=True`); `RagScope(items: tuple[ScopeItem, ...], updated_at: str)`; `SCOPE_VERSION = 1`; `serialize_scope(scope: RagScope) -> dict`; `parse_scope(raw: Any) -> Optional[RagScope]` (None = unscoped for ANY invalid/missing/newer-version input); `SOURCE_TYPE_MEDIA = "media"`, `SOURCE_TYPE_NOTE = "note"`.

- [ ] **Step 1: Write failing tests**

```python
# Tests/Chat/test_rag_scope.py
from tldw_chatbook.Chat.rag_scope import (
    RagScope, ScopeItem, SCOPE_VERSION, parse_scope, serialize_scope,
    SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE,
)

def test_round_trip():
    """serialize→parse preserves items and stamps."""
    scope = RagScope(items=(ScopeItem(SOURCE_TYPE_MEDIA, "42"), ScopeItem(SOURCE_TYPE_NOTE, "n-1")), updated_at="2026-07-21T00:00:00+00:00")
    raw = serialize_scope(scope)
    assert raw["version"] == SCOPE_VERSION
    parsed = parse_scope(raw)
    assert parsed == scope

def test_parse_guards():
    """Missing/None/malformed/newer-version → None (unscoped), never raises."""
    for bad in (None, "", 7, [], {}, {"version": 1}, {"version": 99, "items": []},
                {"version": 1, "items": [{"source_type": "media"}]},
                {"version": 1, "items": "nope", "updated_at": "x"}):
        assert parse_scope(bad) is None

def test_source_id_coerced_to_str():
    """Integer ids from raw payloads coerce to str at the boundary."""
    parsed = parse_scope({"version": 1, "updated_at": "t",
                          "items": [{"source_type": "media", "source_id": 42}]})
    assert parsed.items == (ScopeItem(SOURCE_TYPE_MEDIA, "42"),)

def test_unknown_source_type_dropped():
    """Forward-compat: unknown source types (e.g. 'conversation') are dropped, not fatal."""
    parsed = parse_scope({"version": 1, "updated_at": "t", "items": [
        {"source_type": "conversation", "source_id": "c1"},
        {"source_type": "note", "source_id": "n1"}]})
    assert parsed.items == (ScopeItem(SOURCE_TYPE_NOTE, "n1"),)
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `python -m pytest Tests/Chat/test_rag_scope.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Chat/rag_scope.py
"""Conversation/workspace RAG retrieval scope: model, codecs, resolution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from loguru import logger

logger = logger.bind(module="rag_scope")
SCOPE_VERSION = 1
SOURCE_TYPE_MEDIA = "media"
SOURCE_TYPE_NOTE = "note"
_KNOWN_SOURCE_TYPES = (SOURCE_TYPE_MEDIA, SOURCE_TYPE_NOTE)

@dataclass(frozen=True)
class ScopeItem:
    source_type: str
    source_id: str

@dataclass(frozen=True)
class RagScope:
    items: tuple[ScopeItem, ...]
    updated_at: str

def serialize_scope(scope: RagScope) -> dict:
    """Serialize a scope to its stored JSON-safe dict shape.

    Args:
        scope: The scope to serialize.

    Returns:
        Dict with ``version``, ``updated_at`` and ``items`` keys.
    """
    return {
        "version": SCOPE_VERSION,
        "updated_at": scope.updated_at,
        "items": [{"source_type": i.source_type, "source_id": i.source_id} for i in scope.items],
    }

def parse_scope(raw: Any) -> Optional[RagScope]:
    """Parse a stored scope payload; any invalid input reads as unscoped.

    Args:
        raw: The raw value read from conversation metadata or the
            workspace scope table (may be anything).

    Returns:
        A ``RagScope``, or ``None`` (unscoped) for missing, malformed,
        or newer-versioned payloads. Never raises.
    """
    if not isinstance(raw, dict):
        return None
    version = raw.get("version")
    if not isinstance(version, int) or version > SCOPE_VERSION or version < 1:
        if version is not None and version != SCOPE_VERSION:
            logger.warning("rag_scope payload version {} unsupported; treating as unscoped", version)
        return None
    items_raw = raw.get("items")
    updated_at = raw.get("updated_at")
    if not isinstance(items_raw, list) or not isinstance(updated_at, str):
        return None
    items: list[ScopeItem] = []
    for entry in items_raw:
        if not isinstance(entry, dict):
            return None
        stype = entry.get("source_type")
        sid = entry.get("source_id")
        if stype not in _KNOWN_SOURCE_TYPES:
            continue  # forward-compat: unknown types dropped (spec D5)
        if sid is None:
            return None
        items.append(ScopeItem(str(stype), str(sid)))
    return RagScope(items=tuple(items), updated_at=updated_at)
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit** `git add tldw_chatbook/Chat/rag_scope.py Tests/Chat/test_rag_scope.py && git commit -m "feat(rag-scope): scope model + guarded codecs"`

### Task 2: Resolver with three states + stamp-keyed cache

**Files:**
- Modify: `tldw_chatbook/Chat/rag_scope.py`
- Test: `Tests/Chat/test_rag_scope.py`

**Interfaces:**
- Produces: `EffectiveScope` dataclass: `state: Literal["unscoped","scoped","empty"]`, `allowlist: dict[str, frozenset[str]]` (source_type → ids; empty when not scoped), `cause: Optional[str]`; `resolve_effective_scope(conv_scope: Optional[RagScope], ws_scope: Optional[RagScope], existing_ids: Callable[[str, frozenset[str]], frozenset[str]]) -> EffectiveScope` — pure function; DB access is injected via `existing_ids(source_type, ids) -> surviving ids` for dangling-drop. `ScopeCache` class with `get(conversation_id, workspace_id, conv_stamp, ws_stamp) -> Optional[EffectiveScope]` / `put(...)` keyed on all four values.
- Causes (exact strings later surfaced in diagnostics): `"deleted-items"`, `"no-workspace-overlap"`.

- [ ] **Step 1: Write failing tests** — cases: both None → unscoped; conv only → scoped with conv ids; both → intersection; disjoint → empty/`no-workspace-overlap`; all ids dangling (existing_ids returns empty) → empty/`deleted-items`; partial dangling drops only missing; cache returns entry only on exact 4-tuple match.

```python
def test_intersection_and_causes():
    conv = RagScope(items=(ScopeItem("media", "1"), ScopeItem("media", "2")), updated_at="t1")
    ws = RagScope(items=(ScopeItem("media", "2"), ScopeItem("media", "3")), updated_at="t2")
    keep_all = lambda st, ids: ids
    eff = resolve_effective_scope(conv, ws, keep_all)
    assert eff.state == "scoped" and eff.allowlist["media"] == frozenset({"2"})
    disjoint = RagScope(items=(ScopeItem("media", "9"),), updated_at="t3")
    assert resolve_effective_scope(disjoint, ws, keep_all).cause == "no-workspace-overlap"
    gone = lambda st, ids: frozenset()
    assert resolve_effective_scope(conv, None, gone).cause == "deleted-items"

def test_cache_key_includes_ids_not_just_stamps():
    cache = ScopeCache()
    eff = EffectiveScope(state="unscoped", allowlist={}, cause=None)
    cache.put("c1", "w1", "s1", "s2", eff)
    assert cache.get("c1", "w1", "s1", "s2") is eff
    assert cache.get("c1", "w2", "s1", "s2") is None  # re-linked workspace: same stamps, different key
```

- [ ] **Step 2: Run → FAIL** · **Step 3: Implement** (intersection on `ScopeItem` sets when both present; single-level uses that level alone; dangling-drop via `existing_ids` per source_type; empty→cause; `ScopeCache` = dict keyed by the 4-tuple, `clear()` for tests) · **Step 4: Run → PASS** · **Step 5: Commit** `feat(rag-scope): resolver with intersection, causes, stamp-keyed cache`

### Task 3: Store-level allowlist filtering (V1 fix)

**Files:**
- Modify: `tldw_chatbook/RAG_Search/simplified/vector_store.py` (both `ChromaVectorStore` and `InMemoryVectorStore`: `search`, `search_with_citations`)
- Modify: `tldw_chatbook/RAG_Search/simplified/rag_service.py` (`search`, `_semantic_search`)
- Test: `Tests/RAG/test_scope_store_filtering.py`

**Interfaces:**
- Produces: new keyword-only param on both store classes' `search`/`search_with_citations`: `metadata_allowlist: Optional[Mapping[str, Collection[str]]] = None` (meaning: metadata key → allowed values; a result must match EVERY key's allowlist). `RAGService.search`/`_semantic_search` gain the same `metadata_allowlist` param and thread it to the store. Chroma translation: single key → `where={key: {"$in": sorted(values)}}`; multiple keys → `where={"$and": [{k: {"$in": sorted(v)}} for ...]}`. InMemory: skip candidates failing `str(meta.get(k)) in values`. Existing `filter_metadata` behavior untouched (backward compat).

- [ ] **Step 1: Contract test, both stores** (the spec's parity requirement):

```python
import pytest
@pytest.mark.parametrize("store_kind", ["memory", "chroma"])
def test_allowlist_filters_at_store_level(store_kind, tmp_path):
    """Same scoped search behaves identically on both stores; out-of-scope
    docs are excluded even when they dominate similarity."""
    if store_kind == "chroma":
        pytest.importorskip("chromadb")
    store = _make_store(store_kind, tmp_path)  # helper mirroring test_vector_store_selection.py
    _add(store, "d1", meta={"source_id": "1", "source_type": "media"})
    _add(store, "d2", meta={"source_id": "2", "source_type": "media"})
    hits = store.search(_query_vec(), top_k=10, metadata_allowlist={"source_id": {"1"}})
    assert [h.metadata["source_id"] for h in hits] == ["1"]
```

Plus: `test_semantic_search_threads_allowlist` — `RAGService.search(..., metadata_allowlist=...)` with a spy store asserts the param reaches the store call (kills the silent post-filter path), and a starvation regression: 50 indexed docs, scope of 1, top_k=5 → the in-scope doc IS returned (post-filtering would return ≈nothing).

- [ ] **Step 2: Run → FAIL (unexpected kwarg)** · **Step 3: Implement** in both stores + service threading · **Step 4: Run → PASS; also run `Tests/RAG/` full (no regressions)** · **Step 5: Commit** `feat(rag): store-level metadata allowlist filtering on both vector stores`

### Task 4: FTS allowlist predicates in pipeline legs + `build_scope_filter`

**Files:**
- Modify: `tldw_chatbook/Chat/rag_scope.py` (add `build_semantic_allowlist(eff) -> Optional[dict]`, `media_id_params(eff) -> Optional[list[str]]`, `note_id_params(eff) -> Optional[list[str]]`)
- Modify: `tldw_chatbook/RAG_Search/pipeline_functions_simple.py` (`search_media_fts5`, `search_notes_fts5`, `search_conversations_fts5`, `search_semantic`)
- Test: `Tests/RAG/test_scope_pipeline_enforcement.py`

**Interfaces:**
- Consumes: `EffectiveScope` (Task 2), `metadata_allowlist` (Task 3), `PipelineContext` + diagnostics recording (existing task-250 machinery in `RAG_Search/semantic_availability.py` / `pipeline_types.py`).
- Produces: `PipelineContext["scope"]` carries the `EffectiveScope` (seeded by callers in Task 5); each leg self-enforces: media/notes legs add `AND id IN (SELECT value FROM json_each(?))` (single bound JSON array — no param-limit issue) to their queries when scoped; conversations leg returns `[]` and records diagnostic `scope: conversations-excluded`; `search_semantic` passes `build_semantic_allowlist` result as `metadata_allowlist`. New diagnostic reasons: `SCOPE_REASON_CONVERSATIONS_EXCLUDED = "scope_conversations_excluded"`, `SCOPE_REASON_EMPTY = "scope_empty"` (constants in `rag_scope.py`).

- [ ] **Step 1: Failing tests** — seeded real in-memory DBs (pattern from `Tests/RAG/test_semantic_honest_states.py`): scoped media search returns only allowlisted ids; notes likewise; conversations leg returns `[]` + diagnostic when scope active and normal results when not; `search_semantic` receives the allowlist (spy service); custom-pipeline inheritance: execute a hand-built TOML-style pipeline via `execute_pipeline` with scope in context → same enforcement (self-enforcement proof).
- [ ] **Step 2: Run → FAIL** · **Step 3: Implement** (legs read `context.get("scope")`; unscoped → exact current behavior, zero drift asserted by running existing `Tests/RAG/` suites) · **Step 4: All green** · **Step 5: Commit** `feat(rag-scope): pipeline legs self-enforce scope allowlists`

### Task 5: Conversation storage + chat-path wiring + EMPTY short-circuit

**Files:**
- Modify: `tldw_chatbook/Chat/rag_scope.py` (storage: `read_conversation_scope(db, conversation_id)`, `write_conversation_scope(db, conversation_id, scope|None)` over `conversations.metadata["rag_scope"]` — same seam as chat-dictionaries attach; session-state holder for unpersisted sessions: `SessionScopeHolder` with `flush_to(db, conversation_id)`)
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py` (`get_rag_context_for_chat`: resolve via holder/persisted id → seed `PipelineContext["scope"]`; `EMPTY` → skip retrieval entirely, record `SCOPE_REASON_EMPTY` + cause, notify via the existing `_notify_semantic_leg_state` pathway wording: "Retrieval scope is empty ({cause}); no sources searched.")
- Test: `Tests/Chat/test_rag_scope_storage.py`, extend `Tests/RAG/test_scope_pipeline_enforcement.py`

**Interfaces:**
- Consumes: `persisted_conversation_id` from the active session (grep call sites in `console_chat_controller.py` for the seam name at implementation time — Global Constraint applies).
- Produces: end-to-end scoped chat send: one integration test drives the legacy send path (`get_rag_context_for_chat`) with scope active → injected context contains ONLY in-scope content (spec §5's e2e requirement); EMPTY short-circuit test asserts zero leg executions (spy) + diagnostic cause.

- [ ] Steps: failing tests (storage round-trip on real ChaChaNotes DB incl. malformed-metadata guard; unpersisted holder flush; e2e send; EMPTY) → implement → green (+ `Tests/Event_Handlers/` regression) → commit `feat(rag-scope): conversation scope storage + chat enforcement + EMPTY short-circuit`

### Task 6: Backend B — caller-passed scope in `LibraryLocalRagSearchService`

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py` (`search(..., scope: Optional[EffectiveScope] = None)`; keyword seams get allowlist kwargs; conversations seam skipped when scope passed, diagnosed; semantic delegate passes `metadata_allowlist`)
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py` + `tldw_chatbook/Notes/Notes_Library.py` seam functions (optional `id_allowlist` param → `json_each` predicate; verify at implementation these seams ignore Library visibility filters — assert in test)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` Console run-library-rag call site (resolve + pass scope)
- Test: `Tests/Library/test_library_rag_scope.py`

**Interfaces:**
- Consumes: `EffectiveScope`, `build_scope_filter` helpers (Tasks 2/4).
- Produces: Console Run Library RAG honors scope; Library-screen callers unchanged (no scope kwarg → unscoped — D2 guard test: library_screen call sites pass nothing and results are unfiltered).

- [ ] Steps: failing tests (scoped keyword search returns only allowlisted; conversations seam skipped + diagnosed; Library-screen path unscoped; scoped-zero-results marker "No results within scope (N items searched)") → implement → green (+ `Tests/Library/` full regression, 548-test suite must stay green) → commit `feat(rag-scope): Library service caller-passed scope (Console Run Library RAG)`

### Task 7: Phase-1 close-out

- [ ] Backlog task created at branch time (ID past all open-branch claims; re-verify at merge) with plan/AC/notes lifecycle; `python -c "import tldw_chatbook.app"`; full `Tests/RAG/ Tests/Chat/ Tests/Library/ Tests/Event_Handlers/` run with exact numbers vs baseline; ruff on changed files; PR to dev — **hold for user approval; do not merge**. Include the native-send follow-up backlog task (no RAG injection on native path — pre-existing, out of scope).

---

# Phase 2 — Conversation-level UI: modal, Inspector row, header chip (PR 2)

### Task 8: `ConsoleScopePickerModal` — data loading + list

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_scope_picker_modal.py`
- Test: `Tests/UI/test_console_scope_picker_modal.py`

**Interfaces:**
- Consumes: read-only listing seams (media list via `media_reading_scope_service`, notes via Notes seams — id, title, type, updated stamp, keywords); `RagScope`/`ScopeItem`.
- Produces: `ConsoleScopePickerModal(target_label: str, universe: Optional[frozenset[tuple[str, str]]], initial: Optional[RagScope], on_save: Callable[[Optional[RagScope]], None])` — `universe=None` means full library; non-None restricts offered items (D3). Modal returns via `on_save`: `RagScope` (≥1 item) or `None` (clear — zero-selection save).
- Follow `console_prompt_picker_modal.py` conventions (focus trap, Esc, worker-loaded list, `group=` on workers). Pagination; type tabs All/Media/Notes; sort Recent/Title/Type (default Recent); text filter; tag selector (top-10 chips + autocomplete input; multi-tag OR, AND with text; per-item match union across types); All/Selected view toggle (default Selected when `initial` non-empty); out-of-universe items in Selected view greyed with "outside workspace scope" label; "Select all matching" via id-only seam query honoring `universe`, with count confirmation; footer live count + Save/Clear/Cancel.

- [ ] Steps (TDD per behavior, one commit per green cycle): list loads off-loop + renders types/glyphs → selection survives filter/sort changes → Selected-view default + out-of-universe marking → select-all-matching honors universe + confirmation → zero-selection save calls `on_save(None)` → tag OR/AND semantics. Commit trail: `feat(console): scope picker modal (list/selection/views/tags)`

### Task 9: Inspector "Retrieval scope" row + wiring

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (compose: compact row directly BELOW the Sources tray in `#console-inspector-rail-body`; id `console-retrieval-scope-row`; states "Scope: everything · Narrow…" / "Scope: N items · Edit · Clear"; handlers open modal with conversation target + persist via Task-5 storage/holder; refresh row + run-recipe line on save)
- Test: extend `Tests/UI/test_search_rag_window.py`-style Console suite (`Tests/UI/test_console_scope_row.py`)

**Interfaces:** Consumes Task 5 storage + Task 8 modal. Produces: row position pinned by test (below tray, above run inspector — placement-test pattern from task-400); zero-DB-on-recompose (row renders from session state; scope read happens on modal open / save, off-loop).

- [ ] Steps: failing placement+state tests → implement → green (Console suites at baseline) → commit `feat(console): Inspector retrieval-scope row`

### Task 10: Header chip (effective count)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` chip row (pattern: existing `Sources: 0 staged` chip); id `console-scope-chip`
- Test: extend `Tests/UI/test_console_scope_row.py`

**Interfaces:** chip hidden when unscoped; "Scope: N" = effective count; tooltip "conversation A ∩ workspace B → N" when both levels; `EMPTY` → action-required styling (badge-precedence conventions) + cause in tooltip. Refresh triggers: modal save, session switch.

- [ ] Steps: failing tests (hidden/count/empty styling) → implement → green → commit `feat(console): effective-scope header chip`

### Task 11: Phase-2 close-out

- [ ] Backlog task lifecycle; full Console UI suites vs baseline; QA captures via the textual-serve rig (modal All + Selected views, Inspector row both states, chip incl. EMPTY) into `Docs/superpowers/qa/rag-scope-2026-07/`; PR to dev — **hold for user screen approval; do not merge**.

---

# Phase 3 — Workspace-level scope (PR 3)

### Task 12: `workspace_rag_scopes` table + registry accessors

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (next schema version bump — CHECK CURRENT VERSION at implementation; memory notes v20 was current with v21 reserved earlier, other programs may have advanced it; idempotent PRAGMA-guarded migration creating `workspace_rag_scopes(workspace_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)`)
- Modify: `tldw_chatbook/Workspaces/registry_service.py` (`get_workspace_scope(workspace_id) -> Optional[RagScope]` guarded-parse; `set_workspace_scope(workspace_id, scope|None)`; delete record on workspace deletion — hook the existing delete path)
- Test: `Tests/ChaChaNotesDB/test_workspace_rag_scopes.py`

- [ ] Steps: failing tests (migration idempotent on v(N-1)→vN and re-run; round-trip; delete-cascade; orphan tolerated) → implement → green (+ full `Tests/ChaChaNotesDB/` regression) → commit `feat(rag-scope): workspace scope table + registry accessors (schema vN)`

### Task 13: Workspace entry point + intersection live

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (Scope button beside the workspace row in the Session section → modal with workspace target, `universe=None`; conversation-target modal now passes `universe=workspace scope` when active (D3); resolver call sites pass workspace scope from registry; chip/tooltip intersection display)
- Test: `Tests/UI/test_console_scope_row.py` extensions + resolver-integration test (conversation modal universe restricted; effective chip shows intersection; fork-into-scoped-workspace shows EMPTY `no-workspace-overlap`)

- [ ] Steps: failing tests → implement → green → commit `feat(rag-scope): workspace-level scoping + intersection UI`

### Task 14: Phase-3 close-out

- [ ] Backlog lifecycle; large-scope test (~1k items → `json_each` + Chroma `$in` behavior — the spec's V3 empirical check lands here); full-suite numbers vs baseline; QA captures (workspace modal, intersection chip/tooltip, EMPTY-with-cause); PR — **hold for user approval**.

---

## Self-review (done at write time)

- Spec coverage: D1→Tasks 3/4/5/6; D2→Task 6 D2-guard test; D3→Tasks 8/13; D4→static `RagScope` only; D5→Tasks 4/6 exclusion + diagnostics, vocabulary guard in Task 1; D6→Tasks 9/10/13. §2 storage→Tasks 1/5/12; §3 both backends+self-enforcement→4/5/6; §4 modal details→8; §5 tests distributed incl. e2e (Task 5), contract (Task 3), large-scope (Task 14); §6 verifications: V1 resolved→Task 3, V2 resolved→Global Constraints, V3→Task 14, V4 resolved→Global Constraints follow-up, V5 resolved→Task 12, V6→asserted in Task 6 tests.
- No placeholders; later-task names match earlier definitions (`EffectiveScope`, `metadata_allowlist`, `SessionScopeHolder`, cause strings).
- Type consistency checked: `EffectiveScope.allowlist: dict[str, frozenset[str]]` consumed by `build_semantic_allowlist`/`media_id_params`/`note_id_params` (Task 4) and Backend B (Task 6).
