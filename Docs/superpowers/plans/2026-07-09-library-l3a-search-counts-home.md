# Library L3a — Search promotion + Create counts + Home due-mirror — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote Library Search to a first-class canvas with a working keyword backend, query history, and per-result Open; add live counts to the Create rail rows; mirror flashcards-due into Home Needs Attention routing one-hop to the Study flashcards surface.

**Architecture:** Extends the shipped Library rail+canvas shell (PRs #582/#585/#586/#587). Pure-state modules stay Textual-free; the screen orchestrates; services are reached via `getattr(app, "<service>", None)` with quiet degrade. A new production `LibraryRagSearchService` implementation backs `search` mode with the already-wired FTS seams; `rag` mode degrades quietly when no RAG runtime exists.

**Tech Stack:** Python 3.11+, Textual, pytest (+pytest-asyncio pilots), SQLite (ChaChaNotes/Media DBs), textual-serve + playwright for QA.

**Spec:** `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md` (Phase L3a + Global constraints). Branch `claude/library-l3a` off `origin/dev` (10abdede), worktree `.claude/worktrees/library-l3a`.

## Global Constraints

- Canvas grammar: stacked full-width render-verified widgets (Static / Input / TextArea / Markdown / Collapsible / Button / VerticalScroll). No `Select`; cycling buttons instead. Never a `Horizontal` mixing a `1fr` sibling with fixed-width children.
- Width rule: a canvas child that fills the canvas uses `1fr`, never its own `Nfr`; long text bodies get `width: 100%`, containers `overflow-x: hidden`.
- Services via `getattr(app, "<service>", None)` with quiet degrade; long calls through workers (never block compose).
- Mutation→refetch→recompose; targeted `update()` for status lines; **poll for expected widget state after a recompose — never `query_one` the instant a running-flag flips** (L2b.2 NoMatches lesson).
- State reset discipline on every canvas entry/exit path; note-editor flush on every exit path (`await self._flush_library_note_save()`; conflict aborts).
- Input sanitization on every persisted field (`self._safe_text(...)` / `input_validation` helpers); parameterized SQL only.
- CSS in `css/components/_agentic_terminal.tcss` → `./build_css.sh` → commit BOTH files.
- Test fakes must mirror the real service's method seams/signatures. **Any gated/blocking fake must bound its wait** (`event.wait(30.0)` — unbounded waits wedge pytest shutdown).
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <file> -x -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread` with `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share`.
- Git: stage ONLY changed files by explicit path (never `git add -A`); never touch `.claude/settings.local.json`; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Docstrings on new/changed public seams: Google style with Args/Returns (post-#587 convention).
- Exact copy values in this plan are binding: `Search Library…`, `Recent searches`, `searching · `, `Flashcards due: N`, `Review flashcards`, `done` copies as written per task.

### Key existing anchors (verified 2026-07-09)

| Anchor | Location |
|---|---|
| Rail row table (`browse-search` = kind `mode`) | `tldw_chatbook/Library/library_shell_state.py:84-190` |
| Canvas dispatch by `canvas_kind` | `tldw_chatbook/UI/Screens/library_screen.py:2897-3027` (`compose_content`) |
| Mode canvas (search/collections bodies to relocate) | `library_screen.py:4281-4322` (`_compose_mode_canvas`) |
| Rail-top input handler (conversations filter today) | `library_screen.py:6252-6264` (`handle_library_search_submitted`) |
| RAG panel state builder (mode/gates hardcoded) | `library_screen.py:2115-2142` (`_library_rag_panel_state`) |
| RAG worker (default-group collision) | `library_screen.py:7093-7096` (`_execute_library_rag_search`) |
| RAG service seam + outcome normalizer | `tldw_chatbook/Library/library_rag_service.py` (whole file, 240 lines) |
| RAG pure state (`LibraryRagPanelState`, `LibraryRagResultRow`) | `tldw_chatbook/Library/library_rag_state.py` |
| Search FTS seams | `notes_scope_service.search_notes` (`Notes/notes_scope_service.py:677`), `media_reading_scope_service.search_media` (`Media/media_reading_scope_service.py:582`), `CharactersRAGDB.search_conversations_by_content` (`DB/ChaChaNotes_DB.py:5197`) |
| Snapshot/counts worker | `library_screen.py:753-762` + `_list_local_source_snapshot` `:1005-1125` |
| Rail count suffix | `Widgets/Library/library_rail.py:128-134`, label at `:205-208` |
| Study DB seams | `CharactersRAGDB.get_due_flashcards` (`:7445`), `list_decks` (`:7485`), `list_quizzes` (`:8364`), `count_notes` template (`:6660`) |
| Home triage builder | `tldw_chatbook/Home/dashboard_state.py:619-720` (`build_home_triage_state`), controls `:199-310` |
| Home control dispatch | `tldw_chatbook/UI/Screens/home_screen.py:40-57`, `:331-355` |
| Study deep-link | `app.open_study_screen(scope_context=None, *, initial_section=None)` (`app.py:1660`), valid section `"flashcards"` |
| Library nav-context note branch (open-by-id template) | `library_screen.py:719-744`; media row open `:4501-4535`; conversations `_ensure_selected_conversation_id` `:1166-1177` |

---

### Task 1 (LEAD-EXECUTED, no subagent): Served-TUI smoke of the current Search/RAG panel

Spec mandates "smoke first": the panel predates all L2a rendering discoveries. Capture what `browse-search` renders today, populated, before building on it.

**Files:**
- Create: `Docs/superpowers/qa/library-l3a-2026-07/README.md` (inventory notes)
- Create: `Docs/superpowers/qa/library-l3a-2026-07/pre-search-panel-idle.png`, `pre-search-panel-blocked.png`

- [ ] **Step 1:** Launch textual-serve with isolated HOME (`/private/tmp/tldw-l3a-qa`), seed a few notes/media/conversations via the DB API at `$HOME/.local/share/tldw_cli/default_user/tldw_chatbook_ChaChaNotes.db` (client_id `tldw_cli_local_instance_v1`).
- [ ] **Step 2:** Playwright chromium at viewport **2050x1240, device_scale_factor 1**. Navigate: Library tab → click `Search / RAG` rail row. Capture idle panel; submit a query; capture the blocked "service unavailable" state.
- [ ] **Step 3:** Inventory in the QA README: every region that renders/clips/mis-sizes (the panel uses fixed heights `#library-rag-query-controls: height 11/14`, `#library-rag-source-scope: height 13/16`, `#library-rag-results: 1fr` — verify none clip at canvas width). Note defects for Task 5 to absorb.
- [ ] **Step 4:** Commit QA artifacts: `git add Docs/superpowers/qa/library-l3a-2026-07/ && git commit -m "docs(library): L3a pre-change search panel smoke captures"`.

---

### Task 2: Search history + open-target pure state (`library_rag_state.py`)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py`
- Test: `Tests/Library/test_library_rag_state.py`

**Interfaces:**
- Produces: `LIBRARY_SEARCH_HISTORY_LIMIT = 10`, `LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS = 200`, `update_search_history(history, query) -> tuple[str, ...]`, `searching_status_line(source_types) -> str`, `LibraryRagResultRow.open_source_type: str` property, `LibraryRagResultRow.can_open: bool` property, `LibraryRagPanelState.history: tuple[str, ...]` field (+ `history` param on `LibraryRagPanelState.from_values`).

- [ ] **Step 1: Write the failing tests** (append to `Tests/Library/test_library_rag_state.py`):

```python
class TestUpdateSearchHistory:
    def test_prepends_new_query(self):
        assert update_search_history(("b",), "a") == ("a", "b")

    def test_exact_match_dedupes_to_front(self):
        assert update_search_history(("a", "b", "c"), "b") == ("b", "a", "c")

    def test_caps_at_ten_entries(self):
        history = tuple(f"q{i}" for i in range(10))
        result = update_search_history(history, "new")
        assert len(result) == 10
        assert result[0] == "new"
        assert "q9" not in result

    def test_truncates_entries_to_200_chars(self):
        result = update_search_history((), "x" * 500)
        assert result == ("x" * 200,)

    def test_blank_query_is_ignored(self):
        assert update_search_history(("a",), "   ") == ("a",)


class TestSearchingStatusLine:
    def test_lists_selected_sources(self):
        assert searching_status_line(("notes", "media")) == "searching · notes, media…"

    def test_empty_scope_still_reads_searching(self):
        assert searching_status_line(()) == "searching…"


class TestResultRowOpenTarget:
    def test_note_result_opens_notes(self):
        row = LibraryRagResultRow.from_result(
            {"source_id": "note-42", "title": "T", "snippet": "s",
             "provenance": {"source_type": "note"}}
        )
        assert row.open_source_type == "notes"
        assert row.can_open is True

    def test_media_and_conversation_map(self):
        media = LibraryRagResultRow.from_result(
            {"source_id": "7", "title": "T", "snippet": "s",
             "provenance": {"source_type": "media"}}
        )
        convo = LibraryRagResultRow.from_result(
            {"source_id": "c1", "title": "T", "snippet": "s",
             "provenance": {"source_type": "conversation"}}
        )
        assert media.open_source_type == "media"
        assert convo.open_source_type == "conversations"

    def test_unknown_type_or_missing_id_cannot_open(self):
        no_type = LibraryRagResultRow.from_result(
            {"source_id": "x", "title": "T", "snippet": "s"}
        )
        no_id = LibraryRagResultRow.from_result(
            {"title": "T", "snippet": "s", "provenance": {"source_type": "note"}}
        )
        assert no_type.can_open is False
        assert no_id.can_open is False


class TestPanelStateHistory:
    def test_from_values_carries_history(self):
        state = LibraryRagPanelState.from_values(history=("a", "b"))
        assert state.history == ("a", "b")

    def test_history_defaults_empty(self):
        assert LibraryRagPanelState.from_values().history == ()
```

- [ ] **Step 2: Run to verify failure** — `ImportError`/`AttributeError` expected.
- [ ] **Step 3: Implement** in `library_rag_state.py`:

```python
LIBRARY_SEARCH_HISTORY_LIMIT = 10
LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS = 200


def update_search_history(history: Sequence[str], query: str) -> tuple[str, ...]:
    """Return search history with `query` prepended, deduped, capped at 10.

    Args:
        history: Existing history entries, most recent first.
        query: Newly submitted query; blank input leaves history unchanged.

    Returns:
        New history tuple, entries truncated to 200 chars, length <= 10.
    """
    entry = (query or "").strip()[:LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS]
    if not entry:
        return tuple(str(item) for item in history)
    deduped = [entry] + [str(item) for item in history if str(item) != entry]
    return tuple(deduped[:LIBRARY_SEARCH_HISTORY_LIMIT])


def searching_status_line(source_types: Sequence[str]) -> str:
    """Build the visible in-flight status line for a running search."""
    labels = ", ".join(str(s) for s in source_types if str(s).strip())
    return f"searching · {labels}…" if labels else "searching…"
```

On `LibraryRagResultRow`:

```python
_OPEN_SOURCE_TYPE_MAP = {
    "note": "notes", "notes": "notes",
    "media": "media", "media_chunk": "media",
    "conversation": "conversations", "conversations": "conversations",
    "chat": "conversations",
}

@property
def open_source_type(self) -> str:
    """Library canvas target this result can open, or empty string."""
    raw = str(
        self.provenance.get("source_type")
        or self.provenance.get("item_type")
        or self.provenance.get("type")
        or ""
    ).strip().lower()
    return _OPEN_SOURCE_TYPE_MAP.get(raw, "")

@property
def can_open(self) -> bool:
    """True when the row carries a resolvable parent id and known type."""
    return bool(self.open_source_type and self.source_id)
```

`LibraryRagPanelState`: add field `history: tuple[str, ...] = ()`; `from_values` gains `history: Sequence[str] = ()` and passes `history=tuple(str(h) for h in history)`.

- [ ] **Step 4: Run tests to verify pass** (whole file — existing tests must stay green).
- [ ] **Step 5: Commit** — `git add tldw_chatbook/Library/library_rag_state.py Tests/Library/test_library_rag_state.py && git commit -m "feat(library): search history + open-target pure state for L3a"`.

---

### Task 3: Production `LibraryLocalRagSearchService` + app wiring

**Files:**
- Create: `tldw_chatbook/Library/library_local_rag_search_service.py`
- Modify: `tldw_chatbook/app.py` (wire `self.library_rag_search_service` — place next to the study service wiring around `app.py:2078-2087`)
- Test: `Tests/Library/test_library_local_rag_search_service.py`

**Interfaces:**
- Consumes: `notes_scope_service.search_notes(*, scope, query, limit, offset=0, user_id=None, ...)` (`Notes/notes_scope_service.py:677`); `media_reading_scope_service.search_media(*, mode=None, query=None, limit=20, offset=0, **filters) -> {"items": [...], "total", ...}` (`Media/media_reading_scope_service.py:582`); `CharactersRAGDB.search_conversations_by_content(search_query, limit=10)` (`DB/ChaChaNotes_DB.py:5197`, sync — call via `asyncio.to_thread`); `LibraryRagSearchOutcome` (service may return it directly, `library_rag_service.py:134-136`); `app._rag_service.search(query=..., search_type="semantic", top_k=..., include_citations=True)` (verify exact signature at `RAG_Search/simplified/rag_service.py:476` before implementing the rag branch).
- Produces: `class LibraryLocalRagSearchService` with `async def search(self, query: str, scope: tuple[str, ...], mode: str, **kwargs) -> Any` (protocol at `library_rag_service.py:27`). App attr `library_rag_search_service`.

Behavior contract:
- `mode == "search"` (keyword, must ALWAYS work when any source seam exists): fan out per source type in `scope` (order notes, media, conversations; ignore unknown types quietly); each source fetches up to `top_k` records; a missing/erroring seam contributes zero rows (log warning, never raise). Return `{"results": [...], "runtime_backend": "local-fts"}` — normalizer turns rows→`ready`, none→`empty`. If NO queried seam is available at all, return a `LibraryRagSearchOutcome(status="blocked", recovery_state=<service-unavailable state>)`.
- `mode == "rag"`: if `getattr(app, "_rag_service", None)` is None → `LibraryRagSearchOutcome(status="blocked", recovery_state=_rag_mode_unavailable_recovery_state())` (quiet line + setup routing per spec; do NOT import torch/probe embeddings on this path — `_rag_service` presence IS the gate). Otherwise delegate and map results.
- Result-row dicts: `{"source_id": str(id), "chunk_id": "", "title": <title>, "snippet": <content, rely on state-layer truncation>, "score": <float or None>, "provenance": {"source_type": "note"|"media"|"conversation"}}`. Conversations have no content field: `snippet = f"Matched conversation · {message_count} messages"` (fields from `search_conversations_by_content`: `id, title, message_count, ...`).
- `user_id` for notes: `getattr(app, "notes_user_id", None) or "default_user"` (matches `library_screen.py:1022`).

- [ ] **Step 1: Write the failing tests.** Fakes MUST mirror real seam signatures exactly (keyword-only params as in the real methods). Cover: (a) search mode returns note+media+conversation rows with correct `source_id`/`provenance.source_type` — conversations against a REAL in-memory `CharactersRAGDB(":memory:", client_id="test-client")` seeded via `add_character_card`-free conversation+message creation (mirror `Tests/ChaChaNotesDB` fixtures) so FTS is exercised; (b) a missing media service quietly yields notes/conversations rows only; (c) all seams missing → `LibraryRagSearchOutcome` with `status == "blocked"`; (d) rag mode with `_rag_service=None` → blocked outcome whose recovery `next_action` mentions switching mode to Search; (e) scope filtering — `scope=("notes",)` never touches the media fake (assert not called); (f) end-to-end through `run_library_rag_search` with the service attached to a fake app → `status == "ready"` and rows normalized to `LibraryRagResultRow`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the module:

```python
"""Local production backend for Library Search/RAG retrieval."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_chatbook.Library.library_rag_service import LibraryRagSearchOutcome
from tldw_chatbook.Library.library_rag_state import LIBRARY_RAG_SERVICE_ERROR_SELECTOR
from tldw_chatbook.UI.destination_recovery import DestinationRecoveryState

logger = logger.bind(module="LibraryLocalRagSearchService")

_SEARCH_RUNTIME_BACKEND = "local-fts"


class LibraryLocalRagSearchService:
    """Keyword-first Library retrieval over the app's local source seams.

    `search` mode fans out over notes/media/conversations FTS seams and
    always works when at least one seam is available. `rag` mode delegates
    to the app's `_rag_service` and degrades to a blocked outcome with
    setup routing when that runtime is absent.
    """

    def __init__(self, app_instance: Any) -> None:
        self._app = app_instance

    async def search(self, query: str, scope: tuple[str, ...], mode: str, **kwargs: Any) -> Any:
        top_k = max(1, int(kwargs.get("top_k") or 5))
        if mode == "rag":
            return await self._search_semantic(query, scope, top_k, kwargs)
        return await self._search_keyword(query, scope, top_k)
```

`_search_keyword` gathers the three `_search_<source>` coroutines for source types present in `scope` (each wrapped so exceptions log + return `[]`, and each returns `(available: bool, rows: list[dict])` so "no seam" is distinguishable from "no matches"); if no source was available → blocked outcome via `_no_backend_recovery_state()`; else `{"results": notes + media + conversations, "runtime_backend": _SEARCH_RUNTIME_BACKEND}`. `_search_semantic` gates on `_rag_service`, calls its `search`, maps result items into the row-dict shape (fold unrecognized fields into `provenance`), falls back to `provenance.source_type` from each item when present. Both recovery-state builders mirror the shape of `_service_unavailable_recovery_state()` (`library_rag_service.py:194-207`); the rag one uses `status_label="RAG unavailable"`, `next_action="Install embeddings support or switch mode to Search"`, `recovery_action="Settings > RAG"`.

App wiring (next to study services):

```python
from tldw_chatbook.Library.library_local_rag_search_service import LibraryLocalRagSearchService
self.library_rag_search_service = LibraryLocalRagSearchService(self)
```

- [ ] **Step 4: Run tests to verify pass**, plus `Tests/Library/test_library_rag_service.py` (must stay green).
- [ ] **Step 5: Commit** — message `feat(library): local FTS-backed Library RAG search service`.

---

### Task 4: Rail-row flip — `browse-search`/`browse-collections` mode→canvas + worker group fix

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py` (rows `:121-130` search, browse-collections just above)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`compose_content` dispatch `:2897-3027`; `_compose_mode_canvas` `:4281-4322`; `_execute_library_rag_search` `:7093`)
- Test: `Tests/Library/test_library_shell_state.py`, `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: `canvas_kind == "search"` and `canvas_kind == "collections"` branches in `compose_content`; module constants `LIBRARY_ROW_BROWSE_SEARCH = "browse-search"`, `LIBRARY_ROW_BROWSE_COLLECTIONS = "browse-collections"` in `library_shell_state.py`; RAG worker in group `"library_rag_search"`.
- Consumes: existing `LibrarySearchRagPanel`/`LibraryCollectionsPanel` compose bodies — MOVE them verbatim from `_compose_mode_canvas` (`:4307-4319`) into the new branches.

- [ ] **Step 1: Failing pure-state tests** — in `test_library_shell_state.py`: selecting `browse-search` yields `canvas_kind == "search"` (not `"mode"`); selecting `browse-collections` yields `canvas_kind == "collections"`; both rows' `target_kind == "canvas"`.
- [ ] **Step 2: Failing pilot** — in `test_library_shell.py`: pressing `#library-row-browse-search` mounts `#library-search-rag-panel` (poll for it; do not assert the mode-title block `#library-active-mode-title` exists — it must NOT for search). Mirror an existing rail-press pilot's fixture pattern.
- [ ] **Step 3: Run both to verify failure.**
- [ ] **Step 4: Implement.** (a) Flip the two rows to `target_kind="canvas"` (target_ids stay `"search"`/`"collections"`); hoist the row-id string literals into the two new constants and use them. (b) In `compose_content`, add branches BEFORE the `"mode"` branch:

```python
elif shell.canvas_kind == "search":
    yield LibrarySearchRagPanel(
        self._library_rag_panel_state(),
        id="library-search-rag-panel",
    )
elif shell.canvas_kind == "collections":
    # body moved verbatim from _compose_mode_canvas (was :4312-4319)
```

Keep `_compose_mode_canvas`'s search/collections branches DELETED (not duplicated); study/flashcards/quizzes/import-export stay mode-routed. The collections lazy-load path in `_select_library_rail_row` (`:4430-4435`, keyed on `_active_mode == "collections"`) still fires because canvas presses pass `target_id` as `active_mode`. (c) `_execute_library_rag_search` → `@work(exclusive=True, group="library_rag_search")` (fixes mutual-cancel with `_refresh_local_source_snapshot`). (d) Re-anchor any tests that asserted mode-canvas behavior for search/collections (grep `browse-search`, `library-mode-search`, `_compose_mode_canvas` in Tests/).
- [ ] **Step 5: Run** `Tests/Library/test_library_shell_state.py` + `Tests/UI/test_library_shell.py` + `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (search-panel pilots live here; re-anchor as needed).
- [ ] **Step 6: Commit** — `feat(library): promote search and collections rail rows to first-class canvases`.

---

### Task 5: Search canvas upgrades — mode toggle, live gates, status line, history UI

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_library_rag_panel_state` `:2115-2142`, `_start_library_rag_query` `:6905-6926`, new handlers)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ `./build_css.sh`, commit generated file)
- Test: `Tests/UI/test_library_shell.py` (pilots), `Tests/Library/test_library_rag_state.py` (any state deltas)

**Interfaces:**
- Consumes: Task 2's `history` state + `searching_status_line` + `update_search_history`; Task 4's canvas flip.
- Produces: screen fields `self._library_rag_mode: str = "search"`, `self._library_search_history: tuple[str, ...]` (loaded once in `__init__` from `app_config["library"]["search"]["history"]`, quiet default `()`); panel widgets `#library-rag-mode-toggle` (Button, label `mode: Search ▸` / `mode: RAG Answer ▸`), `#library-rag-searching-line` (Static, `searching_status_line(...)` while status searching), `Collapsible(title="Recent searches", id="library-rag-history")` containing `Button` rows `library-rag-history-{i}` (collapsed when results shown, expanded when idle); threaded persist `_save_library_search_history` mirroring `_save_library_rail_preferences` (`:4368-4374`) writing `save_setting_to_cli_config("library.search", "history", list(history))`.

- [ ] **Step 1: Failing pilots:**
  - mode toggle cycles: press `#library-rag-mode-toggle` → panel status line shows `Mode: RAG Answer`; press again → `Mode: Search`.
  - default mode is `search`; with a fake app lacking `_rag_service`, mode `rag` shows the blocked callout (run disabled) while mode `search` keeps run enabled.
  - submitting a query records history: submit `alpha` then `beta` → `Collapsible` contains buttons labeled `beta`, `alpha`; in-memory app_config `["library"]["search"]["history"] == ["beta", "alpha"]`.
  - clicking a history row re-runs: with a recording fake service, press `#library-rag-history-1` → service called with query `alpha`.
  - while a gated fake service holds the search open, `#library-rag-searching-line` renders `searching · notes, media, conversations…` (gate MUST use `event.wait(30.0)`).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
  - `_library_rag_panel_state`: pass `mode=self._library_rag_mode`, `history=self._library_search_history`, and `provider_ready=(getattr(self.app_instance, "_rag_service", None) is not None)`; `dependencies_ready`/`index_ready` stay `True` (deliberate: no torch import on the UI path; the service double-guards at call time).
  - Panel compose: mode toggle Button directly under `#library-rag-query-status`; searching line inside the results region when `query_state.status`/retrieval is searching; history `Collapsible` after the results region, `collapsed=bool(state.results)`; history rows are full-width Buttons (canvas grammar), label = the entry text.
  - Screen handlers:

```python
@on(Button.Pressed, "#library-rag-mode-toggle")
def cycle_library_rag_mode(self, event: Button.Pressed) -> None:
    event.stop()
    self._library_rag_mode = "rag" if self._library_rag_mode == "search" else "search"
    self._reset_library_rag_retrieval_state()
    self.refresh(recompose=True)

@on(Button.Pressed, ".library-rag-history-row")
def rerun_library_search_from_history(self, event: Button.Pressed) -> None:
    event.stop()
    index = self._trailing_index(event.button.id)  # reuse existing index-parse helper pattern (:7026)
    if index is None or index >= len(self._library_search_history):
        return
    self._library_rag_query = self._library_search_history[index]
    self._start_library_rag_query()
```

  - `_start_library_rag_query`: after a successful dispatch (run action enabled), record + persist history:

```python
self._library_search_history = update_search_history(self._library_search_history, request.query)
self._save_library_search_history(list(self._library_search_history))
```

  - History load in `__init__` reads `app_config` defensively (missing keys → `()`; non-list → `()`; entries coerced `str`, re-passed through `update_search_history` shape by slicing to 10).
  - CSS: history rows + searching line styles in `_agentic_terminal.tcss`; run `./build_css.sh`; commit BOTH css files.
- [ ] **Step 4: Run the pilots + full `Tests/UI/test_product_maturity_gate16_library_search_rag.py`.**
- [ ] **Step 5: Commit** — `feat(library): search canvas mode toggle, live status, query history`.

---

### Task 6: Rail-top rewire + conversations in-canvas filter

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`handle_library_search_submitted` `:6252-6264`, `_library_rail_search_placeholder` `:6278-6293`, rail ctor args `:2872-2882`)
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py` (filter Input)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: Tasks 4/5 (search canvas + `_library_rag_mode`).
- Produces: rail input is the Search feeder (single query truth = `self._library_rag_query`); conversations filtering moves to `#library-conversations-filter` (Enter-to-apply, client-side over the 50-record snapshot — same semantics as today, honestly documented in the handler docstring; service-backed FTS filter stays a tracked follow-up).

- [ ] **Step 1: Failing pilots:**
  - rail submit runs a search: type `zeta` into `#library-search-input`, Enter → search canvas selected (`#library-search-rag-panel` present — poll), recording fake service called with `("zeta", <scope>, "search")`, and focus returned to `#library-search-input`.
  - rail placeholder is `Search Library…` even while `browse-conversations` is selected.
  - conversations still filter: select conversations row, submit `alpha` in `#library-conversations-filter` → only matching rows render; blank submit restores all (re-anchor the existing rail-filter pilots to the new input id — grep `library-search-input` in `Tests/UI/`).
  - empty rail submit selects the search canvas WITHOUT invoking the service.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**

```python
@on(Input.Submitted, "#library-search-input")
async def handle_library_search_submitted(self, event: Input.Submitted) -> None:
    """Submit the rail-top query to the Search canvas (fast `search` mode)."""
    event.stop()
    query = self._safe_text(event.value, max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
    self._library_rag_query = query
    self._library_rag_mode = "search"
    await self._select_library_rail_row(LIBRARY_ROW_BROWSE_SEARCH, "search")
    if query.strip():
        self._start_library_rag_query()
    self.call_after_refresh(self._focus_library_search_input)
```

`_library_rail_search_placeholder` returns `"Search Library…"` unconditionally (keep the method; simplify body + docstring). Rail ctor: `query=self._library_rag_query`. Conversations canvas: `Input(value=self.state.query, placeholder="Filter conversations… (Enter)", id="library-conversations-filter")` directly under the status Static (`library_conversations_canvas.py:52-59`); screen handler sets `self._library_conversation_query = self._safe_text(event.value, max_length=200)`, recomposes, refocuses the filter via `call_after_refresh`. Remove the conversations-query reset special-case from `handle_library_rail_row` ONLY if pilots confirm entry-reset still happens via `_select_library_rail_row` — otherwise keep it (state-reset discipline).
`_start_library_rag_query`'s widget refresh must tolerate the panel mid-recompose (wrap the `_refresh_search_rag_panel_state_widgets` call so `NoMatches` is non-fatal — the recompose renders the same state).
- [ ] **Step 4: Run the pilots + the full `Tests/UI/test_library_shell.py`.**
- [ ] **Step 5: Commit** — `feat(library): rail-top search feeds the search canvas; conversations filter moves in-canvas`.

---

### Task 7: Result → Open + shared open-by-id route

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` (Open button per openable row)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_open_library_item_by_id`, Open handler)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: Task 2's `row.can_open`/`row.open_source_type`; templates: media open `library_screen.py:4501-4535`, notes nav-context branch `:719-744`, conversations `_ensure_selected_conversation_id` `:1166-1177`; `CharactersRAGDB.get_conversation_by_id(conversation_id, include_deleted=False)` (`DB/ChaChaNotes_DB.py:4400`).
- Produces: `async def _open_library_item_by_id(self, source_type: str, record_id: str) -> None` — the shared route L3b's ingest queue reuses; panel Button `library-rag-open-result-{index}` class `library-rag-result-open` rendered only when `row.can_open`.

Route behavior (straight to detail surfaces, never via list selection):
- `media`: flush note save; set `_selected_media_id`, `_library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA`, `_active_mode="media"`, `_library_media_view="viewer"`, clear media transient state (mirror `:4501-4535` exactly), run `_refresh_library_media_detail(record_id)` in its existing worker group, recompose.
- `notes`: flush note save (abort on conflict); set `_active_mode="notes"`, `_library_notes_view="editor"`, `_selected_note_id=record_id`, `_reset_library_note_editor_state()`, `_library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES`, run `_refresh_library_note_detail(record_id)` in its existing group, recompose. (Missing note falls back to list + warning — already handled inside the detail worker `:3254-3270`.)
- `conversations`: if `record_id` not among loaded conversation record ids, fetch `get_conversation_by_id` via `asyncio.to_thread`; `None`/missing db → `notify("Conversation is unavailable.", severity="warning")` and return; else prepend the record to `self._local_source_records["conversations"]`. Then set `_selected_conversation_id = record_id` and `await self._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS, "conversations")`. This closes the known deep-link caveat (id not in snapshot silently fell back to the first row).
- Unknown `source_type` or empty id: return quietly (no Open button was rendered; defensive only).

- [ ] **Step 1: Failing pilots:** (a) Open on a note result lands in the notes editor for that id (fake notes service records `get_note_detail` call with `note_id`); (b) Open on a media result flips to the media viewer + detail fetch called with the id; (c) Open on a conversation NOT in the snapshot fetches by id (real in-memory ChaChaNotes DB record) and selects it (`_selected_conversation_id` holds; preview shows its title); (d) a result without provenance renders NO Open button.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** panel button + screen handler:

```python
@on(Button.Pressed, ".library-rag-result-open")
async def open_library_rag_result(self, event: Button.Pressed) -> None:
    event.stop()
    index = self._trailing_index(event.button.id)
    rows = self._library_rag_results
    if index is None or not (0 <= index < len(rows)):
        return
    row = rows[index]
    await self._open_library_item_by_id(row.open_source_type, row.source_id)
```

- [ ] **Step 4: Run pilots + full `Tests/UI/test_library_shell.py`.**
- [ ] **Step 5: Commit** — `feat(library): per-result Open routes straight to media/notes/conversation detail`.

---

### Task 8: Count seams — DB + local services + scope services

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (three COUNT methods; mirror `count_notes` at `:6660` — `execute_query`, no transaction, Google docstring)
- Modify: `tldw_chatbook/Study_Interop/local_study_service.py`, `tldw_chatbook/Study_Interop/study_scope_service.py`, `tldw_chatbook/Study_Interop/local_quiz_service.py`, `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- Test: `Tests/ChaChaNotesDB/test_study_functionality.py` (DB), `Tests/Study_Interop/test_local_study_service.py`, `test_study_scope_service.py`, `test_local_quiz_service.py`, `test_quiz_scope_service.py`

**Interfaces:**
- Produces: `CharactersRAGDB.count_due_flashcards() -> int`, `CharactersRAGDB.count_decks() -> int`, `CharactersRAGDB.count_quizzes() -> int`; local-service passthroughs; `StudyScopeService.count_due_flashcards(*, mode=None) -> int`, `StudyScopeService.count_decks(*, mode=None) -> int`, `QuizScopeService.count_quizzes(*, mode=None) -> int` — local mode delegates to the local service; non-local mode raises `ValueError` (mirror `NotesScopeService.count_notes`'s scope handling at `Notes/notes_scope_service.py:727`; copy its exact async/sync + mode-resolution idiom from the sibling `list_*` methods in each file).

SQL (exact):

```sql
-- count_due_flashcards (WHERE clause identical to get_due_flashcards :7445)
SELECT COUNT(*) AS cnt FROM flashcards f JOIN decks d ON d.id = f.deck_id
WHERE f.is_deleted = 0 AND f.is_suspended = 0 AND d.is_deleted = 0
  AND (f.next_review IS NULL OR f.next_review <= CURRENT_TIMESTAMP)
-- count_decks
SELECT COUNT(*) AS cnt FROM decks WHERE is_deleted = 0
-- count_quizzes  (column is `deleted`, NOT `is_deleted`, on quizzes)
SELECT COUNT(*) AS cnt FROM quizzes WHERE deleted = 0
```

- [ ] **Step 1: Failing DB tests** (real in-memory DB, extend `test_study_functionality.py`): counts start 0; after creating a deck + 2 due cards + 1 suspended card → `count_due_flashcards() == 2`, `count_decks() == 1`; soft-deleting the deck → both drop to 0; quiz create/soft-delete moves `count_quizzes()`.
- [ ] **Step 2: Failing service tests** (extend existing fakes with the new DB methods, matching signatures): local services delegate; scope services return ints in local mode and raise `ValueError` for server/workspace mode (assert the message mirrors the notes-count wording).
- [ ] **Step 3: Run to verify failure. Step 4: Implement. Step 5: Run all five test files.**
- [ ] **Step 6: Commit** — `feat(study): exact count seams for due flashcards, decks, quizzes`.

---

### Task 9: Create-section rail counts

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py` (`LibraryShellInput` fields; create-row builders)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_list_local_source_snapshot` `:1005-1125`, `_apply_local_source_snapshot` `:764-780`, `_build_library_shell_input` `:3029-3072`)
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py` (`_count_suffix` bypass + emphasis class), `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ build_css)
- Test: `Tests/Library/test_library_shell_state.py`, `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: Task 8's scope-service count methods.
- Produces: `LibraryShellInput` fields `study_decks_count: int | None = None`, `flashcards_due_count: int | None = None`, `quizzes_count: int | None = None`; `LibraryRailRow` fields `count_display: str = ""` (verbatim suffix override) and `count_emphasis: str = ""` (`"bright"`/`"dim"`/`""`).

Row rendering contract (binding copy):
- `create-study` row: title stays `Study decks`, `count=study_decks_count` → `Study decks (3)`.
- `create-quizzes`: `Quizzes (2)`.
- `create-flashcards`: title stays `Flashcards`; when `flashcards_due_count is not None`, `count_display=f" due: {n}"` → renders `Flashcards due: 12`; `count_emphasis="bright"` when n>0 else `"dim"`. Widget: `suffix = row.count_display or self._count_suffix(row.count, row.count_known)`; emphasis adds class `library-rail-row-due-bright` / `library-rail-row-due-dim` to the row button (CSS: bright follows the accent convention used by `.console-action-primary`-adjacent styles; dim uses the muted text color already used for rail hints — match existing variables in `_agentic_terminal.tcss`, do not invent colors).
- Any count `None` (service absent/failed) → row renders with NO count, no emphasis (quiet degrade; spec: rows remain handoffs either way).

Loader: in `_list_local_source_snapshot`, resolve `study_scope_service`/`study_quiz_scope_service`, `getattr(..., "count_due_flashcards"/"count_decks"/"count_quizzes", None)`; add callables to the existing `asyncio.gather` (same 5s `wait_for` envelope); each failure → `None` count (log debug, never surface error copy for study counts — the three browse sources keep their existing error semantics). Thread results through `_apply_local_source_snapshot` storage → `_build_library_shell_input`.

- [ ] **Step 1: Failing pure tests:** `build_library_shell_state` with `flashcards_due_count=12` → flashcards row `count_display == " due: 12"`, `count_emphasis == "bright"`; `=0` → `"dim"`; `None` → both empty; decks/quizzes counts land in `count` with `count_known=True`.
- [ ] **Step 2: Failing pilot:** fake app with study/quiz scope services exposing the count methods (7 due, 3 decks, 2 quizzes) → rail renders `Flashcards due: 7`, `Study decks (3)`, `Quizzes (2)`; a fake WITHOUT the count methods renders the rows uncounted (no crash, no error copy).
- [ ] **Step 3: Run to verify failure. Step 4: Implement (including `./build_css.sh`; commit both CSS files). Step 5: Run both test files.**
- [ ] **Step 6: Commit** — `feat(library): live create-section counts (flashcards due, decks, quizzes)`.

---

### Task 10: Home due-mirror → Study flashcards surface

**Files:**
- Modify: `tldw_chatbook/Home/dashboard_state.py` (`HomeDashboardInput`, `build_home_triage_state`, `build_home_controls`)
- Modify: `tldw_chatbook/Home/active_work_adapter.py` (due-count snapshot)
- Modify: `tldw_chatbook/UI/Screens/home_screen.py` (`HOME_CONTROL_METHODS`, mount worker)
- Modify: `tldw_chatbook/app.py` (`open_home_flashcards_review`, adapter provider wiring)
- Test: `Tests/Home/test_dashboard_state.py`, `Tests/Home/test_active_work_adapter.py`, plus the Home pilot file (grep `HomeScreen` under `Tests/UI/` and extend the existing one)

**Interfaces:**
- Consumes: Task 8's `CharactersRAGDB.count_due_flashcards`; `app.open_study_screen(scope_context=None, *, initial_section=None)` (`app.py:1660`); control dispatch (`home_screen.py:331-355` — a control's method is called with NO kwargs when `target_id is None` and the control is not in `HOME_CONTROL_METHODS_WITH_TARGET_ROUTE`).
- Produces: `HomeDashboardInput.flashcards_due_count: int = 0`; constant `HOME_FLASHCARDS_DUE_ROW_ID = "home-flashcards-due"`; control `home-review-flashcards` (label `Review flashcards`, route `study`, category `flashcards_due`, target_id `None`); `app.open_home_flashcards_review()`; adapter method `refresh_flashcards_due_snapshot()` + ctor param `flashcards_due_provider: Callable[[], int | None] | None = None`.

Behavior:
- `build_home_triage_state`: when `state.flashcards_due_count > 0`, append to `attention_rows` (after the item loop, before sections are built):

```python
if state.flashcards_due_count > 0:
    attention_rows.append(
        HomeRailRow(
            row_id=HOME_FLASHCARDS_DUE_ROW_ID,
            section_id="attention",
            glyph="●",
            title=f"Flashcards due: {state.flashcards_due_count}",
            age_label="",
            source="Library",
            status_category="due",
            detail_route="study",
        )
    )
```

- Canvas guard (the existing `next(...)` at `dashboard_state.py:684-688` would raise `StopIteration` for this synthetic row): inside `if selected is not None:`, branch FIRST on `selected.row_id == HOME_FLASHCARDS_DUE_ROW_ID` and build `HomeCanvasState(title=f"Flashcards due: {state.flashcards_due_count}", lines=("Source: Library · Status: due for review", "Route: study"), actions=build_home_controls(state), next_action=next_action, next_action_is_canvas=False)`.
- `build_home_controls`: append at the END (before `return`) — `if state.flashcards_due_count > 0: controls.append(HomeControl("home-review-flashcards", "Review flashcards", "study", "flashcards_due", None))` (end placement keeps every existing control-order assertion green).
- `home_screen.py`: `HOME_CONTROL_METHODS["home-review-flashcards"] = "open_home_flashcards_review"`; extend the existing mount thread-worker `_refresh_home_chatbook_artifact_snapshot` (`:119-126`) to also call `adapter.refresh_flashcards_due_snapshot` when callable (this IS the spec's "Home's normal entry/refresh cadence").
- Adapter: cache field `self._flashcards_due_count: int = 0`; `refresh_flashcards_due_snapshot()` calls the provider (exceptions → keep 0, log debug); `build_dashboard_input` passes `flashcards_due_count=self._flashcards_due_count`. Provider absent or returning `None` → 0 → no row (spec: service absent → no row).
- `app.py`:

```python
def open_home_flashcards_review(self) -> None:
    """Open the Study screen directly on the flashcards review surface."""
    self.open_study_screen(initial_section="flashcards")

def _local_flashcards_due_count(self) -> int | None:
    """Count due flashcards for the Home mirror; None when the DB is absent."""
    db = getattr(self, "chachanotes_db", None)
    counter = getattr(db, "count_due_flashcards", None)
    if not callable(counter):
        return None
    try:
        return int(counter())
    except Exception:
        logger.debug("Home flashcards-due count failed.", exc_info=True)
        return None
```

wire `flashcards_due_provider=self._local_flashcards_due_count` where the adapter is constructed.

- [ ] **Step 1: Failing pure tests** (`test_dashboard_state.py`): count 12 → attention section contains a row titled `Flashcards due: 12`, source `Library`, route `study`; count 0 → no such row; selecting the row yields a canvas titled `Flashcards due: 12` (no `StopIteration`); controls include `home-review-flashcards` when count>0, absent at 0.
- [ ] **Step 2: Failing adapter test** (`test_active_work_adapter.py`): provider returning 7 → after `refresh_flashcards_due_snapshot()`, `build_dashboard_input(...).flashcards_due_count == 7`; provider raising → 0; no provider → 0.
- [ ] **Step 3: Failing pilot:** Home with a test input `flashcards_due_count=12` → rail shows the row; selecting it and pressing the `home-review-flashcards` control calls a recorded `open_home_flashcards_review` (fake app records; or assert `pending_study_initial_section == "flashcards"` + navigation posted with real app helper).
- [ ] **Step 4: Run to verify failure. Step 5: Implement. Step 6: Run all Home tests + the pilot file.**
- [ ] **Step 7: Commit** — `feat(home): flashcards-due Needs Attention row routes one hop to Study flashcards`.

---

### Task 11 (LEAD-EXECUTED): Live QA captures + evidence README

**Files:**
- Modify/Create: `Docs/superpowers/qa/library-l3a-2026-07/README.md` + captures

- [ ] **Step 1:** Full local gate first: `Tests/Library/ Tests/UI/test_library_shell.py Tests/Home/ Tests/Study_Interop/ Tests/ChaChaNotesDB/test_study_functionality.py Tests/UI/test_product_maturity_gate16_library_search_rag.py` green; `python -c "import tldw_chatbook.app"` OK.
- [ ] **Step 2:** Seeded textual-serve QA at 2050x1240 dsf1 (populated states — L2 lesson: empty-state captures hide row defects). Required captures: search canvas idle w/ history expanded; searching status line; results with Open buttons; result→Open landed in notes editor and media viewer; rail counts row set (`Flashcards due: N` bright + dim variants); Home Needs Attention due row; Study screen landed on flashcards after `Review flashcards`.
- [ ] **Step 3:** README lists each capture + the follow-ups carried (service-backed conversations FTS filter; saved searches; rail-badge refresh at sync completion).
- [ ] **Step 4:** Commit QA. Then STOP: present captures to the user for the explicit screenshot approval gate. NO merge without it.

---

## Task order & dependencies

1 (smoke) → 2 → 3 → 4 → 5 → 6 → 7 (search track, strictly ordered) ; 8 → 9 (counts track; independent of 2-7, may interleave after 4) ; 8 → 10 (Home track) ; 11 last. Single implementer at a time (SDD rule) — run 2..7 then 8,9,10.

## Self-review notes

- Spec coverage: rail flip (T4), smoke-first (T1), single query truth + rail rewire (T6), conversations filter (T6), history (T2/T5), result→Open per resolvable id (T2/T7), RAG degrade verified + real backend (T3, planning verification recorded: NO production service existed; FTS seams chosen; `_rag_service` presence is the gate), shared open-by-id (T7), create counts (T8/T9), Home due-mirror one-hop (T10), QA gate (T11).
- Deviations from spec text (deliberate, flag at review): spec's `build_home_dashboard_state` does not exist — the real seam is `HomeDashboardInput`/`build_home_triage_state`; spec's "capped fetch renders N+" for create counts is moot because exact COUNT seams are added (degrade is count-absent, not capped); Home row title omits the `· Library` suffix in favor of the rail's existing `source="Library"` column which renders identically to other rows.
- Type consistency: `_open_library_item_by_id(source_type: str, record_id: str)` consumed by T7 panel handler; count fields named identically across `LibraryShellInput`/loader/builders; `HOME_FLASHCARDS_DUE_ROW_ID` shared between dashboard_state and tests.
