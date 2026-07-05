# Library Shell (L1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Library screen's nine mode chips and Content Hub summary with the Console-grammar shell — a rail (search input, Browse/Create/Ingest/Details sections with live counts, persisted prefs) driving a canvas whose first rebuilt surface is Browse ▸ Conversations — per spec §2/§5 (L1).

**Architecture:** Pure state first: new `Library/library_shell_state.py` (rail sections/rows/canvas targets) and `Library/library_conversations_state.py` (browse rows + preview) consume counts/records the screen already fetches via `_list_local_source_snapshot()`. New `Widgets/Library/` package renders the rail (with rail-top search) and the conversations canvas. `library_screen.py` keeps its per-mode middle-pane builders for not-yet-rebuilt surfaces (collections, study, flashcards, quizzes, import-export, search) — rail rows swap them into the canvas region — but the mode strip, hub summary, and three-pane contract grid are retired.

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio, existing `_build_test_app` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md` §2, §3–§5 (L1), resolved decisions §6. Reference implementation for every convention: the H1 Home work on this branch (`tldw_chatbook/Widgets/Home/`, `Home/home_rail_state.py`, `Tests/UI/test_home_triage_rail.py`).

## Global Constraints

- Run tests with: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short` (venv at repo main root `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv`).
- The `timeout` shell command is not available.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- Rail conventions (binding, from Console/H1): `ConsoleRailSectionHeader` (from `tldw_chatbook.Widgets.Console.console_rail_section`) for headers with `-`/`+` glyphs; `▸` marker on the selected row; rows are two-line compact Buttons, height 2, titles truncated at 20 chars with `...` and the full title as tooltip (see `Widgets/Home/home_rail.py::_visible_row_title`); ages via `format_console_relative_age` (from `tldw_chatbook.Workspaces.conversation_browser_state`); `.ds-toolbar` action rows are `Horizontal` containers, never a class on buttons.
- Section titles exactly: `Browse`, `Create`, `Ingest`, `Details`. Header line exactly: `Library | Local` or `Library | Server: <label>`.
- Canvas landing/empty copy exactly: `Search, pick a content type, or ingest something new.`
- Rail prefs persisted under `library.rail_state` (global scope, one key `sections`), defaults: browse/create/ingest open, details collapsed.
- Behavior contracts that MUST survive: every surface reachable from today's mode chips stays reachable (collections/study/flashcards/quizzes/import-export/search render in the canvas; media/notes/ingest route to their screens); the existing conversation → Console handoff keeps working; `NavigateToScreen` routing unchanged.
- Due-counts on Create rows are L3 scope — Create rows in L1 carry no counters.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Library screen changes require live screenshot QA at viewport 2050x1240 (device_scale_factor 1) + explicit user approval before merge (Task 8).
- Work in the `claude/library-shell-l1` branch (stacked on `claude/home-library-redesign`; worktree `.claude/worktrees/home-library-redesign`).

## File Structure

- Create `tldw_chatbook/Library/library_shell_state.py` — pure rail/canvas-target builders.
- Create `tldw_chatbook/Library/library_rail_state.py` — rail preferences (pure; copy the `Home/home_rail_state.py` pattern, do not import it).
- Create `tldw_chatbook/Library/library_conversations_state.py` — pure Browse ▸ Conversations rows + preview.
- Create `tldw_chatbook/Widgets/Library/__init__.py`, `library_rail.py`, `library_conversations_canvas.py`.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — compose rework (header line + rail + canvas host), selection dispatch, prefs wiring, search wiring; retire `DestinationModeStrip` usage, `_content_hub_rows`, and the old `library-contract-grid` three-pane layout.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate `tldw_cli_modular.tcss`).
- Tests: create `Tests/Library/test_library_shell_state.py`, `Tests/Library/test_library_rail_state.py`, `Tests/Library/test_library_conversations_state.py`, `Tests/UI/test_library_shell.py`; update the stale legacy suites listed in Task 7.

---

### Task 1: Pure rail/shell state builders

**Files:**
- Create: `tldw_chatbook/Library/library_shell_state.py`
- Test: `Tests/Library/test_library_shell_state.py`

**Interfaces:**
- Consumes: nothing app-side (pure).
- Produces (later tasks rely on exact names):

```python
@dataclass(frozen=True)
class LibraryRailRow:
    row_id: str            # e.g. "browse-conversations"
    section_id: str        # "browse" | "create" | "ingest"
    title: str             # e.g. "Conversations"
    target_kind: str       # "canvas" | "mode" | "screen"
    target_id: str         # "conversations" | mode name | TAB_* value
    count: int | None = None
    count_known: bool = True

@dataclass(frozen=True)
class LibraryRailSectionState:
    section_id: str
    title: str
    rows: tuple[LibraryRailRow, ...]

@dataclass(frozen=True)
class LibraryShellInput:
    media_count: int | None = None
    media_known: bool = True
    conversations_count: int | None = None
    conversations_known: bool = True
    notes_count: int | None = None
    notes_known: bool = True
    collections_count: int | None = None
    collections_known: bool = True
    runtime_source: str = "local"      # "local" | "server"
    server_label: str | None = None
    details_lines: tuple[str, ...] = ()

@dataclass(frozen=True)
class LibraryShellState:
    header_line: str
    sections: tuple[LibraryRailSectionState, ...]
    details_lines: tuple[str, ...]
    selected_row_id: str
    canvas_kind: str       # "empty" | "conversations" | "mode"
    canvas_target: str     # "" | mode name
    canvas_empty_copy: str

def build_library_shell_state(
    state: LibraryShellInput, *, selected_row_id: str = ""
) -> LibraryShellState: ...
```

Fixed row table (order binding; `TAB_MEDIA`/`TAB_NOTES`/`TAB_INGEST` imported from `tldw_chatbook.Constants`):

| row_id | section | title | target_kind | target_id | count source |
|---|---|---|---|---|---|
| browse-media | browse | Media | screen | TAB_MEDIA | media_count |
| browse-conversations | browse | Conversations | canvas | conversations | conversations_count |
| browse-notes | browse | Notes | screen | TAB_NOTES | notes_count |
| browse-collections | browse | Collections | mode | collections | collections_count |
| browse-search | browse | Search / RAG | mode | search | — |
| create-note | create | New note | screen | TAB_NOTES | — |
| create-study | create | Study decks | mode | study | — |
| create-flashcards | create | Flashcards | mode | flashcards | — |
| create-quizzes | create | Quizzes | mode | quizzes | — |
| ingest-import-media | ingest | Import media | screen | TAB_INGEST | — |
| ingest-import-export | ingest | Import / Export | mode | import-export | — |

Rules: header_line = `"Library | Local"` when runtime_source is "local", else `f"Library | Server: {server_label or 'unknown'}"`. `selected_row_id` resolves canvas via the row's target: `canvas` rows → (`"conversations"`, `""`); `mode` rows → (`"mode"`, target_id); `screen` rows and unknown/empty ids → (`"empty"`, `""`) with `canvas_empty_copy` set to the landing copy (screen rows navigate on press and never become the selection — Task 6). `count=None` or unknown never crashes; the widget renders `(n)` when known, `(n+)` when `count_known` is False, and no suffix when count is None.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_library_shell_state.py`:

```python
from tldw_chatbook.Constants import TAB_INGEST, TAB_MEDIA, TAB_NOTES
from tldw_chatbook.Library.library_shell_state import (
    LibraryShellInput,
    build_library_shell_state,
)


def test_shell_sections_rows_and_targets_are_fixed():
    shell = build_library_shell_state(LibraryShellInput(
        media_count=17, conversations_count=128, notes_count=42, collections_count=5,
    ))
    assert shell.header_line == "Library | Local"
    assert [s.section_id for s in shell.sections] == ["browse", "create", "ingest"]
    browse = shell.sections[0]
    assert [r.row_id for r in browse.rows] == [
        "browse-media", "browse-conversations", "browse-notes", "browse-collections",
        "browse-search",
    ]
    assert browse.rows[4].target_kind == "mode" and browse.rows[4].target_id == "search"
    assert browse.rows[4].count is None
    conv = browse.rows[1]
    assert (conv.target_kind, conv.target_id, conv.count) == ("canvas", "conversations", 128)
    media = browse.rows[0]
    assert (media.target_kind, media.target_id) == ("screen", TAB_MEDIA)
    create_ids = [r.row_id for r in shell.sections[1].rows]
    assert create_ids == ["create-note", "create-study", "create-flashcards", "create-quizzes"]
    assert shell.sections[1].rows[0].target_id == TAB_NOTES
    ingest = shell.sections[2]
    assert ingest.rows[0].target_id == TAB_INGEST
    assert ingest.rows[1] == ingest.rows[1]  # frozen dataclass equality sanity
    assert all(r.count is None for r in shell.sections[1].rows)


def test_empty_selection_yields_landing_canvas():
    shell = build_library_shell_state(LibraryShellInput())
    assert shell.canvas_kind == "empty"
    assert shell.canvas_empty_copy == "Search, pick a content type, or ingest something new."


def test_conversations_selection_yields_conversations_canvas():
    shell = build_library_shell_state(
        LibraryShellInput(conversations_count=3), selected_row_id="browse-conversations"
    )
    assert (shell.canvas_kind, shell.canvas_target) == ("conversations", "")
    assert shell.selected_row_id == "browse-conversations"


def test_mode_selection_yields_mode_canvas():
    shell = build_library_shell_state(LibraryShellInput(), selected_row_id="create-flashcards")
    assert (shell.canvas_kind, shell.canvas_target) == ("mode", "flashcards")


def test_screen_and_unknown_rows_resolve_to_empty_canvas():
    for row_id in ("browse-media", "create-note", "nope", ""):
        shell = build_library_shell_state(LibraryShellInput(), selected_row_id=row_id)
        assert shell.canvas_kind == "empty", row_id


def test_server_header_line():
    shell = build_library_shell_state(
        LibraryShellInput(runtime_source="server", server_label="lab-box")
    )
    assert shell.header_line == "Library | Server: lab-box"
```

- [ ] **Step 2: Run to verify RED** — `env HOME=... .venv/bin/python -m pytest -q Tests/Library/test_library_shell_state.py --tb=short`. Expected: ImportError (module missing).

- [ ] **Step 3: Implement** `tldw_chatbook/Library/library_shell_state.py` exactly per the Interfaces block and row table. `LIBRARY_CANVAS_LANDING_COPY = "Search, pick a content type, or ingest something new."` as a module constant. Build the row tuple once per call (counts vary); resolve selection by scanning rows for `row_id == selected_row_id`.

- [ ] **Step 4: Verify GREEN.**

- [ ] **Step 5: Commit** `feat(library): pure shell rail state builders`.

---

### Task 2: Rail preferences module

**Files:**
- Create: `tldw_chatbook/Library/library_rail_state.py`
- Test: `Tests/Library/test_library_rail_state.py`

**Interfaces:**
- Produces: `LibraryRailPreferences(browse_open: bool = True, create_open: bool = True, ingest_open: bool = True, details_open: bool = False)` (frozen dataclass), `coerce_library_rail_preferences(raw: Any) -> LibraryRailPreferences`, `serialize_library_rail_preferences(preferences) -> dict[str, bool]`.

- [ ] **Step 1: Failing tests** — copy the structure of `Tests/Home/test_home_rail_state.py` adapted to the four Library fields: round-trip serialize→coerce, missing-key defaults, `"off"`/`"false"` string falsiness, non-mapping input → defaults.
- [ ] **Step 2: Verify RED** (ImportError).
- [ ] **Step 3: Implement** — copy the coerce/serialize pattern from `tldw_chatbook/Home/home_rail_state.py` (`_coerce_bool`, true/false string sets) with the four Library fields. Do not import from home_rail_state or console_rail_state (screen independence; the pattern is ~30 lines).
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit** `feat(library): rail section preferences`.

---

### Task 3: Pure Browse ▸ Conversations state

**Files:**
- Create: `tldw_chatbook/Library/library_conversations_state.py`
- Test: `Tests/Library/test_library_conversations_state.py`

**Interfaces:**
- Consumes: `format_console_relative_age` from `tldw_chatbook.Workspaces.conversation_browser_state` (age grammar: "now"/"2m"/"1h"/"3d"…).
- Produces:

```python
@dataclass(frozen=True)
class LibraryConversationRow:
    conversation_id: str
    title: str
    secondary: str        # "{n} messages - {age}" (or "{n} messages" when no age)
    selected: bool = False

@dataclass(frozen=True)
class LibraryConversationsCanvasState:
    rows: tuple[LibraryConversationRow, ...]
    status_copy: str      # "" or "N matches for 'q'" when query set
    empty_copy: str       # set only when rows empty
    selected_id: str
    preview_lines: tuple[str, ...]   # for the selected row
    query: str

def build_library_conversations_state(
    records: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
) -> LibraryConversationsCanvasState: ...
```

Rules: record id from first present of keys `("id", "conversation_id", "uuid")` (str-cast); title from `record.get("title")` stripped, fallback `"Untitled conversation"`; updated timestamp from first present of `("updated_at", "last_updated", "last_modified", "updated")`; message count from first present of `("message_count", "messages_count", "messageCount", "message_total", "messages_total")` (mirrors `library_screen._conversation_message_count_label`), fallback secondary `"conversation"`. Sort by updated timestamp descending (missing timestamps last), truncate to `limit`. `query` filters case-insensitive substring on title BEFORE limiting; when query set, `status_copy = f"{len(filtered)} matches for '{query}'"` (singular "match" for 1). `selected_id` not in filtered rows → select the first row (its id becomes `selected_id`); no rows → `selected_id=""` and `empty_copy = "No saved conversations yet. Save a Console chat and it appears here."` (query-empty case) or `f"No conversations match '{query}'."` (query case). `preview_lines` for the selected record: `(title, f"Messages: {count_or_'unknown'}", f"Updated: {age_or_'unknown'}")`.

- [ ] **Step 1: Failing tests** — cover: ordering + age labels (use fixed `now=datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)` and records 3m/2h/none old); query filtering + status copy + no-match empty copy; selected-id fallback to first row; preview lines content; limit truncation; id/title/count key fallbacks (one record using `conversation_id`+`messages_total`).
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** per rules above (pure; no Textual imports).
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit** `feat(library): pure browse-conversations canvas state`.

---

### Task 4: Rail + conversations canvas widgets

**Files:**
- Create: `tldw_chatbook/Widgets/Library/__init__.py`, `tldw_chatbook/Widgets/Library/library_rail.py`, `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`

**Interfaces:**
- Consumes: Task 1–3 dataclasses; `ConsoleRailSectionHeader`; the H1 row conventions from `Widgets/Home/home_rail.py` (`_visible_row_title` 20-char truncation — copy the helper, do not import Home's).
- Produces:
  - `LibraryRail(Vertical)` — `__init__(self, shell: LibraryShellState, preferences: LibraryRailPreferences, **kwargs)`, `sync_state(shell, preferences)`. Composes: `Input(placeholder="Search conversations…", id="library-search-input")` at top; per section a `ConsoleRailSectionHeader(title, section_id=f"library-{section_id}", open=..., id=f"library-rail-section-header-{section_id}")` + body `Vertical(id=f"library-rail-section-body-{section_id}", classes="library-rail-section-body")` of row Buttons; then the Details header + body of `details_lines` Statics. Row buttons: id `f"library-row-{row.row_id}"`, classes `"library-rail-row"` (+ `"library-rail-row-selected"` when `row.row_id == shell.selected_row_id`), compact, height 2, label `f"{marker} {visible_title}{count_suffix}\n    {section_hint}"` — marker `▸`/space; count_suffix `f" ({row.count})"` known / `f" ({row.count}+)"` unknown / `""` when None; second line: `"open list"` for canvas/mode rows, `"opens {target screen title}"` for screen rows is over-specified — use exactly `row.target_kind == "screen" and "opens screen" or "in Library"`. Tooltip = full title. Button attrs: `button.row_id`, `button.target_kind`, `button.target_id` (screen dispatch reads these).
  - `LibraryConversationsCanvas(Vertical)` — `__init__(self, canvas: LibraryConversationsCanvasState, **kwargs)`, `sync_state(canvas)`. Composes: `Static(id="library-conversations-status")` (status/empty copy, hidden when blank); list `Vertical(id="library-conversations-list")` of row Buttons id `f"library-conversation-row-{index}"`, classes `"library-conversation-row"` (+ `-selected`), compact, height 2, label `f"{marker} {visible_title}\n    {row.secondary}"`, tooltip full title, attr `button.conversation_id`; preview `Vertical(id="library-conversation-preview")` with `Static(id="library-conversation-preview-lines")` joining preview_lines and an action row `Horizontal(classes="ds-toolbar")` containing `Button("Open in Console", id="library-conversation-open-console", classes="library-canvas-action", compact=True)`. Preview hidden when no selection.

No dedicated widget-only test task — Task 5's pilot tests cover both widgets under the real stylesheet (H1 precedent: widgets and screen land together, tests at the pilot level).

- [ ] **Step 1: Implement both widgets** per Interfaces (they cannot be tested before the screen hosts them; the RED step lives in Task 5).
- [ ] **Step 2: Commit** `feat(library): rail and conversations canvas widgets` (widgets only; screen wiring next).

---

### Task 5: Screen rework — shell compose, selection dispatch, prefs, search

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: create `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4; existing screen internals: `_list_local_source_snapshot()` (counts + records), `_conversation_records()`, `_source_record_id`, the per-mode middle-pane composition currently inlined in `compose_content` (lines ~2582–2845) and mode metadata dicts (~lines 94–177); `save_setting_to_cli_config`; `NavigateToScreen`.
- Produces: reworked `LibraryScreen` — new compose contract:
  - `Static(header_line, id="library-header-line")`
  - `Horizontal(id="library-shell-grid")` → `LibraryRail` (width `3fr`, min 24) + canvas host `Vertical(id="library-canvas")` (width `13fr`, min 40).
  - Canvas host renders by `shell.canvas_kind`: `"empty"` → `Static(canvas_empty_copy, id="library-canvas-landing")`; `"conversations"` → `LibraryConversationsCanvas`; `"mode"` → the existing middle-pane content for that mode (extract the current per-mode composition into `def _compose_mode_canvas(self, mode: str) -> ComposeResult` during this task — move, don't rewrite; keep every existing widget id inside those mode bodies so their suites keep passing).
  - Selection state `self._library_selected_row_id: str = ""`; conversations sub-state `self._library_conversation_query: str = ""`, `self._library_selected_conversation_id` (reuse the existing attr).

- [ ] **Step 1: Failing pilot tests** in `Tests/UI/test_library_shell.py`. Harness: copy the `HomeHarness` pattern from `Tests/UI/test_home_screen.py` (CSS_PATH to `tldw_cli_modular.tcss`, push `LibraryScreen(app_instance)`), `_build_test_app` from `Tests/UI/test_screen_navigation.py`. Seed records/counts the way the existing Library suites do (see how `Tests/UI/test_library_content_hub.py` injects `_local_source_records`/`_local_source_counts` or service fakes — reuse that mechanism verbatim). Tests:

```python
# 1. shell renders: header line "Library | Local", rail sections Browse/Create/Ingest
#    with counts ("Conversations (2)"), Details header present, landing canvas copy
#    when nothing selected, NO "#library-mode-bar" and NO "#library-contract-grid"
#    and NO hub summary ids ("#library-notes-summary") anywhere.
# 2. pressing "#library-row-browse-conversations" renders the conversations canvas:
#    two rows, first selected (▸), preview lines include "Messages:", and
#    "#library-conversation-open-console" present.
# 3. pressing a conversation row switches selection and preview.
# 4. pressing "#library-conversation-open-console" triggers the existing
#    Console handoff (assert via the same message/nav capture the legacy
#    conversations-mode test used in Tests/UI/test_library_content_hub.py or
#    the old browser suite — preserve that contract exactly).
# 5. pressing "#library-row-create-flashcards" renders the legacy flashcards
#    mode content inside "#library-canvas" (assert one stable widget id from
#    that mode body).
# 6. pressing "#library-row-browse-media" posts NavigateToScreen("media")
#    (screen rows navigate, selection unchanged — landing canvas still shown).
# 7. typing "quarterly" into "#library-search-input" + Enter renders the
#    conversations canvas filtered with status copy "1 match for 'quarterly'".
# 8. Details toggle press flips "#library-rail-section-body-details" display and
#    persists {"library": {"rail_state": {"sections": {"details_open": True}}}}
#    via app_config (mirror Tests/UI/test_home_triage_rail.py::test_home_details_toggle_persists).
```

Write all eight as real tests with real assertions (the comment block above defines intent; the H1 file shows the mechanics).

- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** the rework:
  - Replace `compose_content`'s title/purpose/status header, `DestinationModeStrip`, and `library-contract-grid` three-pane body with the shell contract above. Delete `_content_hub_rows` and the "sources" hub mode (landing canvas replaces it). Extract per-mode middle-pane composition to `_compose_mode_canvas` (move code; keep ids).
  - Row press handler: `target_kind == "screen"` → post the same `NavigateToScreen` the legacy chips/actions used; `"canvas"`/`"mode"` → set `_library_selected_row_id`, rebuild shell state, `sync_state`/recompose canvas.
  - Search: `Input.Submitted` on `#library-search-input` → set `_library_conversation_query`, force `_library_selected_row_id = "browse-conversations"`, refresh canvas.
  - Prefs: read via `coerce_library_rail_preferences(self.app_instance.app_config.get("library", {}).get("rail_state", {}).get("sections", {}))`; on section toggle, save via `save_setting_to_cli_config("library.rail_state", "sections", serialize_library_rail_preferences(prefs))` — copy the exact call shape from `home_screen.py`'s Details-toggle handler.
  - Counts: from `self._local_source_counts` + `self._local_source_total_known` (media/conversations/notes) and the collections count the collections mode already computes (reuse its service/state accessor — locate it in `Library/library_collections_service.py` at implementation time; if no cheap count exists, pass `collections_count=None` rather than inventing a query).
- [ ] **Step 4: Run** the new file + `Tests/Library/` — all green — then the legacy Library UI suites (`Tests/UI/test_library_content_hub.py`, `test_product_maturity_phase3_library_contract_layout.py`, `test_product_maturity_phase3_library_study_context.py`, `test_product_maturity_gate16_library_search_rag.py`, `test_post_release_workspaces_library_depth.py`, `test_personas_library_pane.py`). Expected: legacy suites partially RED (they assert the strip/hub) — that's Task 7's job; do not fix them here beyond confirming the failures are assertion-staleness, not crashes.
- [ ] **Step 5: Commit** `feat(library): shell rail and conversations canvas replace chips and hub`.

---

### Task 6: CSS

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`, regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: extend `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Failing presence test** — mirror `test_generated_stylesheet_includes_home_triage_rules` from `Tests/UI/test_home_triage_rail.py`: selectors `#library-shell-grid`, `#library-header-line`, `.library-rail-row`, `.library-rail-row-selected`, `.library-conversation-row`, `.library-canvas-action`, `#library-search-input` present in component + generated CSS; stale selectors `#library-mode-bar`, `#library-contract-grid` absent from both.
- [ ] **Step 2: RED**, then add the rules — copy the H1 blocks verbatim with renamed selectors: `.library-rail-row`/`.library-conversation-row` get the `.home-rail-row` block (left align, `content-align: left middle`, `text-wrap: nowrap`, `text-overflow: ellipsis`, `margin: 0 0 1 0`, height 2); `-selected` variants get `$ds-focus-bg`/`$ds-focus-fg` + bold underline; `.library-canvas-action` copies `.home-canvas-action`; `#library-shell-grid`/`#library-header-line` copy `#home-triage-grid`/`#home-header-line`; `#library-search-input { margin: 0 0 1 0; }`. Remove the now-dead `#library-mode-bar`/`#library-contract-grid` rules.
- [ ] **Step 3: `./build_css.sh`**, run the presence test GREEN.
- [ ] **Step 4: Commit** `style(library): shell rail styles, retire mode-strip and contract-grid rules` (both tcss files).

---

### Task 7: Legacy suite rework

**Files:**
- Modify: the six legacy UI suites listed in Task 5 Step 4.

- [ ] **Step 1:** For each failing test, update assertions preserving intent: hub-summary assertions become rail-count assertions (`"Conversations (2)"` in rail text); mode-strip navigation assertions become rail-row presses (`#library-row-create-flashcards` etc.); three-pane layout assertions become shell-grid assertions (`#library-shell-grid`, `#library-canvas`); mode-content assertions (study handoff copy, search-rag status, collections rows, workspace depth) stay pointed at the SAME inner widget ids now rendered inside `#library-canvas`. Delete only tests whose entire subject was retired (the hub summary table itself, the mode strip chip list) — and note each deletion in the commit message.
- [ ] **Step 2:** Full run: `Tests/Library/ Tests/UI/test_library_shell.py` + all six legacy suites + `Tests/UI/test_home_triage_rail.py` + `Tests/UI/test_master_shell_design_system_contract.py`. All green.
- [ ] **Step 3: Commit** `test(library): re-anchor legacy suites on the shell rail contract`.

---

### Task 8: Verification, screenshot QA, approval gate

- [ ] **Step 1:** Full affected run (everything in Task 7 Step 2) green; note counts.
- [ ] **Step 2:** Live captures to `Docs/superpowers/qa/library-shell-l1-2026-07/` — textual-serve + playwright bundled chromium, viewport **2050x1240, device_scale_factor 1**, fresh isolated HOME, `wait_until="commit"`, wait then click the Library tab in the top bar. Required captures: (1) fresh landing (rail counts at zero-state, landing canvas copy); (2) populated Browse ▸ Conversations (seed real conversations into the isolated HOME's ChaChaNotes DB before serving — real data preferred over overrides; the H1 seeded-launcher pattern in the session scratchpad shows the serve shape); (3) conversation selection switch; (4) a legacy mode inside the canvas (flashcards row) proving reachability survived; (5) Details expanded. Include populated states — empty states hide row-rendering defects (H1 lesson).
- [ ] **Step 3:** Write the QA README (branch, recipe, per-capture description), commit `docs(library): shell L1 QA evidence`.
- [ ] **Step 4:** Present captures to the user for explicit approval before merge (standing rule). Do not open the PR before approval.

---

## Self-review notes

- Spec coverage: rail sections/counts/prefs/search input (Tasks 1–2, 4–5), Browse ▸ Conversations canvas (Tasks 3–5), chips and hub retired (Tasks 5–7), landing copy verbatim (Task 1), Console conventions (Tasks 4, 6), testing pattern + approval gate (Tasks 5, 7, 8). Search is conversations-scoped in L1 (placeholder says so honestly); cross-content search and the RAG canvas are L3 per §5.
- Reachability audit (chips → new homes): Content Hub→landing canvas; Conversations→browse-conversations; Search/RAG→browse-search (mode canvas; the rail-top input covers the quick path, conversations-scoped in L1; gate16 re-anchors on `#library-row-browse-search` in Task 7); Collections→browse-collections; Study/Flashcards/Quizzes→create rows; Import/Export→ingest-import-export; Workspaces→Details (depth lines feed `details_lines`; the full mode body stays reachable inside the Details body via the existing widgets — Task 5 extraction covers it).
- Type consistency: `LibraryShellState.canvas_kind`/`canvas_target` names match Tasks 1, 5; prefs function names match Tasks 2, 5; `library-row-{row_id}` id shape matches Tasks 4, 5, 7.
