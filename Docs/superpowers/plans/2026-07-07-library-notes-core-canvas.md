# Library L2b.1 — Notes Core Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Browse ▸ Notes becomes an in-Library canvas — list rows → in-canvas editor with autosave — replacing the route to the standalone Notes screen (deprecation itself is L2b.2).

**Architecture:** Mirror the shipped media canvas/viewer pattern exactly: a pure display-state module (`Library/library_notes_state.py`, no Textual imports), one posting-style widget (`Widgets/Library/library_notes_canvas.py`), and orchestration-only wiring in `library_screen.py` with `_library_notes_view` list/editor mode state. All service calls go through `notes_scope_service` offloaded via `_run_library_service_call(..., isolate_in_worker=True)`.

**Tech Stack:** Python ≥3.11, Textual, pytest (+pytest-asyncio pilots), real in-memory ChaChaNotes DB for mutation tests.

## Global Constraints (from the spec — binding on every task)

- Spec: `Docs/superpowers/specs/2026-07-07-library-l2b-l3-design.md` (Phase L2b.1 + Global constraints). Work on branch `claude/library-l2b` in worktree `.claude/worktrees/library-l2`.
- Run tests with: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q <target> --tb=short`. The `timeout` shell command is not available.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- RENDERING RULES (binding): stacked full-width widgets only (Static/Input/TextArea/Markdown/Collapsible/Button/VerticalScroll); NO `Select` (cycling Button instead); never a `Horizontal` mixing a `1fr` sibling with fixed-width children; a canvas child that fills the canvas uses `width: 1fr` never its own `Nfr`; long text Statics `width: 100%`; container `overflow-x: hidden`.
- Service seam (verified): `getattr(self.app_instance, "notes_scope_service", None)`; user id `getattr(self.app_instance, "notes_user_id", None) or "default_user"`; scope string `"local_note"`. Methods: `list_notes(scope=, limit=, offset=0, user_id=)`, `search_notes(scope=, query=, limit=, user_id=)`, `get_note_detail(scope=, note_id=, user_id=)`, `save_note(scope=, title=, content=, note_id=None, version=None, user_id=, keywords=None)` (create when `note_id=None`; update returns truthy/False; with `keywords` returns a dict incl. bumped `version`), `delete_note(scope=, note_id=, version=, user_id=)`. All async → always call through `self._run_library_service_call(fn, ..., isolate_in_worker=True)`.
- EDITOR AMENDMENT (binding): autosave persists silently and NEVER recomposes — status via targeted `query_one(...).update(...)`; version bumped in memory from the save response; refetch+recompose only on transitions (Back, row switch, rail re-entry). Flush pending changes on EVERY exit path. Version conflict during autosave → pause autosave, keep editor content, show `changed elsewhere · [Overwrite] [Reload]`; never silently reload an editor.
- Every persisted field is sanitized with the existing `self._sanitize_media_field(...)` helper pattern in `library_screen.py` (reuse it; it wraps `input_validation`).
- Test fakes must mirror the real seam (method names + argument shapes above); every mutation also gets a REAL-ChaChaNotes-DB regression test.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`. Never touch `.claude/settings.local.json`.**
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Library screen changes require live screenshot QA at 2050×1240 + explicit user approval before merge (Task 8).

## File Structure

- Create: `tldw_chatbook/Library/library_notes_state.py` — pure list/editor display state
- Create: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` — the canvas widget (list + editor modes)
- Modify: `tldw_chatbook/Library/library_shell_state.py` — `browse-notes`/`create-note` rows flip to canvas
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — per-source page sizes; notes view state + handlers
- Create: `Tests/Library/test_library_notes_state.py`; Modify: `Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py` (harness fake)
- Create: `Tests/Notes/test_notes_scope_service_library_canvas.py` — real-DB mutation tests

---

### Task 1: Pure notes display-state module

**Files:**
- Create: `tldw_chatbook/Library/library_notes_state.py`
- Test: `Tests/Library/test_library_notes_state.py`

**Interfaces:**
- Consumes: `format_console_relative_age` from `tldw_chatbook.Workspaces.conversation_browser_state` (same as media state).
- Produces (later tasks rely on these exact names):
  - `LibraryNotesListRow(note_id: str, title: str, age_label: str)`
  - `build_library_notes_list_state(records, *, filter_note: str = "", now=None) -> LibraryNotesListState` where `LibraryNotesListState(rows: tuple[LibraryNotesListRow, ...], header_copy: str, status_copy: str, empty_copy: str)`
  - `NOTES_SORT_MODES = ("newest", "oldest", "title")`, `next_notes_sort_mode(mode: str) -> str`, `sort_notes_records(records, mode) -> list`
  - `build_library_note_editor_state(detail, *, now=None) -> LibraryNoteEditorState` with fields `note_id, title, content, keywords_text, version, meta_line, has_note`
  - `notes_autosave_status_text(state: str, *, word_count: int) -> str` where state ∈ `{"idle","saving","saved","conflict","error"}`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Library/test_library_notes_state.py
"""Pure display-state contracts for the Library notes canvas."""
from datetime import datetime, timezone

from tldw_chatbook.Library.library_notes_state import (
    NOTES_SORT_MODES,
    LibraryNoteEditorState,
    LibraryNotesListRow,
    build_library_note_editor_state,
    build_library_notes_list_state,
    next_notes_sort_mode,
    notes_autosave_status_text,
    sort_notes_records,
)

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

NOTE_A = {"id": "n-1", "title": "Q3 retro", "content": "alpha body",
          "last_modified": "2026-07-07T11:57:00+00:00", "version": 2}
NOTE_B = {"id": "n-2", "title": "Reading list", "content": "bravo body",
          "last_modified": "2026-07-06T12:00:00+00:00", "version": 1}


def test_list_state_builds_rows_with_age_and_header():
    state = build_library_notes_list_state([NOTE_A, NOTE_B], now=NOW)
    assert state.header_copy == "Notes (2)"
    assert state.rows[0] == LibraryNotesListRow(note_id="n-1", title="Q3 retro", age_label="3m")
    assert state.rows[1].age_label == "1d"
    assert state.empty_copy == ""


def test_list_state_empty_uses_quiet_copy():
    state = build_library_notes_list_state([], now=NOW)
    assert state.rows == ()
    assert state.empty_copy == "No notes yet. Create one to see it here."


def test_list_state_filter_note_reflects_active_filter():
    state = build_library_notes_list_state([NOTE_A], filter_note="retro", now=NOW)
    assert state.status_copy == "filter: retro · 1 result"


def test_list_state_tolerates_missing_fields():
    state = build_library_notes_list_state([{"id": "x"}], now=NOW)
    assert state.rows[0].title == "Untitled"
    assert state.rows[0].age_label == ""


def test_sort_mode_cycles_and_wraps():
    assert next_notes_sort_mode("newest") == "oldest"
    assert next_notes_sort_mode("oldest") == "title"
    assert next_notes_sort_mode("title") == "newest"
    assert next_notes_sort_mode("bogus") == "newest"


def test_sort_records_newest_oldest_title():
    newest = sort_notes_records([NOTE_B, NOTE_A], "newest")
    assert [n["id"] for n in newest] == ["n-1", "n-2"]
    oldest = sort_notes_records([NOTE_A, NOTE_B], "oldest")
    assert [n["id"] for n in oldest] == ["n-2", "n-1"]
    by_title = sort_notes_records([NOTE_A, NOTE_B], "title")
    assert [n["id"] for n in by_title] == ["n-1", "n-2"]  # "Q3..." < "Reading..."


def test_editor_state_builds_fields_and_meta_line():
    detail = {"id": "n-1", "title": "Q3 retro", "content": "alpha body",
              "version": 2, "last_modified": "2026-07-07T11:57:00+00:00",
              "created_at": "2026-07-01T10:00:00+00:00",
              "keywords": ["retro", "q3"]}
    state = build_library_note_editor_state(detail, now=NOW)
    assert state.note_id == "n-1"
    assert state.title == "Q3 retro"
    assert state.content == "alpha body"
    assert state.keywords_text == "retro, q3"
    assert state.version == 2
    assert state.has_note is True
    assert "Created 6d" in state.meta_line and "Modified 3m" in state.meta_line
    assert "v2" in state.meta_line


def test_editor_state_none_detail_yields_empty():
    state = build_library_note_editor_state(None, now=NOW)
    assert state.has_note is False
    assert state.note_id == ""


def test_autosave_status_text_variants():
    assert notes_autosave_status_text("idle", word_count=2) == "2 words"
    assert notes_autosave_status_text("saving", word_count=2) == "2 words · saving…"
    assert notes_autosave_status_text("saved", word_count=2) == "2 words · saved"
    assert notes_autosave_status_text("conflict", word_count=2) == "2 words · changed elsewhere"
    assert notes_autosave_status_text("error", word_count=2) == "2 words · save failed"
```

- [ ] **Step 2: Run to verify failure**

Run: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_notes_state.py --tb=short`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Library.library_notes_state`

- [ ] **Step 3: Implement the module**

```python
# tldw_chatbook/Library/library_notes_state.py
"""Pure display-state contract for the Library notes canvas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tldw_chatbook.Workspaces.conversation_browser_state import format_console_relative_age

NOTES_SORT_MODES = ("newest", "oldest", "title")
_UPDATED_KEYS = ("last_modified", "updated_at", "created_at")
_EMPTY_NOTES_COPY = "No notes yet. Create one to see it here."


@dataclass(frozen=True)
class LibraryNotesListRow:
    note_id: str
    title: str
    age_label: str


@dataclass(frozen=True)
class LibraryNotesListState:
    rows: tuple[LibraryNotesListRow, ...]
    header_copy: str
    status_copy: str
    empty_copy: str


@dataclass(frozen=True)
class LibraryNoteEditorState:
    note_id: str
    title: str
    content: str
    keywords_text: str
    version: int | None
    meta_line: str
    has_note: bool


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _updated_raw(record: Mapping[str, Any]) -> str:
    for key in _UPDATED_KEYS:
        value = _text(record.get(key))
        if value:
            return value
    return ""


def _row(record: Mapping[str, Any], *, now: datetime) -> LibraryNotesListRow:
    raw = _updated_raw(record)
    return LibraryNotesListRow(
        note_id=_text(record.get("id")),
        title=_text(record.get("title")) or "Untitled",
        age_label=format_console_relative_age(raw, now=now) if raw else "",
    )


def build_library_notes_list_state(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    filter_note: str = "",
    now: datetime | None = None,
) -> LibraryNotesListState:
    reference_now = now if now is not None else datetime.now(timezone.utc)
    rows = tuple(
        _row(record, now=reference_now)
        for record in (records or ())
        if isinstance(record, Mapping) and _text(record.get("id"))
    )
    status_copy = ""
    if filter_note:
        noun = "result" if len(rows) == 1 else "results"
        status_copy = f"filter: {filter_note} · {len(rows)} {noun}"
    return LibraryNotesListState(
        rows=rows,
        header_copy=f"Notes ({len(rows)})",
        status_copy=status_copy,
        empty_copy="" if rows else _EMPTY_NOTES_COPY,
    )


def next_notes_sort_mode(mode: str) -> str:
    try:
        index = NOTES_SORT_MODES.index(mode)
    except ValueError:
        return NOTES_SORT_MODES[0]
    return NOTES_SORT_MODES[(index + 1) % len(NOTES_SORT_MODES)]


def sort_notes_records(
    records: Sequence[Mapping[str, Any]], mode: str
) -> list[Mapping[str, Any]]:
    items = [r for r in records if isinstance(r, Mapping)]
    if mode == "title":
        return sorted(items, key=lambda r: _text(r.get("title")).lower())
    reverse = mode != "oldest"
    return sorted(items, key=_updated_raw, reverse=reverse)


def _keywords_text(detail: Mapping[str, Any]) -> str:
    keywords = detail.get("keywords")
    if isinstance(keywords, str):
        return keywords.strip()
    if isinstance(keywords, Sequence):
        items = []
        for item in keywords:
            if isinstance(item, Mapping):
                item = item.get("keyword") or item.get("text") or item.get("label")
            text = _text(item)
            if text:
                items.append(text)
        return ", ".join(items)
    return ""


def build_library_note_editor_state(
    detail: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> LibraryNoteEditorState:
    if not isinstance(detail, Mapping) or not _text(detail.get("id")):
        return LibraryNoteEditorState(
            note_id="", title="", content="", keywords_text="",
            version=None, meta_line="", has_note=False,
        )
    reference_now = now if now is not None else datetime.now(timezone.utc)
    version_raw = detail.get("version")
    try:
        version: int | None = int(version_raw) if version_raw is not None else None
    except (TypeError, ValueError):
        version = None
    parts: list[str] = []
    created = _text(detail.get("created_at"))
    if created:
        parts.append(f"Created {format_console_relative_age(created, now=reference_now)}")
    modified = _updated_raw(detail)
    if modified:
        parts.append(f"Modified {format_console_relative_age(modified, now=reference_now)}")
    if version is not None:
        parts.append(f"v{version}")
    return LibraryNoteEditorState(
        note_id=_text(detail.get("id")),
        title=_text(detail.get("title")),
        content=str(detail.get("content") or ""),
        keywords_text=_keywords_text(detail),
        version=version,
        meta_line=" · ".join(parts),
        has_note=True,
    )


def notes_autosave_status_text(state: str, *, word_count: int) -> str:
    base = f"{word_count} words" if word_count != 1 else "1 word"
    suffix = {
        "saving": " · saving…",
        "saved": " · saved",
        "conflict": " · changed elsewhere",
        "error": " · save failed",
    }.get(state, "")
    return f"{base}{suffix}"
```

- [ ] **Step 4: Run to verify pass** — same command. Expected: all PASS.
- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/library_notes_state.py Tests/Library/test_library_notes_state.py
git commit -m "feat(library): pure notes canvas display state

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Per-source page sizes + rail row flips

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (line ~89 `LIBRARY_SOURCE_PAGE_SIZE = 5` and the snapshot fetch ~755–780)
- Modify: `tldw_chatbook/Library/library_shell_state.py` (rows `browse-notes` ~101–109, `create-note` ~130–139)
- Test: `Tests/UI/test_library_shell.py`, `Tests/Library/test_library_shell_state.py` (existing row tests re-anchor)

**Interfaces:**
- Produces: `LIBRARY_SOURCE_PAGE_SIZES = {"notes": 100, "media": 50, "conversations": 50}` (module constant in `library_screen.py`, replacing every `LIBRARY_SOURCE_PAGE_SIZE` use); `browse-notes` row with `target_kind="canvas", target_id="notes"`; `create-note` row with `target_kind="canvas", target_id="notes-create"`.

- [ ] **Step 1: Re-anchor failing tests.** In `Tests/Library/test_library_shell_state.py` find the assertions for `browse-notes` (`target_kind == "screen"`) and `create-note`, and change them to:

```python
def test_browse_notes_row_targets_notes_canvas():
    state = build_library_shell_state(_shell_input())
    notes_row = _row_by_id(state, "browse-notes")
    assert notes_row.target_kind == "canvas"
    assert notes_row.target_id == "notes"


def test_create_note_row_targets_notes_create_canvas():
    state = build_library_shell_state(_shell_input())
    row = _row_by_id(state, "create-note")
    assert row.target_kind == "canvas"
    assert row.target_id == "notes-create"
```

(Adapt helper names to the file's existing helpers — read the file first; the existing tests for these two rows must be UPDATED, not duplicated.)

- [ ] **Step 2: Run to verify the two tests fail** (`--tb=short`, target the two test names with `-k "browse_notes_row or create_note_row"`). Expected: FAIL on target_kind.
- [ ] **Step 3: Implement.** In `library_shell_state.py`: `browse-notes` row → `target_kind="canvas", target_id="notes"` (drop the `TAB_NOTES` use); `create-note` row → `target_kind="canvas", target_id="notes-create"`. Remove the now-unused `TAB_NOTES` import if nothing else uses it. In `library_screen.py`: replace `LIBRARY_SOURCE_PAGE_SIZE = 5` with

```python
LIBRARY_SOURCE_PAGE_SIZES = {"notes": 100, "media": 50, "conversations": 50}
```

and update the three snapshot-fetch call sites (`limit=LIBRARY_SOURCE_PAGE_SIZES["notes"]`, `results_per_page=LIBRARY_SOURCE_PAGE_SIZES["media"]`, `limit=LIBRARY_SOURCE_PAGE_SIZES["conversations"]`) plus any other `LIBRARY_SOURCE_PAGE_SIZE` references (grep; media/conversations state builders may import it — keep their behavior by passing the per-source value).
- [ ] **Step 4: Run the full affected suites** (`Tests/Library/ Tests/UI/test_library_shell.py Tests/UI/test_destination_shells.py`). Fix any test that asserted the old cap (e.g. "5 of N" status copy) by re-anchoring to the new per-source sizes. Expected: all PASS.
- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Library/library_shell_state.py Tests/Library/ Tests/UI/test_library_shell.py
git commit -m "feat(library): notes rows target in-Library canvas; per-source page sizes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Note:** after this task the `browse-notes` row selects an empty canvas kind (`notes` renders nothing yet) — Task 3 fills it. This is the same intentionally-inert intermediate state L2a used.

---

### Task 3: Notes list canvas (widget + screen wiring + filter + sort)

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (canvas compose branch; handlers)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: Task 1 builders; snapshot records at `self._local_source_records["notes"]` (already fetched); `search_notes` seam for the filter.
- Produces: widget `LibraryNotesCanvas(list_state, sort_mode)` with ids used by later tasks: `#library-notes-list`, `#library-notes-filter` (Input), `#library-notes-sort` (Button `sort: Newest ▸`), `#library-notes-row-{i}` (Button, carries `.note_id` attribute), `#library-notes-empty`, `#library-notes-status`. Screen state: `self._library_notes_view: str` (`"list"`/`"editor"`), `self._library_notes_sort: str`, `self._library_notes_filter: str`, `self._library_notes_filter_records: list | None`.

- [ ] **Step 1: Write failing pilot tests** (append to `Tests/UI/test_library_shell.py`; reuse `_build_test_app`/`_seed_conversations` — extend the seed helper to accept `notes=[...]` dicts routed into the harness fake's `list_notes` response, mirroring how media seeding works; check `Tests/UI/test_destination_shells.py`'s fake service and add `notes` records support there):

```python
@pytest.mark.asyncio
async def test_library_shell_notes_row_opens_notes_list_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        header = str(screen.query_one("#library-notes-header").renderable)
        assert header == "Notes (2)"
        assert screen.query_one("#library-notes-filter")
        assert screen.query_one("#library-notes-sort")


@pytest.mark.asyncio
async def test_library_shell_notes_sort_button_cycles():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sort")
        assert screen._library_notes_sort == "newest"
        screen.query_one("#library-notes-sort").press()
        await pilot.pause()
        assert screen._library_notes_sort == "oldest"


@pytest.mark.asyncio
async def test_library_shell_notes_filter_queries_search_seam():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(); await pilot.pause()
        service = app.notes_scope_service
        assert service.search_calls[-1]["query"] == "retro"
        assert screen._library_notes_filter == "retro"
```

with a module-level fixture helper:

```python
def _two_notes():
    return [
        {"id": "n-1", "title": "Q3 retro", "content": "alpha budget line",
         "last_modified": "2026-07-07T11:57:00+00:00", "version": 2},
        {"id": "n-2", "title": "Reading list", "content": "bravo",
         "last_modified": "2026-07-06T12:00:00+00:00", "version": 1},
    ]
```

The harness fake (`Tests/UI/test_destination_shells.py`'s notes service stand-in — if none exists, add `StaticNotesScopeService` beside the media one) must implement, with the REAL signatures: `async list_notes(*, scope, limit, offset=0, user_id)`, `async search_notes(*, scope, query, limit, user_id)` (returns records whose title/content contains query, case-insensitive, and records the call in `self.search_calls`), `async get_note_detail(*, scope, note_id, user_id)`, `async save_note(*, scope, title, content, note_id=None, version=None, user_id=None, keywords=None)` (create appends with version=1; update requires `version` == stored version else return `False`; bumps version; records in `self.save_calls`), `async delete_note(*, scope, note_id, version, user_id)` (version-checked; records in `self.delete_calls`).

- [ ] **Step 2: Run to verify failure.** Expected: FAIL (no `#library-notes-row-0` / missing fake methods).
- [ ] **Step 3: Implement the widget:**

```python
# tldw_chatbook/Widgets/Library/library_notes_canvas.py
"""Library notes canvas: list mode (rows + filter + sort)."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_notes_state import LibraryNotesListState

_SORT_LABELS = {"newest": "Newest", "oldest": "Oldest", "title": "Title"}


class LibraryNotesCanvas(Vertical):
    """Render the notes list: header, filter, sort control, rows."""

    def __init__(self, list_state: LibraryNotesListState, *, sort_mode: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.list_state = list_state
        self.sort_mode = sort_mode
        self.styles.width = "1fr"
        self.styles.min_width = 40

    def compose(self) -> ComposeResult:
        yield Static(self.list_state.header_copy, id="library-notes-header",
                     classes="destination-section", markup=False)
        yield Input(placeholder="Filter notes… (Enter)", id="library-notes-filter")
        yield Button(
            f"sort: {_SORT_LABELS.get(self.sort_mode, 'Newest')} ▸",
            id="library-notes-sort", classes="library-canvas-action", compact=True,
        )
        if self.list_state.status_copy:
            yield Static(self.list_state.status_copy, id="library-notes-status", markup=False)
        if not self.list_state.rows:
            yield Static(self.list_state.empty_copy, id="library-notes-empty", markup=False)
            return
        with Vertical(id="library-notes-list"):
            for index, row in enumerate(self.list_state.rows):
                button = Button(
                    f"{row.title}\n{row.age_label}" if row.age_label else row.title,
                    id=f"library-notes-row-{index}",
                    classes="library-notes-row", compact=True,
                )
                button.note_id = row.note_id
                yield button
```

**Screen wiring** (`library_screen.py`, mirroring the media canvas branch exactly): in `compose_content()`'s canvas host, add a branch for `canvas_kind == "notes"` that renders `LibraryNotesCanvas(list_state, sort_mode=self._library_notes_sort, id="library-notes-canvas")` where `list_state = build_library_notes_list_state(sort_notes_records(self._library_notes_filter_records if self._library_notes_filter_records is not None else self._local_source_records.get("notes", []), self._library_notes_sort), filter_note=self._library_notes_filter)`. Initialize in `__init__`: `self._library_notes_view = "list"`, `self._library_notes_sort = "newest"`, `self._library_notes_filter = ""`, `self._library_notes_filter_records = None`. Handlers:

```python
@on(Button.Pressed, "#library-notes-sort")
def handle_library_notes_sort(self, event: Button.Pressed) -> None:
    event.stop()
    self._library_notes_sort = next_notes_sort_mode(self._library_notes_sort)
    self.refresh(recompose=True)

@on(Input.Submitted, "#library-notes-filter")
def handle_library_notes_filter(self, event: Input.Submitted) -> None:
    event.stop()
    submitted = self._safe_text(event.value, max_length=200).strip()
    if submitted == self._library_notes_filter:
        return
    self._library_notes_filter = submitted
    if not submitted:
        self._library_notes_filter_records = None
        self.refresh(recompose=True)
        return
    self.run_worker(self._run_library_notes_filter(submitted), exclusive=True,
                    group="library_notes_filter")

async def _run_library_notes_filter(self, query: str) -> None:
    service = getattr(self.app_instance, "notes_scope_service", None)
    if service is None:
        return
    try:
        records = await self._run_library_service_call(
            service.search_notes, scope="local_note", query=query,
            limit=LIBRARY_SOURCE_PAGE_SIZES["notes"],
            user_id=getattr(self.app_instance, "notes_user_id", None) or "default_user",
            isolate_in_worker=True,
        )
    except Exception:
        logger.warning("Library notes filter failed.", exc_info=True)
        return
    self._library_notes_filter_records = list(records or [])
    self.refresh(recompose=True)
    self.call_after_refresh(self._focus_library_notes_filter_input)

def _focus_library_notes_filter_input(self) -> None:
    try:
        self.query_one("#library-notes-filter", Input).focus()
    except (NoMatches, QueryError):
        pass
```

State reset: in `_select_library_rail_row` (where media resets its view state), also reset `self._library_notes_view = "list"`, `self._library_notes_filter = ""`, `self._library_notes_filter_records = None`. Re-seed the filter Input's value on recompose by passing `value=self._library_notes_filter` in the widget (add `filter_value` param to the widget constructor and `Input(value=self.filter_value, ...)`).
- [ ] **Step 4: Run the pilots + full affected suites.** Expected: PASS.
- [ ] **Step 5: Commit** (`feat(library): notes list canvas with filter and sort`).

---

### Task 4: Editor mode (read-only render + Back + reset discipline)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (editor compose)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: `build_library_note_editor_state`, `get_note_detail` seam.
- Produces ids later tasks rely on: `#library-note-back`, `#library-note-title` (Input), `#library-note-body` (TextArea), `#library-note-keywords` (Input), `#library-note-meta` (Static), `#library-note-save`, `#library-note-preview`, `#library-note-use-in-console`, `#library-note-export-md`, `#library-note-export-txt`, `#library-note-copy`, `#library-note-delete` (danger class, far end). Screen state: `self._library_note_detail`, `self._selected_note_id`, `self._library_note_version` (int | None, bumped in memory by saves).

- [ ] **Step 1: Failing pilots:** clicking `#library-notes-row-0` fetches detail and renders the editor (`#library-note-title` value == "Q3 retro", `#library-note-body` text == content, meta Static contains "v2"); `#library-note-back` returns to the list; re-entering Browse ▸ Notes from the rail lands on the LIST view (reset discipline); switching to another rail row and back does not leak editor state. Write all four tests concretely following Task 3's pilot shape (row press → `_wait_for_selector` → assertions on widget values and `screen._library_notes_view`).
- [ ] **Step 2: Run: FAIL.**
- [ ] **Step 3: Implement.** Widget gains editor fields (`editor_state`, `preview: bool = False` for Task 8) and an editor compose branch (stacked, in order): Back button → title Input (`value=editor_state.title`) → body `TextArea(editor_state.content, id="library-note-body")` → keywords Input (`value=editor_state.keywords_text`, placeholder `Keywords (comma-separated)`) → meta Static (meta_line + autosave status, `id="library-note-meta"`) → actions `Horizontal(classes="ds-toolbar")` with the buttons listed above (`Delete` last with class `library-media-action-danger` — reuse the existing danger CSS). Screen: row press handler (class `library-notes-row`) reads `button.note_id`, sets `self._selected_note_id`, `self._library_notes_view = "editor"`, kicks `self.run_worker(self._refresh_library_note_detail(note_id), exclusive=True, group="library_note_detail")` which calls `get_note_detail` (offloaded, re-checks `note_id == self._selected_note_id` after await — the L2a race guard), stores `self._library_note_detail` + `self._library_note_version`, recomposes. Back handler resets to list + clears detail/selected id. Extend `_select_library_rail_row` reset to also clear `_library_note_detail`/`_selected_note_id`. TextArea CSS (Task 8 adds file): min-height 12, max-height 20.
- [ ] **Step 4: Run pilots + suites: PASS.**
- [ ] **Step 5: Commit** (`feat(library): in-canvas note editor renders detail with back navigation`).

---

### Task 5: Save, autosave, conflict policy, flush-on-exit

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_shell.py`, Create: `Tests/Notes/test_notes_scope_service_library_canvas.py`

**Interfaces:**
- Consumes: `save_note` seam (update path: `note_id=`, `version=`, optional `keywords=`).
- Produces: `self._library_note_dirty: bool`, `self._library_note_autosave_state: str`, methods `_save_library_note(*, explicit: bool)`, `_flush_pending_note_save()`, autosave timer via `self.set_timer(2.0, ...)` re-armed on change events.

- [ ] **Step 1: Failing tests.**
  - Pilot: editing `#library-note-body` then pressing `#library-note-save` calls the fake's `save_note` with `note_id`, current `version`, sanitized title/content, keywords parsed from the keywords Input (`[k.strip() for k in value.split(",") if k.strip()] or None`); in-memory `screen._library_note_version` bumps (fake returns dict with `version`); NO recompose of the editor (assert the TextArea widget instance is the same object before/after save).
  - Pilot: changing the body arms autosave; after the debounce fires (use `set_timer` with a short interval in tests via monkeypatching `LIBRARY_NOTES_AUTOSAVE_SECONDS = 0.05`), `save_note` was called without pressing Save; `#library-note-meta` text contains "saved".
  - Pilot (flush-on-exit): edit body → immediately press `#library-note-back` → fake received the save BEFORE view switched; same for switching rail rows.
  - Pilot (conflict): fake configured to return `False` (version mismatch) → meta shows "changed elsewhere"; `#library-note-conflict-overwrite` and `#library-note-conflict-reload` buttons appear (recompose IS allowed for the conflict transition); Overwrite re-saves with `version=None`… **No** — `version=None` on update is not supported by the seam contract (update passes `version` through to the versioned update). Overwrite instead: re-fetch detail silently, take the NEW version number, re-save the user's editor content with that version. Reload: re-fetch and recompose the editor from the fetched detail (discarding local edits). Assert both behaviors against the fake.
  - Real-DB (`Tests/Notes/test_notes_scope_service_library_canvas.py`): construct the real `NotesScopeService` over a real `NotesInteropService` with an in-memory ChaChaNotes DB (mirror the construction in `app.py:1579` and existing `Tests/Notes/` fixtures — read one for the established fixture pattern); test create→update round-trip with correct version, update with stale version returns falsy (conflict path is real), keywords round-trip via `save_note(..., keywords=["a","b"])` returns dict with bumped version and the keywords persisted.
- [ ] **Step 2: Run: FAIL.**
- [ ] **Step 3: Implement.** `TextArea.Changed`/`Input.Changed` handlers (scoped to the three editor inputs) set `_library_note_dirty = True`, set autosave state `"idle"→"saving"` label lazily, and re-arm a single timer (`self._library_notes_autosave_timer`; cancel previous; `self.set_timer(LIBRARY_NOTES_AUTOSAVE_SECONDS, self._autosave_library_note)`). `_save_library_note(explicit)`: reads the three widgets synchronously, sanitizes (`_sanitize_media_field` pattern: title ≤300, content ≤2_000_000, keywords each ≤100), calls `save_note` offloaded exclusive (`group="library_note_save"`); on truthy result bump `_library_note_version` (+1 or from returned dict), set state `"saved"`, `_library_note_dirty = False`, update `#library-note-meta` via targeted `update()` (NEVER recompose); on `False` set state `"conflict"` and recompose (conflict UI is a transition). `_flush_pending_note_save()`: if dirty, cancel timer and `await _save_library_note(explicit=False)`; called at the TOP of: Back handler, notes row press, `_select_library_rail_row`. Conflict buttons per Step 1 semantics.
- [ ] **Step 4: Run pilots + real-DB tests + full suites: PASS.**
- [ ] **Step 5: Commit** (`feat(library): note save + autosave with conflict policy and exit flush`).

---

### Task 6: Create flows (blank via rail; from template)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Test: `Tests/UI/test_library_shell.py` + real-DB create test in `Tests/Notes/test_notes_scope_service_library_canvas.py`

**Interfaces:**
- Consumes: `save_note(note_id=None)` create path (returns created id); `NOTE_TEMPLATES` dict from `tldw_chatbook.Event_Handlers.notes_events` (key → mapping with `title`/`content`); `template_display_label(key, template)` from `tldw_chatbook.Widgets.Note_Widgets.notes_workbench_panes`.
- Produces: canvas kind `"notes-create"` renders the notes canvas in create mode: a `New note` header, a `[Blank note]` button (`#library-notes-create-blank`), then one row-button per template (`#library-notes-template-{i}`, carries `.template_key`).

- [ ] **Step 1: Failing pilots:** selecting the rail `create-note` row renders `#library-notes-create-blank` + at least one template row; pressing blank calls fake `save_note` with `note_id=None`, `title="Untitled"`, empty content, then lands in editor mode with the created id selected; pressing a template row creates with that template's title/content. Real-DB test: create-from-template persists title+content and returns an id that `get_note_detail` round-trips.
- [ ] **Step 2: Run: FAIL.**
- [ ] **Step 3: Implement.** Screen: `canvas_kind == "notes-create"` branch renders the create view (widget gains `mode: str` — `"list"`/`"editor"`/`"create"`). Create handlers call `save_note(scope="local_note", title=..., content=..., note_id=None, user_id=...)` offloaded exclusive (`group="library_note_create"`), then set `_selected_note_id` to the created id (service returns the id for local create; the fake mirrors this), switch `_library_notes_view = "editor"`, select the `browse-notes` rail row (`self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES` — add that constant beside the existing two), refresh the snapshot (so the list/count includes the new note), and fetch detail. Template rows come from `NOTE_TEMPLATES.items()` sorted by key, labels via `template_display_label`; unknown/malformed template values degrade to blank-note behavior.
- [ ] **Step 4: Run: PASS.**
- [ ] **Step 5: Commit** (`feat(library): create note (blank/template) lands in the in-canvas editor`).

---

### Task 7: Delete (inline confirm) + keywords persistence check

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Test: `Tests/UI/test_library_shell.py` + real-DB delete test

**Interfaces:**
- Consumes: `delete_note(scope=, note_id=, version=, user_id=)`.
- Produces: `self._library_note_confirming_delete: bool`; ids `#library-note-delete-confirm`, `#library-note-delete-cancel`, confirm copy Static `#library-note-delete-confirm-copy` ("Delete this note? This cannot be undone from Library.").

- [ ] **Step 1: Failing pilots:** Delete press shows confirm affordance without calling the service; Confirm calls `delete_note` with current version, returns to list mode, note gone from the list (snapshot refreshed), state reset; Cancel restores the normal toolbar. Real-DB: delete round-trip (created → deleted → `get_note_detail` returns None/missing) and stale-version delete returns falsy without removing.
- [ ] **Step 2: Run: FAIL.**
- [ ] **Step 3: Implement** exactly on the media-viewer delete pattern (confirm state + swapped toolbar + exclusive worker `group="library_note_delete"`; flush pending save BEFORE entering confirm so version is current).
- [ ] **Step 4: Run: PASS.**
- [ ] **Step 5: Commit** (`feat(library): delete note from the canvas with inline confirm`).

---

### Task 8: Preview, Export, Copy, Use in Console + CSS

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate `tldw_cli_modular.tcss`)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: `ChatHandoffPayload.from_source_content` + `self.app_instance.open_chat_with_handoff` gated on `context_handoff_enabled` (copy the media viewer's `_selected_media_handoff_payload` pattern in `library_screen.py`, with `source="notes"`, `item_type="note"`, `discovery_owner="notes"`, `suggested_prompt="Use this note as context and help me work with it."`, metadata `{"note_version": ..., "keywords": [...]}`); export content built by a pure helper `build_note_export_content(title, content, keywords_text, note_id, export_format, *, now)` added to `library_notes_state.py` (mirror the frontmatter/plain formats from `notes_screen._build_export_content` verbatim); file save via the same `FileSave`-dialog flow `notes_screen._export_current_note` uses (read lines 1690–1720 and reuse the dialog + `selected_path.write_text(...)` shape); clipboard via the same mechanism `notes_screen._copy_current_note_to_clipboard` uses (read lines 1673–1688 and call the same app API).
- Produces: `preview` toggle state on the widget — body renders `Markdown(content)` instead of the TextArea when `self._library_note_preview` is True (button label toggles `Preview`/`Edit`).

- [ ] **Step 1: Failing tests:** pure unit for `build_note_export_content` (markdown format contains frontmatter `title:`/`keywords:`/`note_id:` lines; text format contains `Title:` header and the `====` rule); pilot for Preview toggling (Markdown widget present, TextArea absent, then back); pilot for Use in Console building a payload with `source == "notes"` and the current note id (assert via a recorded fake `open_chat_with_handoff` — same hook the media tests use); pilot for Copy calling the clipboard seam (monkeypatch/record).
- [ ] **Step 2: Run: FAIL.**
- [ ] **Step 3: Implement**, including CSS in `_agentic_terminal.tcss`: `#library-note-body { height: auto; min-height: 12; max-height: 20; }`, `.library-notes-row { width: 100%; }`, editor label/meta styling reusing the existing `.library-media-edit-label` / muted-meta conventions, then `./build_css.sh`.
- [ ] **Step 4: Run everything: PASS.** Also add a generated-CSS presence test if the house suite has one for new selectors (grep `Tests/` for `tldw_cli_modular` presence tests and extend).
- [ ] **Step 5: Commit** (`feat(library): note preview, export, copy, use-in-console + canvas styles`).

---

### Task 9: Whole-branch review, live QA, approval gate

**Files:** QA evidence under `Docs/superpowers/qa/library-notes-l2b1-2026-07/`.

- [ ] **Step 1:** Full affected suites green (`Tests/Library/ Tests/Notes/ Tests/UI/test_library_shell.py Tests/UI/test_destination_shells.py`).
- [ ] **Step 2:** Final whole-branch review (most capable model) over `merge-base(origin/dev, HEAD)..HEAD` with the review-package script; fix Critical/Important; re-run suites.
- [ ] **Step 3:** Live QA: seed real notes into the isolated HOME's ChaChaNotes DB (reuse `seed_notes_qa.py` pattern — path `$HOME/.local/share/tldw_cli/default_user/tldw_chatbook_ChaChaNotes.db`, client `tldw_cli_local_instance_v1`), serve via textual-serve with worktree PYTHONPATH, capture at 2050×1240: list populated, filter active, editor open (populated meta + autosave "saved"), preview on, create-from-template view, delete confirm. Commit evidence + README.
- [ ] **Step 4:** Present captures to the user for the explicit approval gate. **Do not open a PR without approval.**
- [ ] **Step 5:** On approval: push branch `claude/library-l2b`, open PR to `dev` titled `feat(library): Browse ▸ Notes in-canvas editor (L2b.1)`.
