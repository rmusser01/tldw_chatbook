# Library multi-select row export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-11-library-multiselect-export-design.md`. Branch `claude/followups-multiselect-export` off dev `b15880fd`. Anchors are exact at branch point; grep symbols, line numbers drift.

**Goal:** Let a user enter a per-source "select mode" in a Library browse canvas (Media / Conversations / Notes), check individual rows, and export exactly those rows as a chatbook.

**Architecture:** Reuse `ExportScope.kind` + a new hashable `ids` override so an `ExportScope(kind="media", ids=(...))` means "export exactly these ids" (empty `ids` = today's whole-source behavior, unchanged). A pure `RowSelection` helper holds the per-source checked-id set. The three canvas state builders gain `checked`/`select_mode`/`selected_count`; the three canvases render a ☑/☐ glyph + a selection action row in select mode; the screen wires a Select toggle + row-press interception, all driven through the same full-screen `self.refresh(recompose=True)` every Library interaction already uses.

**Tech Stack:** Python ≥3.11, Textual, pytest. No new third-party deps.

## Global Constraints

- **No behavior change when `ids` is empty:** every existing export path (`ExportScope(kind=...)` with no ids) behaves byte-for-byte as today.
- **`ids` is a `tuple[str, ...]`** so `ExportScope` stays frozen/hashable/`==`-comparable (the counts worker's `scope != self._library_export_scope` stale-guard requires it).
- **`ids` only for a single-source kind:** `ExportScope(kind="everything", ids=(...))` raises.
- **Reuse `kind`, do NOT add a `"selected"` kind:** `library_export_state.py:167` (`show_media_fields = scope.kind in _MEDIA_BEARING_SCOPE_KINDS`) must keep working for selected-media — reusing `kind="media"` does; a new kind would silently hide the media-quality field.
- **`ids` branch runs BEFORE `_effective_media_type`** in every resolver.
- **Per-source only** (no cross-source tray); **visible/rendered rows only**.
- **Selection lifecycle (plan deviation from spec — see note below):** the set is cleared on the source's filter/sort change and whenever select mode is toggled off. It **persists across canvas switches** (the spec said "clear on leaving the canvas", but the codebase has no single clean canvas-leave hook, and reconcile-on-render already keeps a persisted set honest — a returning user resumes the mode they left, and a stale id is dropped on the next render). This is a deliberate, safe simplification; if the user wants strict clear-on-leave, add a clear call to the rail browse-row switch.
- **Reconcile on render:** in select mode the selection set is intersected with the currently-rendered ids each build (drops vanished ids), so the count and export stay WYSIWYG.
- **Notes select-mode entry** first `await self._flush_library_note_save()`.
- **Recompose model:** row toggle and all select-mode actions recompose via the screen's `self.refresh(recompose=True)` (the only working canvas-update path in this codebase — `sync_state` is dead). No new targeted-update infrastructure.
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: `ExportScope.ids` + resolver/count/label branches (pure)

**Files:**
- Modify: `tldw_chatbook/Library/library_export_scope.py`
- Test: `Tests/Library/test_library_export_scope_ids.py` (create)

**Interfaces:**
- Produces: `ExportScope(kind, media_type=None, ids: tuple[str, ...] = ())`; `resolve_export_selections`/`count_export_scope`/`export_scope_label` honor `ids`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_library_export_scope_ids.py`:
```python
import pytest

from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.Library.library_export_scope import (
    ExportScope, resolve_export_selections, count_export_scope, export_scope_label,
)


class _RecordingMediaDB:
    def __init__(self): self.calls = 0
    def get_all_active_media_ids(self, media_type=None):
        self.calls += 1
        return [1, 2, 3]


class _RecordingChaChaDB:
    def __init__(self): self.calls = 0
    def get_all_conversation_ids(self):
        self.calls += 1
        return ["c1", "c2"]
    def get_all_note_ids(self):
        self.calls += 1
        return ["n1"]


def test_ids_only_valid_for_single_source():
    ExportScope(kind="media", ids=("1", "2"))          # ok
    with pytest.raises(ValueError):
        ExportScope(kind="everything", ids=("1",))


def test_scope_with_ids_is_hashable_and_equal():
    a = ExportScope(kind="media", ids=("1", "2"))
    b = ExportScope(kind="media", ids=("1", "2"))
    assert a == b and hash(a) == hash(b)
    assert a in {b}


def test_resolve_with_ids_returns_them_without_querying():
    media, chacha = _RecordingMediaDB(), _RecordingChaChaDB()
    sel = resolve_export_selections(ExportScope(kind="media", ids=("7", "8")), media, chacha)
    assert sel == {ContentType.MEDIA: ["7", "8"]}
    assert media.calls == 0 and chacha.calls == 0   # no whole-source query


def test_resolve_without_ids_unchanged():
    media, chacha = _RecordingMediaDB(), _RecordingChaChaDB()
    sel = resolve_export_selections(ExportScope(kind="media"), media, chacha)
    assert sel == {ContentType.MEDIA: ["1", "2", "3"]}
    assert media.calls == 1


def test_count_with_ids():
    media, chacha = _RecordingMediaDB(), _RecordingChaChaDB()
    counts = count_export_scope(ExportScope(kind="notes", ids=("a", "b", "c")), media, chacha)
    assert counts == {"media": 0, "conversations": 0, "notes": 3}
    assert chacha.calls == 0


def test_label_with_ids():
    counts = count_export_scope(
        ExportScope(kind="conversations", ids=("x", "y")), _RecordingMediaDB(), _RecordingChaChaDB()
    )
    assert export_scope_label(ExportScope(kind="conversations", ids=("x", "y")), counts) \
        == "Selected conversations · 2 items"
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_export_scope_ids.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `ExportScope.__init__() got an unexpected keyword argument 'ids'`.

- [ ] **Step 3: Add the `ids` field + validation**

In `library_export_scope.py`, add to the `ExportScope` dataclass (after `media_type`):
```python
    ids: tuple[str, ...] = ()
```
Extend `__post_init__`:
```python
    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"Unknown export scope kind: {self.kind!r}. Expected one of {_VALID_KINDS}."
            )
        if self.ids and self.kind == "everything":
            raise ValueError("ExportScope.ids may only scope a single source, not 'everything'.")
```
Add near `_VALID_KINDS`:
```python
_KIND_TO_CONTENT_TYPE = {
    "media": ContentType.MEDIA,
    "conversations": ContentType.CONVERSATION,
    "notes": ContentType.NOTE,
}
```

- [ ] **Step 4: Branch the three functions on `scope.ids` (before any media_type logic)**

`resolve_export_selections` — add at the very top of the body (before the existing `selections = {}`):
```python
    if scope.ids:
        return {_KIND_TO_CONTENT_TYPE[scope.kind]: list(scope.ids)}
```
`count_export_scope` — add right after `counts = {"media": 0, "conversations": 0, "notes": 0}`:
```python
    if scope.ids:
        counts[scope.kind] = len(scope.ids)
        return counts
```
`export_scope_label` — add as the first branch:
```python
    if scope.ids:
        return f"Selected {scope.kind} · {counts.get(scope.kind, len(scope.ids))} items"
```

- [ ] **Step 5: Run to verify it passes + full existing scope suite**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_export_scope_ids.py Tests/Library/ -k "export_scope or export" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (new tests green; pre-existing scope tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Library/library_export_scope.py Tests/Library/test_library_export_scope_ids.py
git commit -m "feat(export): ExportScope.ids override for explicit-subset export (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `RowSelection` pure helper

**Files:**
- Create: `tldw_chatbook/Library/row_selection.py`
- Test: `Tests/Library/test_row_selection.py` (create)

**Interfaces:**
- Consumes: `ExportScope` (Task 1).
- Produces: `RowSelection(kind)` with `.ids -> frozenset[str]`, `.count -> int`, `.is_selected(id)`, `.toggle(id)`, `.select_all(iterable)`, `.clear()`, `.reconcile(iterable)`, `.export_scope() -> ExportScope`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_row_selection.py`:
```python
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def test_toggle_add_remove():
    s = RowSelection("media")
    s.toggle("1"); s.toggle("2")
    assert s.is_selected("1") and s.count == 2
    s.toggle("1")
    assert not s.is_selected("1") and s.count == 1
    s.toggle("")  # empty id ignored
    assert s.count == 1


def test_select_all_and_clear():
    s = RowSelection("notes")
    s.select_all(["a", "b", "", "c"])   # empties skipped
    assert s.ids == frozenset({"a", "b", "c"})
    s.clear()
    assert s.count == 0


def test_reconcile_drops_absent_ids():
    s = RowSelection("conversations")
    s.select_all(["a", "b", "c"])
    s.reconcile(["b", "c", "d"])        # 'a' no longer rendered
    assert s.ids == frozenset({"b", "c"})


def test_export_scope_sorts_ids():
    s = RowSelection("media")
    s.select_all(["10", "2", "1"])
    assert s.export_scope() == ExportScope(kind="media", ids=("1", "10", "2"))
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_row_selection.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Library.row_selection`.

- [ ] **Step 3: Implement the module**

Create `tldw_chatbook/Library/row_selection.py`:
```python
"""Per-source checked-row accumulator for the Library multi-select export.

Pure and Textual-free: the screen owns one instance per browsable source
(media/conversations/notes) and drives it from the row-press handlers, then
turns the checked ids into an ``ExportScope`` for the export canvas.
"""
from __future__ import annotations

from typing import Iterable

from tldw_chatbook.Library.library_export_scope import ExportScope


class RowSelection:
    def __init__(self, kind: str) -> None:
        # kind is one of "media" | "conversations" | "notes" — the ExportScope
        # single-source kind these ids belong to.
        self._kind = kind
        self._ids: set[str] = set()

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._ids)

    @property
    def count(self) -> int:
        return len(self._ids)

    def is_selected(self, row_id: str) -> bool:
        return row_id in self._ids

    def toggle(self, row_id: str) -> None:
        if not row_id:
            return
        self._ids.discard(row_id) if row_id in self._ids else self._ids.add(row_id)

    def select_all(self, rendered_ids: Iterable[str]) -> None:
        self._ids.update(rid for rid in rendered_ids if rid)

    def clear(self) -> None:
        self._ids.clear()

    def reconcile(self, rendered_ids: Iterable[str]) -> None:
        """Drop any selected id no longer present in the rendered rows."""
        self._ids &= set(rendered_ids)

    def export_scope(self) -> ExportScope:
        return ExportScope(kind=self._kind, ids=tuple(sorted(self._ids)))
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-1 tests. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Library/row_selection.py Tests/Library/test_row_selection.py
git commit -m "feat(export): pure RowSelection accumulator for per-source multi-select (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `checked` / `select_mode` / `selected_count` in the three state builders (pure)

**Files:**
- Modify: `tldw_chatbook/Library/library_media_state.py`, `library_conversations_state.py`, `library_notes_state.py`
- Test: `Tests/Library/test_library_multiselect_state.py` (create)

**Interfaces:**
- Produces:
  - `LibraryMediaRow`/`LibraryConversationRow`/`LibraryNotesListRow` gain `checked: bool = False`.
  - `build_library_media_state(..., select_mode: bool = False, selected_ids: frozenset[str] = frozenset())`; same two new kw params on `build_library_conversations_state` and `build_library_notes_list_state`.
  - Each `*CanvasState`/`LibraryNotesListState` gains `select_mode: bool` and `selected_count: int` (= number of rendered rows whose `checked` is True).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Library/test_library_multiselect_state.py`:
```python
from tldw_chatbook.Library.library_media_state import build_library_media_state
from tldw_chatbook.Library.library_conversations_state import build_library_conversations_state
from tldw_chatbook.Library.library_notes_state import build_library_notes_list_state


def test_media_checked_and_count():
    records = [
        {"id": "1", "title": "A", "type": "video", "updated_at": None},
        {"id": "2", "title": "B", "type": "audio", "updated_at": None},
    ]
    state = build_library_media_state(records, select_mode=True, selected_ids=frozenset({"1"}))
    by_id = {r.media_id: r for r in state.rows}
    assert by_id["1"].checked is True and by_id["2"].checked is False
    assert state.select_mode is True and state.selected_count == 1


def test_media_default_no_select_mode():
    records = [{"id": "1", "title": "A", "type": "video", "updated_at": None}]
    state = build_library_media_state(records)
    assert state.select_mode is False and state.selected_count == 0
    assert state.rows[0].checked is False


def test_conversations_checked_and_count():
    records = [
        {"id": "c1", "title": "One", "message_count": 1, "updated_at": None},
        {"id": "c2", "title": "Two", "message_count": 2, "updated_at": None},
    ]
    state = build_library_conversations_state(records, select_mode=True, selected_ids=frozenset({"c2"}))
    by_id = {r.conversation_id: r for r in state.rows}
    assert by_id["c2"].checked is True and by_id["c1"].checked is False
    assert state.selected_count == 1


def test_notes_checked_and_count():
    records = [{"id": "n1", "title": "N1"}, {"id": "n2", "title": "N2"}]
    state = build_library_notes_list_state(records, select_mode=True, selected_ids=frozenset({"n1"}))
    by_id = {r.note_id: r for r in state.rows}
    assert by_id["n1"].checked is True and by_id["n2"].checked is False
    assert state.select_mode is True and state.selected_count == 1
```
(If a builder's minimal record shape differs — e.g. required keys — copy the shape an existing `Tests/Library/test_library_*_state.py` uses; the `checked`/`select_mode`/`selected_count` assertions are the point.)

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_multiselect_state.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `build_library_media_state() got an unexpected keyword argument 'select_mode'`.

- [ ] **Step 3: Media state**

In `library_media_state.py`: add `checked: bool = False` to `LibraryMediaRow`; add `select_mode: bool = False` and `selected_count: int = 0` to the `LibraryMediaCanvasState` dataclass. Add params to `build_library_media_state`:
```python
def build_library_media_state(
    records: Sequence[Mapping[str, Any]],
    *,
    active_type: str = "All",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
) -> LibraryMediaCanvasState:
```
In the row comprehension add `checked=entry.media_id in selected_ids,` to each `LibraryMediaRow(...)`. After building `rows`, compute `selected_count = sum(1 for r in rows if r.checked)` and pass `select_mode=select_mode, selected_count=selected_count` into the returned `LibraryMediaCanvasState(...)`.

- [ ] **Step 4: Conversations state**

Same shape in `library_conversations_state.py`: `checked: bool = False` on `LibraryConversationRow`; `select_mode`/`selected_count` on `LibraryConversationsCanvasState`; the two new kw params on `build_library_conversations_state`; `checked=entry.conversation_id in selected_ids,` in the row comprehension; `selected_count = sum(1 for r in rows if r.checked)`.

- [ ] **Step 5: Notes state**

In `library_notes_state.py`: add `checked: bool = False` to `LibraryNotesListRow`; add `select_mode: bool` and `selected_count: int` to `LibraryNotesListState`. Change `_row` and its caller to thread `selected_ids`:
```python
def _row(record: Mapping[str, Any], *, now: datetime, selected_ids: frozenset[str]) -> LibraryNotesListRow:
    raw = _updated_raw(record)
    note_id = _text(record.get("id"))
    return LibraryNotesListRow(
        note_id=note_id,
        title=_text(record.get("title")) or "Untitled",
        age_label=format_console_relative_age(raw, now=now) if raw else "",
        checked=note_id in selected_ids,
    )
```
Add `select_mode: bool = False, selected_ids: frozenset[str] = frozenset()` to `build_library_notes_list_state`; pass `selected_ids=selected_ids` into each `_row(...)`; `selected_count = sum(1 for r in rows if r.checked)`; set both new fields on the returned `LibraryNotesListState`.

- [ ] **Step 6: Run to verify it passes + existing state suites**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/test_library_multiselect_state.py Tests/Library/ -k "media_state or conversations_state or notes_state or notes_list" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (defaults keep every existing state test green).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Library/library_media_state.py tldw_chatbook/Library/library_conversations_state.py tldw_chatbook/Library/library_notes_state.py Tests/Library/test_library_multiselect_state.py
git commit -m "feat(export): checked/select_mode/selected_count in Library row state builders (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Media canvas + screen wiring (reference slice)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_multiselect_media.py` (create)

**Interfaces:**
- Consumes: `RowSelection` (Task 2); the media state's `checked`/`select_mode`/`selected_count` (Task 3); `ExportScope`, `_open_library_export_canvas` (existing).
- Produces: screen fields `self._library_media_row_selection: RowSelection`, `self._library_media_select_mode: bool`; handlers for `#library-media-select-toggle`, `#library-media-select-all`, `#library-media-select-clear`, `#library-media-export-selected`; select-aware `handle_library_media_row`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_multiselect_media.py` (uses `SimpleNamespace` fakes; the row-press and export handlers are plain methods that only touch the fields listed):
```python
from types import SimpleNamespace
import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def _media_fake(select_mode):
    return SimpleNamespace(
        _library_media_select_mode=select_mode,
        _library_media_row_selection=RowSelection("media"),
        _opened=[],
        _refreshed=0,
        _viewer_opened=[],
    )


def test_row_press_in_select_mode_toggles_not_opens():
    fake = _media_fake(select_mode=True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    fake._open_library_media_viewer = lambda mid: fake._viewer_opened.append(mid)
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    assert fake._library_media_row_selection.is_selected("7")
    assert fake._viewer_opened == []          # viewer NOT opened
    assert fake._refreshed == 1


def test_row_press_normal_mode_opens_viewer():
    fake = _media_fake(select_mode=False)
    fake._open_library_media_viewer = lambda mid: fake._viewer_opened.append(mid)
    event = SimpleNamespace(button=SimpleNamespace(media_id="7"), stop=lambda: None)
    LibraryScreen.handle_library_media_row(fake, event)
    assert fake._viewer_opened == ["7"]
    assert not fake._library_media_row_selection.is_selected("7")


@pytest.mark.asyncio
async def test_export_selected_builds_ids_scope():
    fake = _media_fake(select_mode=True)
    fake._library_media_row_selection.select_all(["3", "1", "2"])
    async def _open(scope): fake._opened.append(scope)
    fake._open_library_export_canvas = _open
    event = SimpleNamespace(stop=lambda: None)
    await LibraryScreen.handle_library_media_export_selected(fake, event)
    assert fake._opened == [ExportScope(kind="media", ids=("1", "2", "3"))]
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_multiselect_media.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `handle_library_media_export_selected` doesn't exist / `_library_media_row_selection` AttributeError.

- [ ] **Step 3: Screen state fields**

In `library_screen.py.__init__` (alongside `self._selected_media_id`, ~:561-579) add:
```python
        self._library_media_select_mode: bool = False
        self._library_media_row_selection = RowSelection("media")
```
Add the import near the other Library imports:
```python
from tldw_chatbook.Library.row_selection import RowSelection
```

- [ ] **Step 4: Feed select state into the media build + reconcile**

Change `_build_library_media_state` (~:2843) to pass select state and reconcile the set to the rendered rows:
```python
    def _build_library_media_state(self) -> LibraryMediaCanvasState:
        """Build the media canvas display state from local records."""
        state = build_library_media_state(
            self._local_source_records.get("media", ()),
            active_type=self._library_media_type_filter,
            selected_id=self._selected_media_id,
            select_mode=self._library_media_select_mode,
            selected_ids=self._library_media_row_selection.ids,
        )
        if self._library_media_select_mode:
            self._library_media_row_selection.reconcile(r.media_id for r in state.rows)
        return state
```

- [ ] **Step 5: Select-aware row press + the four action handlers**

Replace `handle_library_media_row` (~:5311) so a press toggles in select mode:
```python
    @on(Button.Pressed, ".library-media-row")
    def handle_library_media_row(self, event: Button.Pressed) -> None:
        """Select mode: toggle the row's checkbox. Normal mode: open the viewer."""
        event.stop()
        media_id = str(getattr(event.button, "media_id", "") or "")
        if self._library_media_select_mode:
            self._library_media_row_selection.toggle(media_id)
            self.refresh(recompose=True)
            return
        self._open_library_media_viewer(media_id)
```
Add (next to the other media handlers):
```python
    @on(Button.Pressed, "#library-media-select-toggle")
    def handle_library_media_select_toggle(self, event: Button.Pressed) -> None:
        """Enter/exit media select mode; entering starts from an empty set."""
        event.stop()
        self._library_media_select_mode = not self._library_media_select_mode
        self._library_media_row_selection.clear()
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-select-all")
    def handle_library_media_select_all(self, event: Button.Pressed) -> None:
        event.stop()
        rows = self._build_library_media_state().rows
        self._library_media_row_selection.select_all(r.media_id for r in rows)
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-select-clear")
    def handle_library_media_select_clear(self, event: Button.Pressed) -> None:
        event.stop()
        self._library_media_row_selection.clear()
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-export-selected")
    async def handle_library_media_export_selected(self, event: Button.Pressed) -> None:
        event.stop()
        await self._open_library_export_canvas(self._library_media_row_selection.export_scope())
```

- [ ] **Step 6: Clear selection + exit select mode on filter change**

In `handle_library_media_type_filter_pressed` (~:5285), after setting `self._library_media_type_filter = ...` and before `self.refresh(recompose=True)`, add:
```python
        self._library_media_select_mode = False
        self._library_media_row_selection.clear()
```

- [ ] **Step 7: Canvas — Select toggle, glyph, action row, hide Export in select mode**

In `library_media_canvas.py compose()`, replace the existing `Export…` button block (~:60-66) with this — build the Export button as a variable so its `.display` can be toggled off in select mode, add the Select toggle, and add an action row shown only in select mode:
```python
        select_mode = getattr(self.canvas, "select_mode", False)
        export_btn = Button("Export…", id="library-media-export",
                            classes="library-canvas-action", compact=True)
        export_btn.display = not select_mode
        yield export_btn
        rendered_count = len(self.canvas.rows)   # NOT self.canvas.count (that is the
                                                 # pre-filter total; only rendered rows
                                                 # are selectable). Conversations state
                                                 # has no `.count` field, so len(rows) is
                                                 # also the portable idiom for the siblings.
        select_btn = Button("Done" if select_mode else "Select",
                            id="library-media-select-toggle",
                            classes="library-canvas-action", compact=True)
        select_btn.disabled = rendered_count == 0
        yield select_btn
        if select_mode:
            action_row = Horizontal(classes="ds-toolbar")
            action_row.styles.height = "auto"
            with action_row:
                yield Static(f"{self.canvas.selected_count} selected",
                             id="library-media-selected-count", markup=False)
                yield Button(f"Select all {rendered_count} shown",
                             id="library-media-select-all",
                             classes="library-canvas-action", compact=True)
                yield Button("Clear", id="library-media-select-clear",
                             classes="library-canvas-action", compact=True)
                export_selected = Button("Export selected",
                                         id="library-media-export-selected",
                                         classes="library-canvas-action", compact=True)
                export_selected.disabled = self.canvas.selected_count == 0
                yield export_selected
```
(Add `from textual.containers import Horizontal` and `from textual.widgets import Static` to the imports if not already present.) In the row loop (~:80), pick the glyph from checked in select mode:
```python
                if select_mode:
                    marker = "☑" if row.checked else "☐"
                else:
                    marker = "▸" if row.selected else " "
```

- [ ] **Step 8: Run to verify it passes + a canvas smoke**

Run the Step-1 tests:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_multiselect_media.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Add a canvas smoke to the same file (mount `LibraryMediaCanvas` with a select-mode state via `app.run_test()`, assert `#library-media-select-all` exists and `#library-media-export-selected` is disabled at 0 selected). Run again. Then the import smoke:
```
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.UI.Screens.library_screen; print('import ok')"
```
Expected: PASS + `import ok`.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_multiselect_media.py
git commit -m "feat(export): media multi-select mode + export selected (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Conversations canvas + screen wiring

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: `Tests/UI/test_library_multiselect_conversations.py` (create)

Mirror Task 4 for conversations. **Interfaces produced:** `self._library_conversations_select_mode`, `self._library_conversations_row_selection = RowSelection("conversations")`; handlers for `#library-conversations-select-toggle`/`-select-all`/`-select-clear`/`-export-selected`; select-aware `handle_library_conversation_row`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_multiselect_conversations.py` — same shape as the media tests, using `conversation_id`, `RowSelection("conversations")`, `handle_library_conversation_row`, `handle_library_conversations_export_selected`, and asserting the normal-mode path sets `_selected_conversation_id` + `_library_selected_row_id` (rather than opening a viewer). Concretely:
```python
from types import SimpleNamespace
import pytest
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen, LIBRARY_ROW_BROWSE_CONVERSATIONS
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def _fake(select_mode):
    return SimpleNamespace(
        _library_conversations_select_mode=select_mode,
        _library_conversations_row_selection=RowSelection("conversations"),
        _selected_conversation_id="", _library_selected_row_id="", _refreshed=0, _opened=[],
    )


def test_convo_row_select_mode_toggles():
    fake = _fake(True); fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    ev = SimpleNamespace(button=SimpleNamespace(conversation_id="c5"), stop=lambda: None)
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._library_conversations_row_selection.is_selected("c5")
    assert fake._selected_conversation_id == ""     # did NOT open/select the detail
    assert fake._refreshed == 1


def test_convo_row_normal_mode_selects():
    fake = _fake(False); fake.refresh = lambda **k: None
    ev = SimpleNamespace(button=SimpleNamespace(conversation_id="c5"), stop=lambda: None)
    LibraryScreen.handle_library_conversation_row(fake, ev)
    assert fake._selected_conversation_id == "c5"
    assert fake._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS


@pytest.mark.asyncio
async def test_convo_export_selected_scope():
    fake = _fake(True); fake._library_conversations_row_selection.select_all(["c2", "c1"])
    async def _open(s): fake._opened.append(s)
    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_conversations_export_selected(fake, SimpleNamespace(stop=lambda: None))
    assert fake._opened == [ExportScope(kind="conversations", ids=("c1", "c2"))]
```

- [ ] **Step 2: Run to verify it fails**

Run the new file; Expected: FAIL (missing handler/field).

- [ ] **Step 3: Screen fields + build + handlers**

In `__init__`: `self._library_conversations_select_mode: bool = False` and `self._library_conversations_row_selection = RowSelection("conversations")`. In `_build_library_conversations_state` (~:2835) pass `select_mode=self._library_conversations_select_mode, selected_ids=self._library_conversations_row_selection.ids` and, when in select mode, `self._library_conversations_row_selection.reconcile(r.conversation_id for r in state.rows)`. Replace `handle_library_conversation_row` (~:5271):
```python
    @on(Button.Pressed, ".library-conversation-row")
    def handle_library_conversation_row(self, event: Button.Pressed) -> None:
        event.stop()
        conversation_id = str(getattr(event.button, "conversation_id", "") or "")
        if self._library_conversations_select_mode:
            self._library_conversations_row_selection.toggle(conversation_id)
            self.refresh(recompose=True)
            return
        if conversation_id:
            self._selected_conversation_id = conversation_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        self.refresh(recompose=True)
```
Add `handle_library_conversations_select_toggle` / `_select_all` / `_select_clear` / `_export_selected` mirroring Task 4 Step 5 (ids `#library-conversations-select-*`, using `_library_conversations_row_selection`; select-all builds `self._build_library_conversations_state().rows`). Clear select mode + selection on filter change in `handle_library_conversations_filter_submitted` (~:7634) before its `self.refresh(recompose=True)`.

- [ ] **Step 4: Canvas**

In `library_conversations_canvas.py compose()`, apply the same Export-hide + Select toggle + action-row + glyph changes as Task 4 Step 7, with `#library-conversations-*` ids and `row.conversation_id`.

- [ ] **Step 5: Run to verify it passes + import smoke**

Run the new test file + the `import tldw_chatbook.UI.Screens.library_screen` smoke. Expected: PASS + `import ok`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_multiselect_conversations.py
git commit -m "feat(export): conversations multi-select mode + export selected (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Notes canvas + screen wiring + backlog Done

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `backlog/tasks/task-159 - Multi-select-row-export-for-the-Library.md`
- Test: `Tests/UI/test_library_multiselect_notes.py` (create)

Mirror Task 4/5 for notes, with two notes-specific differences: **(a)** notes list rows had no marker before, so add the ☑/☐ glyph in `_compose_list` (normal mode shows no marker — keep today's markerless label); **(b)** the notes row handler is `async` and flushes; entering notes select mode must `await self._flush_library_note_save()` first.

**Interfaces produced:** `self._library_notes_select_mode`, `self._library_notes_row_selection = RowSelection("notes")`; `#library-notes-select-toggle`/`-select-all`/`-select-clear`/`-export-selected`; select-aware `handle_library_notes_row`. The notes canvas needs `select_mode`/`selected_count` — pass them from the list_state (Task 3 already added them to `LibraryNotesListState`).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_multiselect_notes.py`:
```python
from types import SimpleNamespace
import pytest
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.row_selection import RowSelection
from tldw_chatbook.Library.library_export_scope import ExportScope


def _fake(select_mode):
    return SimpleNamespace(
        _library_notes_select_mode=select_mode,
        _library_notes_row_selection=RowSelection("notes"),
        _selected_note_id="", _library_note_dirty=False, _refreshed=0, _opened=[], _flushed=0,
        _library_notes_view="list",
    )


@pytest.mark.asyncio
async def test_notes_row_select_mode_toggles_and_does_not_open_editor():
    fake = _fake(True)
    fake.refresh = lambda **k: setattr(fake, "_refreshed", fake._refreshed + 1)
    async def _flush(): fake._flushed += 1
    fake._flush_library_note_save = _flush
    ev = SimpleNamespace(button=SimpleNamespace(note_id="n9"), stop=lambda: None)
    await LibraryScreen.handle_library_notes_row(fake, ev)
    assert fake._library_notes_row_selection.is_selected("n9")
    assert fake._library_notes_view == "list"      # editor NOT opened
    assert fake._refreshed == 1


@pytest.mark.asyncio
async def test_notes_export_selected_scope():
    fake = _fake(True); fake._library_notes_row_selection.select_all(["n2", "n1"])
    async def _open(s): fake._opened.append(s)
    fake._open_library_export_canvas = _open
    await LibraryScreen.handle_library_notes_export_selected(fake, SimpleNamespace(stop=lambda: None))
    assert fake._opened == [ExportScope(kind="notes", ids=("n1", "n2"))]
```

- [ ] **Step 2: Run to verify it fails**

Run the new file; Expected: FAIL (missing handler/field).

- [ ] **Step 3: Screen fields + select-aware handler + actions**

`__init__`: `self._library_notes_select_mode: bool = False`, `self._library_notes_row_selection = RowSelection("notes")`. In the notes compose branch (~:2690), pass `select_mode`/`selected_ids` into `build_library_notes_list_state(...)` and reconcile in select mode. Replace `handle_library_notes_row` (~:6359) so select mode toggles (still `async`, still flushes so a dirty edit isn't stranded):
```python
    @on(Button.Pressed, ".library-notes-row")
    async def handle_library_notes_row(self, event: Button.Pressed) -> None:
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        note_id = str(getattr(event.button, "note_id", "") or "")
        if self._library_notes_select_mode:
            self._library_notes_row_selection.toggle(note_id)
            self.refresh(recompose=True)
            return
        # ... existing "open editor" body unchanged ...
```
Add `handle_library_notes_select_toggle` (async — `await self._flush_library_note_save()` before flipping the flag, then clear + recompose), `_select_all`, `_select_clear`, `_export_selected` (mirror Task 4, ids `#library-notes-select-*`, using `_library_notes_row_selection` and the notes list_state's rows). Clear select mode + selection on sort/filter change in `handle_library_notes_sort` (~:5377) and `handle_library_notes_filter` (~:5388) (and the empty-submit branch), before their recompose.

- [ ] **Step 4: Canvas (`_compose_list`)**

In `library_notes_canvas.py._compose_list`, add the Select toggle to the existing `ds-toolbar` (and hide `#library-notes-export` in select mode), add the action row (shown in select mode) with `#library-notes-select-all/-clear/-export-selected` + the `N selected` Static, and in the row loop prefix the label with `("☑ " if row.checked else "☐ ")` **only when** `list_state.select_mode` (normal mode keeps today's markerless label). Read `select_mode`/`selected_count` from `list_state`.

- [ ] **Step 5: Run to verify it passes + import smoke + mark backlog Done**

Run the notes test file + `import tldw_chatbook.UI.Screens.library_screen` smoke. Then tick ACs + status in the backlog file:
```bash
perl -0pi -e 's/- \[ \] (#\d)/- [x] $1/g' "backlog/tasks/task-159 - Multi-select-row-export-for-the-Library.md"
perl -0pi -e 's/^status: .*/status: Done/m' "backlog/tasks/task-159 - Multi-select-row-export-for-the-Library.md"
```
Add a short `## Implementation Notes` section (ExportScope.ids override; RowSelection helper; per-source select mode across the three canvases; reconcile/WYSIWYG; recompose-based like every other Library interaction).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_multiselect_notes.py "backlog/tasks/task-159 - Multi-select-row-export-for-the-Library.md"
git commit -m "feat(export): notes multi-select mode + export selected; task 159 done (159)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 6)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Library/ Tests/UI/test_library_multiselect_media.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_library_multiselect_notes.py Tests/UI/test_library_shell.py \
  -q -p no:cacheprovider -o addopts="" --timeout=600 --timeout-method=thread
```
Then the whole-branch review (opus) and finishing-a-development-branch. Served-TUI visual QA of select mode (glyph, action row, Export selected → export canvas showing "Selected media · N items") is worthwhile but optional.
