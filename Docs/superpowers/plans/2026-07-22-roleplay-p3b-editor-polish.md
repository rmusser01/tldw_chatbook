# Roleplay P3b — Editor Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the character + persona editors — avatar thumbnail, alt-greetings list editor, live per-field validation, save-in-place with dirty re-arm, and the persona `is_active`/`mode`/net-new `personality_traits` fields.

**Architecture:** Five mostly-independent tasks over two widget files (`personas_character_editor_widget.py`, `persona_profile_editor_widget.py`) + their screen seams in `personas_screen.py`, reusing the existing pure image-render primitive (`ConsoleImageRenderCache`) and mirroring the Lore list-editor for greetings. Task 1 (save-in-place) is foundational — it changes the editor lifecycle the others build on.

**Tech Stack:** Python 3.11+, Textual (Container/DataTable/Switch/Select/TextArea), Pydantic DTOs, PIL/rich-pixels/textual-image, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-22-roleplay-p3b-editor-polish-design.md` (committed `a472545a0`).

## Global Constraints

- **NO ChaChaNotes SQLite migration** (stays v22; no new DB tables/columns). The net-new persona `personality_traits` is a **Pydantic DTO + file-backed JSON** field (new key; older profiles default `""`), NOT a DB migration.
- **Reuse `ConsoleImageRenderCache` + `resolve_default_mode` VERBATIM** (`Chat/console_image_view.py`) — no new decoder. Avatar decode runs **off-thread** (`asyncio.to_thread`), driven from the screen (it has `app.config`), never on the widget.
- **Preserve alt-greetings multi-line fidelity** — a greeting containing newlines must round-trip byte-identical. The list model replaces the newline-blob + `_loaded_greetings_text` diff rule; it must not regress it.
- **Save-in-place carries the new optimistic-lock version** — character: re-read the saved record (`load_character(saved_id)` → handler `current_character_data`, or `fetch_character_by_id`) *before* reading its version; persona: the returned `saved` dict already carries the incremented version. Fall back to today's flip-to-card + notify if a character record can't be re-read.
- **Create → edit transition** after saving a *new* entity: `mark_saved` sets the new id, the finisher flips `_edit_mode` `create`→`edit`, refreshes the list + marks the row active, and stays in the editor.
- **Apply dirty-re-arm + validation to BOTH editors** (their state machines are duplicated). Extract a shared mixin only if it stays clearly simpler; otherwise mirror the change in both.
- **Characters-only avatars** — no persona image field.
- **Save is blocked** (no `*SaveRequested` posted) while an **error**-level validation finding stands; warnings don't block. Validation runs debounced on-change AND authoritatively at Save.
- All screen DB/decode I/O off-thread, wrapped so errors notify (never crash the worker).
- **Implementers stage ONLY their task's files** — never `git add -A`, never `.superpowers/`.
- **No background/broad test sweeps; never broad-`pkill` pytest** — scope to this worktree.
- **CONCURRENT-SESSION HAZARD:** other sessions are actively editing `personas_screen.py` (RP-UX tasks 425–445, e.g. Start-Chat #754). Keep changes localized to the editor widgets + the save/validation/avatar seams; expect a non-trivial rebase before merge. Branch off the LATEST `origin/dev` (`bf21cfb4b` at plan time — re-verify).
- **Test command** (prefix every run; run ONLY your task's files):
  ```bash
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <targets> \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```
  (venv in MAIN checkout; imports resolve to worktree source. UI tests are slow — the 300s timeout is deliberate.)

---

## File Structure

**Modified:**
- `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py` — `mark_saved` (T1), avatar thumbnail API + Remove (T2), greetings list editor (T3), live validation (T4), Advanced re-tune (T5).
- `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` — `mark_saved` (T1), live validation (T4), `is_active`/`mode`/`personality_traits` fields (T5).
- `tldw_chatbook/UI/Screens/personas_screen.py` — save-in-place finishers (T1), avatar render worker (T2), persona save-handler DTO threading (T5).
- `tldw_chatbook/tldw_api/character_persona_schemas.py` — add `personality_traits` to `PersonaProfileCreate`/`Update` (T5).

**Tests (create/extend under `Tests/UI/` and `Tests/Character_Chat/`):** grep `Tests/` for existing `PersonasCharacterEditorWidget` / `PersonaProfileEditorWidget` / personas-screen host-app fixtures first and mirror them (host `App` that mounts the widget; screen fixture with a real `CharactersRAGDB` + scope service).

---

## Task 1: Dirty re-arm + save-in-place (both editors + finishers)

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`, `persona_profile_editor_widget.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_after_character_save` ~:4804, `_after_profile_save` ~:4913)
- Test: `Tests/UI/test_personas_editor_save_in_place.py`

**Interfaces:**
- Produces: `PersonasCharacterEditorWidget.mark_saved(record: dict) -> None`; `PersonaProfileEditorWidget.mark_saved(record: dict) -> None`.

- [ ] **Step 1: Write the failing widget test (re-arm)**

Create `Tests/UI/test_personas_editor_save_in_place.py`. Mirror the existing editor host-app harness (grep `Tests/UI` for `PersonasCharacterEditorWidget`).

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import EditorContentChanged

pytestmark = pytest.mark.asyncio


class _CharHost(App):
    def __init__(self):
        super().__init__(); self.dirty = 0
    def compose(self) -> ComposeResult:
        yield PersonasCharacterEditorWidget()
    def on_editor_content_changed(self, m): self.dirty += 1


async def test_character_mark_saved_rearms_dirty():
    app = _CharHost()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"id": 5, "name": "A", "version": 1})
        await pilot.pause()
        ed.query_one("#personas-char-editor-name", Input).value = "A2"  # first edit
        await pilot.pause()
        assert app.dirty == 1
        # Simulate a save landing: new version, re-arm.
        ed.mark_saved({"id": 5, "name": "A2", "version": 2})
        await pilot.pause()
        assert ed._dirty_posted is False
        ed.query_one("#personas-char-editor-name", Input).value = "A3"  # second edit
        await pilot.pause()
        assert app.dirty == 2  # re-armed → re-posted
        assert ed.get_character_data()["version"] == 2  # new optimistic-lock version
```

Add the persona analog (`PersonaProfileEditorWidget`, ids `#personas-editor-name`, `mark_saved({"id":"p1","name":"B","version":2})`, assert `_dirty_posted is False` and a second edit re-posts, and `collect()["version"] == 2`).

- [ ] **Step 2: Run — verify it fails**

Run the test file. Expected: FAIL — `AttributeError: 'PersonasCharacterEditorWidget' object has no attribute 'mark_saved'`.

- [ ] **Step 3: Implement `mark_saved` on both widgets**

Character editor — add after `load_character` (~:241):

```python
    def mark_saved(self, record: Dict[str, Any]) -> None:
        """Re-baseline dirty state to a just-persisted record (save-in-place).

        Adopts the saved record as the new base (so the next Save carries the
        new ``version`` and any DB-normalized keys), rebaselines the greeting
        fidelity anchors, resets the dirty snapshot from the CURRENT form (which
        already shows the saved values), and clears validation. Does NOT
        repopulate the form — the user's saved edits stay on screen.
        """
        self._character_data = dict(record or {})
        self._loaded_greetings = [
            str(g) for g in (self._character_data.get("alternate_greetings") or [])
        ]
        self._loaded_greetings_text = "\n".join(self._loaded_greetings)
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self.query_one("#personas-char-editor-validation", Static).update("")
```

Persona editor — add after `new_persona` (~:109):

```python
    def mark_saved(self, record: Dict[str, Any]) -> None:
        """Re-baseline dirty state to a just-persisted persona (save-in-place)."""
        self._persona_id = str(record.get("id", "")) or self._persona_id
        self._version = record.get("version", self._version)
        self._loaded_snapshot = self._form_snapshot()
        self._dirty_posted = False
        self.query_one("#personas-editor-validation", Static).update("")
```

> Note for T3: when the greetings list editor lands, the character `mark_saved` rebaselines `self._greetings` from `record["alternate_greetings"]` instead of `_loaded_greetings*`. T3 updates this method.

- [ ] **Step 4: Run — verify widget tests pass**

Run the test file. Expected: PASS (both widget re-arm tests).

- [ ] **Step 5: Write the failing screen integration test (save-in-place)**

Append a screen test using the personas-screen host-app fixture (grep `Tests/UI` for the fixture that wires a real `CharactersRAGDB`). It should: enter Characters mode, open New editor, type a name, press Save, then assert (a) the editor center (`#ccp-character-editor-view`) is still displayed (NOT the card), (b) `screen.state.has_unsaved_changes is False`, (c) `screen._edit_mode == "edit"` (create→edit), (d) a second field edit re-flags `has_unsaved_changes` True. Shape it to the real fixture's helpers; keep those four assertions.

- [ ] **Step 6: Run — verify it fails**

Expected: FAIL — the editor flips to the card (`#ccp-character-card-view` displayed) and/or `_edit_mode == "view"`.

- [ ] **Step 7: Make the finishers save-in-place**

In `personas_screen.py` `_after_character_save` (~:4804): read the persisted record BEFORE deciding, keep the editor open, transition mode, call `mark_saved`. Replace the tail (from ~:4818 `self._edit_mode = "view"` through ~:4841) with logic equivalent to:

```python
        # Re-read the just-persisted record (authoritative version). load the
        # card into the handler so _full_character_record returns fresh data.
        await self.character_handler.load_character(saved_id)
        saved_record = self._full_character_record(saved_id)
        editor = self.query_one(PersonasCharacterEditorWidget)
        if saved_record is None:
            # Could not re-read → fall back to today's flip-to-card so we never
            # leave the editor holding a stale version.
            self._edit_mode = "view"
            self._show_center("#ccp-character-card-view")
        else:
            self._edit_mode = "edit"          # create→edit stays in the editor
            editor.mark_saved(saved_record)
            self._show_center("#ccp-character-editor-view")
        self._set_active_row_unsaved(False)
        await self.character_handler.refresh_character_list()
        name = str((saved_record or {}).get("name") or submitted_name or "Saved character")
        self.state.select_entity(entity_kind="character", entity_id=saved_id, entity_name=name)
        self.state.has_unsaved_changes = False
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=name, kind="character", authority="Local")
        inspector.set_unsaved(False)
        inspector.show_validation(())
        self._sync_inspector_console_actions()
        self.query_one(PersonasLibraryPane).mark_active_row("character", saved_id)
        self._sync_title_and_console_actions()
        self._notify("Character saved.", severity="information")
```

(Keep the existing early-return branch at ~:4807 for the "user left the screen" case. Confirm the exact surrounding names when you read the method — `_character_editor_generation` bump, `_focus_library_list` — preserve them where they still apply; the editor now stays focused, so drop the `_focus_library_list` call on the stay-in-editor branch.)

In `_after_profile_save` (~:4913): mirror it — the returned `saved` dict already has the version, so:

```python
        editor = self.query_one(PersonaProfileEditorWidget)
        self._edit_mode = "edit"
        editor.mark_saved(saved)
        self._show_center("#ccp-persona-editor-view")
        # ... keep the existing has_unsaved_changes=False / select_entity /
        #     inspector / _render_profile_rows lines, but REMOVE the
        #     _show_center("#ccp-persona-card-view") and the card show_persona.
```

- [ ] **Step 8: Run — verify the screen test passes**

Run the test file. Expected: PASS. Also run the existing personas character + persona editor/screen suites you can name (grep them) to confirm no regression in the save flow.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_editor_save_in_place.py
git commit -m "feat(personas): P3b Task 1 — save-in-place + dirty re-arm (mark_saved, both editors)"
```

---

## Task 2: Avatar thumbnail preview (character editor)

**Files:**
- Modify: `personas_character_editor_widget.py` (avatar row ~:200, DEFAULT_CSS ~:89)
- Modify: `personas_screen.py` (avatar upload chain ~:3841/3889; editor-open path)
- Test: `Tests/UI/test_personas_character_editor_avatar.py`

**Interfaces:**
- Produces (widget): `set_avatar_thumbnail(renderable: object | None) -> None` (mounts the renderable, or the text fallback when None); `current_avatar_bytes() -> bytes | None`; a Remove button `#personas-char-editor-avatar-remove` posting a new `CharacterImageRemoveRequested()` message (add to `personas_pane_messages.py`).
- Consumes (screen): `ConsoleImageRenderCache`, `resolve_default_mode` from `Chat/console_image_view.py`; the graphics-vs-pixels mount from `Widgets/Console/console_transcript.py:_image_row_widget`.

- [ ] **Step 1: Write the failing widget test**

```python
import pytest
from textual.app import App, ComposeResult
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterImageRemoveRequested,
)
from rich_pixels import Pixels
from PIL import Image

pytestmark = pytest.mark.asyncio


class _Host(App):
    def __init__(self): super().__init__(); self.removed = 0
    def compose(self) -> ComposeResult: yield PersonasCharacterEditorWidget()
    def on_character_image_remove_requested(self, m): self.removed += 1


async def test_thumbnail_mounts_and_text_fallback():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "image": b"x"})
        await pilot.pause()
        # None → text fallback present, no image widget
        ed.set_avatar_thumbnail(None)
        await pilot.pause()
        assert ed.query("#personas-char-editor-avatar-thumb").is_empty
        # A pixels renderable → a widget is mounted in the avatar row
        px = Pixels.from_image(Image.new("RGBA", (8, 8)))
        ed.set_avatar_thumbnail(px)
        await pilot.pause()
        assert not ed.query("#personas-char-editor-avatar-thumb").is_empty


async def test_current_avatar_bytes_and_remove():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        ed.load_character({"name": "A", "image": b"abc"})
        await pilot.pause()
        assert ed.current_avatar_bytes() == b"abc"
        await pilot.click("#personas-char-editor-avatar-remove")
        await pilot.pause()
        assert app.removed == 1
```

- [ ] **Step 2: Run — verify it fails**

Expected: FAIL (`set_avatar_thumbnail`/`current_avatar_bytes`/`#personas-char-editor-avatar-remove` absent; `CharacterImageRemoveRequested` unimportable).

- [ ] **Step 3: Add the message + widget API**

In `personas_pane_messages.py`, add `class CharacterImageRemoveRequested(Message): ...` (mirror `CharacterImageUploadRequested`). In the character editor:
- Add a Remove button to the avatar row and a thumbnail mount container. Change the avatar row (compose ~:200) to include a `Static("", id="personas-char-editor-avatar-thumb")` holder and the Remove button:

```python
            with Horizontal(id="personas-char-editor-avatar-row"):
                yield Static("Avatar: none", id="personas-char-editor-avatar-status")
                yield Button("Upload", id="personas-char-editor-avatar-upload", classes="console-action-subdued")
                yield Button("Remove", id="personas-char-editor-avatar-remove", classes="console-action-subdued")
            yield Container(id="personas-char-editor-avatar-thumb")
```

- Grow the CSS: change `#personas-char-editor-avatar-row` off `height: 1` (keep the status/buttons on one line) and give `#personas-char-editor-avatar-thumb` a small box, e.g. `height: 10; max-width: 24; max-height: 10;` (a compact editor thumbnail, smaller than the 80×40 chat box).
- Methods:

```python
    def current_avatar_bytes(self) -> bytes | None:
        data = self._character_data.get("image")
        return data if isinstance(data, (bytes, bytearray)) else None

    def set_avatar_thumbnail(self, renderable: object | None) -> None:
        """Mount a prepared avatar renderable (pixels Static or graphics Image),
        or clear to the text status when None. The screen prepares the
        renderable off-thread; this only mounts it."""
        holder = self.query_one("#personas-char-editor-avatar-thumb", Container)
        holder.remove_children()
        if renderable is None:
            return
        from textual.widgets import Static as _S
        # A rich renderable (Pixels) mounts inside a Static; a Textual widget
        # (textual_image Image) mounts directly.
        from textual.widget import Widget as _W
        holder.mount(renderable if isinstance(renderable, _W) else _S(renderable))

    @on(Button.Pressed, "#personas-char-editor-avatar-remove")
    def _remove_avatar_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(CharacterImageRemoveRequested())
```

Import `Container` in the widget. Keep `set_avatar_image` (upload staging) as-is; the screen calls the render step after it.

- [ ] **Step 4: Run — verify widget tests pass**

Run the test file. Expected: PASS.

- [ ] **Step 5: Wire the screen render worker + Remove handler**

In `personas_screen.py`, add a screen-owned lazy `ConsoleImageRenderCache` + a render method (mirror `chat_screen.py:2727-2789`):

```python
    async def _render_character_editor_avatar(self) -> None:
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except QueryError:
            return
        data = editor.current_avatar_bytes()
        editor._set_avatar_status_from_record()  # keep the text status accurate
        if not data:
            editor.set_avatar_thumbnail(None)
            return
        token = self._character_editor_generation  # session-token guard
        from tldw_chatbook.Chat.console_image_view import (
            ConsoleImageRenderCache, resolve_default_mode,
        )
        if getattr(self, "_avatar_render_cache", None) is None:
            self._avatar_render_cache = ConsoleImageRenderCache()
        cache = self._avatar_render_cache
        mode = resolve_default_mode(self.app_instance.app_config)  # confirm accessor
        try:
            ok = await asyncio.to_thread(cache.prepare, "char-editor-avatar", bytes(data))
        except Exception:
            logger.opt(exception=True).debug("avatar decode failed")
            ok = False
        if token != self._character_editor_generation or not self.is_mounted:
            return  # a different editor session started while decoding
        renderable = None
        if ok:
            if mode == "graphics":
                try:
                    from textual_image.widget import Image as _Img
                    pil = cache.get_pil("char-editor-avatar")
                    renderable = _Img(pil) if pil is not None else None
                except Exception:
                    renderable = cache.get_pixels("char-editor-avatar")
            else:
                renderable = cache.get_pixels("char-editor-avatar")
        editor.set_avatar_thumbnail(renderable)
```

Call `_render_character_editor_avatar()` (via `run_worker(..., exit_on_error=False)` or awaited in an existing async handler) at: (a) the end of `_stage_character_avatar_from_path` (after `set_avatar_image`); (b) when the character editor is shown for edit/new (in the edit-requested / new handler, after `load_character`); and add the Remove handler:

```python
    @on(CharacterImageRemoveRequested)
    def _handle_character_image_remove(self, message) -> None:
        message.stop()
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except QueryError:
            return
        editor._character_data.pop("image", None)
        editor._mark_dirty()
        self.run_worker(self._render_character_editor_avatar(), exit_on_error=False)
```

(Confirm the exact app-config accessor — `self.app_instance.app_config` vs `self.app.config`; match what `chat_screen.py` uses for `resolve_default_mode`.)

- [ ] **Step 6: Add a screen render test + run**

Screen integration test (real screen): open the character editor for a character WITH an image → assert a widget mounts under `#personas-char-editor-avatar-thumb`; press Remove → assert it clears + `has_unsaved_changes` True. Run it + the widget tests. Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_character_editor_avatar.py
git commit -m "feat(personas): P3b Task 2 — character avatar thumbnail preview + Remove"
```

---

## Task 3: Alternate-greetings list editor (character editor)

**Files:**
- Modify: `personas_character_editor_widget.py` (Advanced section ~:197-199; `_populate_form` ~:266-270; `get_character_data` ~:338-344; `_form_snapshot` ~:405; `mark_saved` from T1)
- Test: `Tests/UI/test_personas_character_editor_greetings.py`

**Interfaces:** widget-local only (`_greetings: list[str]`); no new messages, no screen changes.

- [ ] **Step 1: Write the failing test (incl. the multi-line fidelity guarantee)**

```python
import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea, DataTable
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
pytestmark = pytest.mark.asyncio

class _Host(App):
    def compose(self) -> ComposeResult: yield PersonasCharacterEditorWidget()

async def test_greetings_load_add_delete_reorder_and_multiline_fidelity():
    app = _Host()
    async with app.run_test() as pilot:
        ed = app.query_one(PersonasCharacterEditorWidget)
        multi = "Hello there.\n\nA second paragraph."   # a greeting WITH newlines
        ed.load_character({"name": "A", "alternate_greetings": [multi, "Hi!"]})
        await pilot.pause()
        # unedited round-trip is byte-identical (the fidelity guarantee)
        assert ed.get_character_data()["alternate_greetings"] == [multi, "Hi!"]
        # add
        ed._greetings_add("Third")
        assert ed.get_character_data()["alternate_greetings"] == [multi, "Hi!", "Third"]
        # update index 1
        ed._greetings_update(1, "Hi again!")
        assert ed.get_character_data()["alternate_greetings"][1] == "Hi again!"
        # move index 2 up
        ed._greetings_move(2, -1)
        assert ed.get_character_data()["alternate_greetings"] == [multi, "Third", "Hi again!"]
        # delete index 0 (the multi-line one)
        ed._greetings_delete(0)
        assert ed.get_character_data()["alternate_greetings"] == ["Third", "Hi again!"]
```

(Use the widget-local mutation helpers directly for a deterministic test; add a second test that drives the buttons via `pilot.click` for the Add/Move/Delete ids + `RowHighlighted` selection.)

- [ ] **Step 2: Run — verify it fails**

Expected: FAIL — `_greetings_add`/table absent.

- [ ] **Step 3: Replace the TextArea with the list editor**

In compose, replace the alt-greetings `ds-field-row` (~:197-199) with:

```python
                with Vertical(classes="ds-field-row"):
                    yield Label("Alternate greetings")
                    yield DataTable(id="personas-char-editor-greetings-table", cursor_type="row")
                    yield TextArea(id="personas-char-editor-greeting-edit")
                    with Horizontal(classes="ds-toolbar"):
                        yield Button("Add", id="personas-char-editor-greeting-add", classes="console-action-subdued")
                        yield Button("Update", id="personas-char-editor-greeting-update", classes="console-action-subdued")
                        yield Button("Delete", id="personas-char-editor-greeting-delete", classes="console-action-subdued")
                        yield Button("Move up", id="personas-char-editor-greeting-move-up", classes="console-action-subdued")
                        yield Button("Move down", id="personas-char-editor-greeting-move-down", classes="console-action-subdued")
```

Import `DataTable`. Add `_greetings: List[str] = []` in `__init__`, a `_selected_greeting_index: int | None`, and on_mount register the table column (`add_column("Greeting", key="g")`). Implement (mirror `personas_lore_detail.py` for selection tracking):

```python
    @staticmethod
    def _greeting_preview(text: str) -> str:
        first = (text or "").splitlines()[0] if text else ""
        return (first[:60] + "…") if len(first) > 60 or "\n" in (text or "") else first

    def _render_greetings_table(self) -> None:
        table = self.query_one("#personas-char-editor-greetings-table", DataTable)
        table.clear()
        for i, g in enumerate(self._greetings):
            table.add_row(self._greeting_preview(g), key=str(i))

    def _greetings_add(self, text: str = "") -> None:
        self._greetings.append(text)
        self._render_greetings_table(); self._mark_dirty()

    def _greetings_update(self, index: int, text: str) -> None:
        if 0 <= index < len(self._greetings):
            self._greetings[index] = text
            self._render_greetings_table(); self._mark_dirty()

    def _greetings_delete(self, index: int) -> None:
        if 0 <= index < len(self._greetings):
            del self._greetings[index]
            self._render_greetings_table(); self._mark_dirty()

    def _greetings_move(self, index: int, offset: int) -> None:
        j = index + offset
        if 0 <= index < len(self._greetings) and 0 <= j < len(self._greetings):
            self._greetings[index], self._greetings[j] = self._greetings[j], self._greetings[index]
            self._render_greetings_table(); self._mark_dirty()
```

Wire the buttons (Add uses the edit TextArea text; Update/Delete/Move use `_selected_greeting_index`; on `Add`/select, load the greeting into the edit TextArea). Track selection on both `@on(DataTable.RowSelected)` and `@on(DataTable.RowHighlighted)` (set `_selected_greeting_index = int(event.row_key.value)` and load `self._greetings[i]` into the edit TextArea). **The greeting-edit TextArea must NOT feed the dirty `_field_changed` snapshot** (it's a scratch field) — exclude its id there, or gate it.

- [ ] **Step 4: Rewire load/save/snapshot + drop the fidelity-diff**

- `_populate_form` (~:266-270): replace `_loaded_greetings`/`_loaded_greetings_text`/`_area("alt-greetings").text = ...` with `self._greetings = [str(g) for g in (record.get("alternate_greetings") or [])]` then `self._render_greetings_table()`.
- `get_character_data` (~:338-344): replace the fidelity-diff block with `data["alternate_greetings"] = list(self._greetings)`.
- `_form_snapshot` (~:405): replace the `_area("alt-greetings").text` element with `tuple(self._greetings)` so dirty detection sees list mutations. (Add/Update/Delete/Move already call `_mark_dirty` directly, so this mainly keeps the snapshot self-consistent for the re-arm comparison.)
- `mark_saved` (from T1): rebaseline `self._greetings` from `record.get("alternate_greetings")` and drop the `_loaded_greetings*` lines (they no longer exist). Remove the now-dead `_loaded_greetings`/`_loaded_greetings_text` fields from `__init__`.

- [ ] **Step 5: Run — verify it passes**

Run the greetings test file + the T1 save-in-place file (mark_saved changed). Expected: PASS. Grep and run any existing test that asserted the old alt-greetings TextArea behavior and update/replace it (the newline-blob is gone).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py Tests/UI/test_personas_character_editor_greetings.py
git commit -m "feat(personas): P3b Task 3 — alt-greetings list editor (no newline-corruption)"
```

---

## Task 4: Live per-field validation (both editors)

**Files:**
- Modify: `personas_character_editor_widget.py`, `persona_profile_editor_widget.py`
- Test: extend the two editor test files

**Interfaces:**
- Produces: `validate() -> list[tuple[str, str, str]]` on both widgets (each item `(field_id, message, level)`, level ∈ `{"error","warning"}`); `_run_validation()` (debounced); `.is-invalid` CSS class toggled on the offending field rows.

- [ ] **Step 1: Write the failing test**

```python
async def test_blank_name_marks_field_and_blocks_save(<char host>):
    ed.load_character({"name": "A"})
    ed.query_one("#personas-char-editor-name", Input).value = ""
    await pilot.pause()  # debounced validation runs
    row = ed.query_one("#personas-char-editor-name").parent  # the ds-field-row
    assert row.has_class("is-invalid")
    # Save is blocked: no CharacterSaveRequested posted
    saves = []
    ... capture CharacterSaveRequested on the host ...
    await pilot.click("#personas-char-editor-save")
    assert saves == []
    # fix it → class clears, save posts
    ed.query_one("#personas-char-editor-name", Input).value = "A"
    await pilot.pause()
    assert not row.has_class("is-invalid")
```

Add: oversized-avatar error (stage `set_avatar_image(b"x"*(PERSONAS_AVATAR_MAX_BYTES+1))` → an error marks the avatar row); persona blank-name analog.

- [ ] **Step 2: Run — verify it fails.** Expected: FAIL (no `.is-invalid`, save still posts).

- [ ] **Step 3: Implement `validate()` + debounced runner + per-field marking**

On each editor: a `validate()` returning `(field_id, message, level)` tuples. Character checks: name required (error); avatar bytes > `PERSONAS_AVATAR_MAX_BYTES` (import the const, error); any greeting blank/whitespace (warning). Persona: name required (error). A `_run_validation()`:

```python
    _FIELD_ERROR_CLASS = "is-invalid"

    def _run_validation(self) -> list[tuple[str, str, str]]:
        findings = self.validate()
        invalid_ids = {fid for fid, _msg, level in findings if level == "error"}
        for fid in self._validated_field_ids():          # the set of field ids
            row = self.query_one(f"#{fid}").parent
            row.set_class(fid in invalid_ids, self._FIELD_ERROR_CLASS)
        self.show_validation(tuple(f"{fid}: {msg}" for fid, msg, _l in findings))
        return findings
```

Call `_run_validation()` from `_field_changed` (debounced via `self.set_timer`, cancel the prior timer) AND at Save. In `_save_pressed`, replace the inline name check with: `if any(l == "error" for _, _, l in self._run_validation()): return` before posting. Add a `.is-invalid` border/color rule to DEFAULT_CSS. Provide `_validated_field_ids()` returning the ids `validate()` can flag.

- [ ] **Step 4: Run — verify it passes.** Run both editor test files. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py Tests/UI/test_personas_character_editor_avatar.py Tests/UI/test_personas_editor_save_in_place.py
git commit -m "feat(personas): P3b Task 4 — live per-field validation (both editors)"
```

---

## Task 5: Persona fields (is_active/mode + net-new personality_traits) + grouping

**Files:**
- Modify: `tldw_chatbook/tldw_api/character_persona_schemas.py` (`PersonaProfileCreate` ~:556, `PersonaProfileUpdate` ~:570)
- Modify: `persona_profile_editor_widget.py` (compose/collect/load_persona/_form_snapshot/validate)
- Modify: `personas_screen.py` (`_handle_profile_save_requested` DTO build ~:4877-4897)
- Modify: `personas_character_editor_widget.py` (Advanced re-tune — minor)
- Test: `Tests/Character_Chat/test_persona_personality_traits_roundtrip.py`, extend the persona editor test

**Interfaces:** DTO gains `personality_traits: str`; editor collect/load gain `is_active`/`mode`/`personality_traits`.

**Verified:** the persona service is DTO-driven (`create_persona_profile` validates `PersonaProfileCreate` → `model_dump` → JSON store; `update_persona_profile` uses `model_dump(exclude_none=True)`; `_persona_profile_view` is a passthrough). **So the service methods need NO change** — adding `personality_traits` to the DTO + passing it from the save handler makes it persist and reload automatically. `mode` (`PersonaMode`) values: read the `PersonaMode` literal in `character_persona_schemas.py` for the exact options (default `session_scoped`).

- [ ] **Step 1: Write the failing service/store round-trip test**

```python
def test_personality_traits_persists_and_reloads(tmp_path):
    from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
    from tldw_chatbook.tldw_api.character_persona_schemas import PersonaProfileCreate
    svc = LocalCharacterPersonaService(<db>, persona_store_path=tmp_path / "p.json")  # match ctor
    created = svc.create_persona_profile(PersonaProfileCreate(name="P", personality_traits="brave, kind"))
    assert created["personality_traits"] == "brave, kind"
    # reload from a fresh service instance (proves the JSON store carries it)
    svc2 = LocalCharacterPersonaService(<db>, persona_store_path=tmp_path / "p.json")
    got = svc2.get_persona_profile(created["id"])
    assert got["personality_traits"] == "brave, kind"
```

(Confirm the `LocalCharacterPersonaService` ctor + how it reads the store on init; mirror an existing persona-service test.)

- [ ] **Step 2: Run — verify it fails.** Expected: FAIL — `PersonaProfileCreate` has no `personality_traits` (validation error or dropped).

- [ ] **Step 3: Add `personality_traits` to the DTOs**

In `character_persona_schemas.py`: `PersonaProfileCreate` add `personality_traits: str = ""`; `PersonaProfileUpdate` add `personality_traits: str | None = None`. (Freeform string, matching the character `personality` field — NOT the archetype `list[str]`.)

- [ ] **Step 4: Run — verify the round-trip test passes.** Expected: PASS.

- [ ] **Step 5: Surface the fields in the persona editor + thread through save**

Persona editor compose — add (after system-prompt, ~:72), importing `Switch`, `Select`:

```python
            with Vertical(classes="ds-field-row"):
                yield Label("Personality traits")
                yield TextArea(id="personas-editor-personality-traits")
            with Vertical(classes="ds-field-row"):
                yield Label("Mode")
                yield Select([(m, m) for m in PERSONA_MODES], id="personas-editor-mode", allow_blank=False)
            with Horizontal(classes="ds-field-row"):
                yield Label("Enabled")
                yield Switch(id="personas-editor-enabled", value=True)
```

(`PERSONA_MODES` = the `PersonaMode` literal values — import or inline them from the schema.) Wire:
- `load_persona`: set `#personas-editor-personality-traits` text = `data.get("personality_traits","")`; `#personas-editor-mode` value = `data.get("mode","session_scoped")`; `#personas-editor-enabled` value = `bool(data.get("is_active", True))`.
- `collect`: add `"personality_traits"`, `"mode"`, `"is_active"` (from the widgets).
- `_form_snapshot`: append the three values.
- `Switch.Changed` also marks dirty (add `@on(Switch.Changed)` → the same `_field_changed` guard path, or include Switch in the change handler).

Screen `_handle_profile_save_requested` (~:4877-4897) — include the fields when building the DTOs:

```python
                    request = PersonaProfileCreate(
                        id=data.get("id") or None, name=str(data.get("name") or ""),
                        description=data.get("description"),
                        mode=data.get("mode") or "session_scoped",
                        system_prompt=data.get("system_prompt"),
                        is_active=bool(data.get("is_active", True)),
                        personality_traits=str(data.get("personality_traits") or ""),
                    )
                    # ... and the Update branch: name/description/mode/system_prompt +
                    #     is_active=data.get("is_active"), personality_traits=data.get("personality_traits")
```

- [ ] **Step 6: Character Advanced re-tune (minor)**

Confirm the greetings list editor (T3) reads well inside `#personas-char-editor-advanced`; adjust ordering/labels only if needed. No field removals. (Keep this small — it's the least substantive item.)

- [ ] **Step 7: Add persona editor field tests + run**

Extend the persona editor test: `load_persona({"is_active": False, "mode": "session_scoped", "personality_traits": "x"})` populates the Switch/Select/TextArea; `collect()` returns them; toggling Enabled marks dirty; a persona save-in-place integration test persists + reloads the new fields (drives the handler threading). Run the persona test files + the service round-trip. Expected: PASS. Run `python -c "import tldw_chatbook.app"` (env prefix).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/tldw_api/character_persona_schemas.py tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py Tests/Character_Chat/test_persona_personality_traits_roundtrip.py Tests/UI/test_personas_editor_save_in_place.py
git commit -m "feat(personas): P3b Task 5 — persona is_active/mode + net-new personality_traits + field grouping"
```

*(If Task 5 gets unwieldy, split: 5a = DTO + service round-trip + save-handler threading; 5b = editor fields + grouping.)*

---

## Self-Review Notes (author)

- **Spec coverage:** avatar preview (T2), alt-greetings list (T3), live validation (T4), dirty re-arm + save-in-place incl. create→edit + version-carry (T1), persona is_active/mode + net-new personality_traits (T5), character Advanced re-tune (T5) — all mapped. No SQLite migration; personality_traits is DTO+JSON only.
- **Type consistency:** `mark_saved(record)` (T1) is updated by T3 (greetings rebaseline); `set_avatar_thumbnail`/`current_avatar_bytes`/`CharacterImageRemoveRequested` (T2); `_greetings` + `_greetings_{add,update,delete,move}` (T3); `validate() -> list[tuple[str,str,str]]` + `_run_validation` + `.is-invalid` (T4); `personality_traits`/`is_active`/`mode` on DTO+editor+handler (T5). Consistent across tasks.
- **Plan-time confirmations (read fresh):** the exact app-config accessor for `resolve_default_mode`; the `PersonaMode` literal values; the personas-screen host-app test fixture; the `LocalCharacterPersonaService` ctor for the store test; whether `_character_editor_generation`/`_focus_library_list` still apply on the stay-in-editor branch; the greeting-edit TextArea's exclusion from the dirty snapshot.
