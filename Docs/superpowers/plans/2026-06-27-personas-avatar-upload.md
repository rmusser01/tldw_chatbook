# Personas Avatar Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore avatar upload in the ds-native Personas character editor with staged-until-Save persistence.

**Architecture:** Keep the editor responsible for UI state and pending form data, while `PersonasScreen` owns file picker orchestration, path validation, notifications, and Save integration. Store staged avatar data as raw bytes in `_character_data["image"]`, then let the existing character create/update flow persist it on Save.

**Tech Stack:** Python 3.11+, Textual, existing `EnhancedFileOpen`, `path_validation.validate_path_simple`, existing CCP character persistence helpers, pytest mounted UI tests.

---

## Source Material

- Backlog task: `backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md`
- Approved design: `Docs/superpowers/specs/2026-06-27-personas-avatar-upload-design.md`
- Editor widget: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
- Pane messages: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Personas screen: `tldw_chatbook/UI/Screens/personas_screen.py`
- Widget tests: `Tests/UI/test_personas_character_widgets.py`
- Workbench tests: `Tests/UI/test_personas_workbench.py`

## Scope Check

This is one bounded UI workflow restoration.

- In scope: upload button, upload-request message, image-only file picker flow, path-based staging helper, raw-byte staging, avatar status update, dirty-state signaling, Save-path persistence, focused tests, Backlog task hygiene.
- Out of scope: avatar preview rendering, image resizing/compression, remove-avatar control, avatar URL editing, schema changes, sync changes, legacy editor resurrection, character-card import semantics.

ADR required: no
ADR path: N/A
Reason: this restores a scoped UI workflow using existing editor, file picker, dirty-state, and character persistence boundaries. No schema, storage policy, sync policy, provider/runtime boundary, or long-lived application architecture changes.

## File Structure

- Modify `backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md`
  - Track status, implementation plan, acceptance criteria, and implementation notes.

- Create/modify `Docs/superpowers/plans/2026-06-27-personas-avatar-upload.md`
  - Keep this implementation plan as the executable source of truth.

- Modify `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
  - Add `CharacterImageUploadRequested`.

- Modify `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
  - Render the upload button in the avatar row.
  - Add a public `set_avatar_image(image_data: bytes)` editor API.
  - Refactor dirty-state announcement into a helper used by text fields and avatar staging.
  - Keep avatar status derived from pending editor state.

- Modify `tldw_chatbook/UI/Screens/personas_screen.py`
  - Handle `CharacterImageUploadRequested`.
  - Open `EnhancedFileOpen` with image-only filters and `character_avatar_upload` context.
  - Add a path-based helper that validates, reads bytes off the UI thread, stages into the editor, and notifies on failure.

- Modify `Tests/UI/test_personas_character_widgets.py`
  - Replace the old "no upload button" regression.
  - Add widget-level tests for message emission, byte staging, status update, and dirty notification.

- Modify `Tests/UI/test_personas_workbench.py`
  - Add screen-level tests for path validation, editor mutation, dirty state, and Save persistence.

---

### Task 0: Backlog Task Hygiene

**Files:**
- Modify: `backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md`
- Modify: `Docs/superpowers/plans/2026-06-27-personas-avatar-upload.md`

- [ ] **Step 1: Confirm the task is In Progress**

Run:

```bash
backlog task 100 --plain
```

Expected: status is `In Progress`.

- [ ] **Step 2: Add the implementation plan summary to Backlog**

Run:

```bash
backlog task edit 100 --plan "1. Add the CharacterImageUploadRequested message, upload button, and editor-side staged image API.
2. Add widget tests proving the upload button emits the message and staged bytes update avatar status/dirty state.
3. Add the PersonasScreen file-picker handler and path-based avatar staging helper using validate_path_simple.
4. Add screen tests for valid staging, invalid paths/extensions, stale edit mode, and Save-path persistence.
5. Run focused Personas tests and git diff --check.
6. Update acceptance criteria and implementation notes before marking Done.

ADR required: no
ADR path: N/A
Reason: scoped UI workflow restoration using existing editor, file picker, dirty-state, and character persistence boundaries; no schema, sync, storage, provider/runtime, or application-architecture change."
```

Expected: `TASK-100` contains the plan summary and ADR check.

- [ ] **Step 3: Commit the planning artifacts**

Run:

```bash
git add "backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md" Docs/superpowers/plans/2026-06-27-personas-avatar-upload.md
git commit -m "docs: plan personas avatar upload"
```

Expected: commit succeeds with only the task file and plan staged.

---

### Task 1: Add Editor Upload Contract

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
- Test: `Tests/UI/test_personas_character_widgets.py`

- [ ] **Step 1: Write failing widget tests**

In `Tests/UI/test_personas_character_widgets.py`, update the imports:

```python
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterEditorCancelled,
    CharacterImageUploadRequested,
    CharacterSaveRequested,
    EditCharacterRequested,
    EditorContentChanged,
)
```

Replace `test_no_upload_button_and_avatar_status_is_read_only` with tests shaped like:

```python
async def test_upload_button_posts_image_upload_request(self):
    received = []

    class CaptureApp(WidgetApp):
        def on_character_image_upload_requested(
            self, message: CharacterImageUploadRequested
        ) -> None:
            received.append(message)

    app = CaptureApp()
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#personas-char-editor-avatar-upload", Button)
        button.press()
        await pilot.pause()
        assert len(received) == 1


async def test_set_avatar_image_stages_bytes_updates_status_and_marks_dirty(self):
    dirty_events = []

    class CaptureApp(WidgetApp):
        def on_editor_content_changed(self, message: EditorContentChanged) -> None:
            dirty_events.append(message)

    app = CaptureApp()
    async with app.run_test() as pilot:
        editor = pilot.app.query_one(PersonasCharacterEditorWidget)
        record = dict(CHARACTER)
        record.pop("image", None)
        editor.load_character(record)
        await pilot.pause()

        editor.set_avatar_image(b"\x89PNG staged")
        await pilot.pause()

        assert editor.get_character_data()["image"] == b"\x89PNG staged"
        assert (
            str(
                pilot.app.query_one(
                    "#personas-char-editor-avatar-status", Static
                ).renderable
            )
            == "Avatar: embedded"
        )
        assert len(dirty_events) == 1
```

Expected: tests fail because the message, button, and `set_avatar_image` API do not exist.

- [ ] **Step 2: Run the widget tests and confirm failure**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_widgets.py -q
```

Expected: failure references missing `CharacterImageUploadRequested`, missing `#personas-char-editor-avatar-upload`, or missing `set_avatar_image`.

- [ ] **Step 3: Add the message class**

In `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`, add:

```python
class CharacterImageUploadRequested(Message):
    """User requested to choose an image for the active character editor."""
```

- [ ] **Step 4: Add the editor button and styles**

In `PersonasCharacterEditorWidget.DEFAULT_CSS`, extend the avatar row styles:

```python
    PersonasCharacterEditorWidget #personas-char-editor-avatar-upload {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
```

In `compose()`, change the avatar row to:

```python
with Horizontal(id="personas-char-editor-avatar-row"):
    yield Static("Avatar: none", id="personas-char-editor-avatar-status")
    yield Button(
        "Upload",
        id="personas-char-editor-avatar-upload",
        classes="console-action-subdued",
    )
```

- [ ] **Step 5: Add editor staging helpers**

In `PersonasCharacterEditorWidget`, add helpers equivalent to:

```python
def _set_avatar_status_from_record(self) -> None:
    avatar = "embedded" if (
        self._character_data.get("image") or self._character_data.get("avatar")
    ) else "none"
    self.query_one("#personas-char-editor-avatar-status", Static).update(
        f"Avatar: {avatar}"
    )


def _mark_dirty(self) -> None:
    if self._loading or self._dirty_posted or self._loaded_snapshot is None:
        return
    self._dirty_posted = True
    self.post_message(EditorContentChanged())


def set_avatar_image(self, image_data: bytes) -> None:
    """Stage avatar image bytes for persistence on the next Save."""
    if not isinstance(image_data, bytes) or not image_data:
        raise ValueError("Avatar image data must be non-empty bytes.")
    self._character_data["image"] = image_data
    self._set_avatar_status_from_record()
    self._mark_dirty()
```

Then replace the repeated avatar-status update in `_populate_form()` with `self._set_avatar_status_from_record()`.

- [ ] **Step 6: Reuse `_mark_dirty()` for text/input changes**

Update `_field_changed()` so it still suppresses programmatic population, but delegates the one-time announcement:

```python
if self._loading or self._dirty_posted or self._loaded_snapshot is None:
    return
if self._form_snapshot() == self._loaded_snapshot:
    return
self._mark_dirty()
```

- [ ] **Step 7: Emit upload requests from the button**

Import `CharacterImageUploadRequested`, then add:

```python
@on(Button.Pressed, "#personas-char-editor-avatar-upload")
def _upload_avatar_pressed(self, event: Button.Pressed) -> None:
    event.stop()
    self.post_message(CharacterImageUploadRequested())
```

- [ ] **Step 8: Run widget tests until green**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_widgets.py -q
```

Expected: all widget tests pass.

- [ ] **Step 9: Commit Task 1**

Run:

```bash
git add Tests/UI/test_personas_character_widgets.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py
git commit -m "feat: stage character avatar uploads in editor"
```

Expected: commit succeeds.

---

### Task 2: Add Personas Screen Upload Flow

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Test: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Write failing screen tests for path staging**

In `Tests/UI/test_personas_workbench.py`, extend imports:

```python
from textual.widgets import Button, Static, Input

from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
```

Add tests near the import/export/path-based flow tests:

```python
async def test_stage_character_avatar_from_path_updates_editor_and_dirty_state(
    self, mock_app_instance, stub_characters, tmp_path
):
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"\x89PNG staged avatar")
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()

        await screen._stage_character_avatar_from_path(str(avatar))
        await pilot.pause()

        editor = screen.query_one(PersonasCharacterEditorWidget)
        assert editor.get_character_data()["image"] == b"\x89PNG staged avatar"
        assert (
            str(screen.query_one("#personas-char-editor-avatar-status", Static).renderable)
            == "Avatar: embedded"
        )
        assert screen.state.has_unsaved_changes is True
```

Add an invalid-extension regression:

```python
async def test_stage_character_avatar_rejects_unsupported_extension_without_mutation(
    self, mock_app_instance, stub_characters, tmp_path
):
    bad = tmp_path / "avatar.txt"
    bad.write_text("not an image")
    app = PersonasTestApp(mock_app_instance)
    notifications = TestImportExport._capture_notifications(app)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()

        await screen._stage_character_avatar_from_path(str(bad))
        await pilot.pause()

        editor = screen.query_one(PersonasCharacterEditorWidget)
        assert "image" not in editor.get_character_data()
        assert screen.state.has_unsaved_changes is False
        assert any("Unsupported avatar image type" in msg for msg, _ in notifications)
```

Add a stale-mode regression:

```python
async def test_stage_character_avatar_requires_open_editor(
    self, mock_app_instance, stub_characters, tmp_path
):
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"\x89PNG staged avatar")
    app = PersonasTestApp(mock_app_instance)
    notifications = TestImportExport._capture_notifications(app)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()

        await screen._stage_character_avatar_from_path(str(avatar))
        await pilot.pause()

        assert screen.state.has_unsaved_changes is False
        assert any("Open a character editor" in msg for msg, _ in notifications)
```

Expected: tests fail because `_stage_character_avatar_from_path` does not exist.

- [ ] **Step 2: Run the targeted failing tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q -k "avatar"
```

Expected: failure references missing avatar staging helper.

- [ ] **Step 3: Import the new message and path validation**

In `tldw_chatbook/UI/Screens/personas_screen.py`, import:

```python
from ...Utils.path_validation import validate_path_simple
```

and add `CharacterImageUploadRequested` to the existing pane-message imports.

- [ ] **Step 4: Add supported suffix constants**

Near other screen-level constants, add:

```python
PERSONAS_AVATAR_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
PERSONAS_AVATAR_IMAGE_SUFFIX_COPY = "PNG, JPG, JPEG, WEBP, or GIF"
```

- [ ] **Step 5: Add the read/validate helper**

Add a synchronous helper so tests and the async wrapper have one validation source:

```python
def _read_avatar_image_bytes(self, path: str) -> bytes:
    candidate = validate_path_simple(path, require_exists=True)
    if not candidate.is_file():
        raise ValueError("Choose an existing avatar image file.")
    if candidate.suffix.lower() not in PERSONAS_AVATAR_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported avatar image type. Use {PERSONAS_AVATAR_IMAGE_SUFFIX_COPY}."
        )
    data = candidate.read_bytes()
    if not data:
        raise ValueError("Avatar image file is empty.")
    return data
```

- [ ] **Step 6: Add the async path-based staging helper**

Add:

```python
async def _stage_character_avatar_from_path(self, path: str) -> None:
    if self._edit_mode not in ("create", "edit"):
        self._notify("Open a character editor before uploading an avatar.", "warning")
        return
    try:
        image_data = await asyncio.to_thread(self._read_avatar_image_bytes, path)
    except ValueError as exc:
        self._notify(str(exc), "error")
        return
    except OSError as exc:
        logger.error(f"Error reading avatar image from {path}: {exc}", exc_info=True)
        self._notify(f"Avatar upload failed: {exc}", "error")
        return
    try:
        self.query_one(PersonasCharacterEditorWidget).set_avatar_image(image_data)
    except Exception as exc:
        logger.error("Could not stage avatar image in editor.", exc_info=True)
        self._notify(f"Avatar upload failed: {exc}", "error")
        return
    self._notify("Avatar staged. Save the character to persist it.", "information")
```

- [ ] **Step 7: Run the path-staging tests until green**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q -k "avatar"
```

Expected: the new path helper tests pass.

- [ ] **Step 8: Add the dialog worker and message handler**

Add:

```python
@on(CharacterImageUploadRequested)
def _handle_character_image_upload_requested(
    self, message: CharacterImageUploadRequested
) -> None:
    message.stop()
    if self._edit_mode not in ("create", "edit"):
        self._notify("Open a character editor before uploading an avatar.", "warning")
        return
    if self._io_dialog_active:
        logger.debug("Import/export dialog already active; ignoring avatar upload request.")
        return
    self._io_dialog_active = True
    self.run_worker(self._avatar_upload_dialog_worker(), group="personas-io")
```

Add:

```python
async def _avatar_upload_dialog_worker(self) -> None:
    from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

    try:
        picker = EnhancedFileOpen(
            title="Upload Character Avatar",
            filters=Filters(
                (
                    "Image Files",
                    lambda p: p.suffix.lower() in PERSONAS_AVATAR_IMAGE_SUFFIXES,
                ),
                ("PNG Files", lambda p: p.suffix.lower() == ".png"),
                ("JPEG Files", lambda p: p.suffix.lower() in (".jpg", ".jpeg")),
                ("WEBP Files", lambda p: p.suffix.lower() == ".webp"),
                ("GIF Files", lambda p: p.suffix.lower() == ".gif"),
            ),
            context="character_avatar_upload",
        )
        try:
            file_path = await self.app.push_screen_wait(picker)
        except Exception:
            logger.warning("Could not show the avatar upload file dialog.", exc_info=True)
            return
        if file_path:
            await self._stage_character_avatar_from_path(str(file_path))
    finally:
        self._io_dialog_active = False
```

- [ ] **Step 9: Add a message-handler test**

Add a test that posts `CharacterImageUploadRequested` while in create mode, monkeypatches `_avatar_upload_dialog_worker`, and asserts exactly one worker is launched unless `_io_dialog_active` is already true.

Minimal shape:

```python
async def test_avatar_upload_request_launches_dialog_worker(
    self, mock_app_instance, stub_characters
):
    calls: list[int] = []
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()

        def worker():
            calls.append(1)

            async def _noop():
                pass

            return _noop()

        screen._avatar_upload_dialog_worker = worker
        screen.post_message(CharacterImageUploadRequested())
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert calls == [1]
```

- [ ] **Step 10: Run workbench avatar tests until green**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q -k "avatar"
```

Expected: all avatar-related workbench tests pass.

- [ ] **Step 11: Commit Task 2**

Run:

```bash
git add Tests/UI/test_personas_workbench.py tldw_chatbook/UI/Screens/personas_screen.py
git commit -m "feat: wire personas avatar upload flow"
```

Expected: commit succeeds.

---

### Task 3: Prove Save Persists Staged Avatar Bytes

**Files:**
- Modify: `Tests/UI/test_personas_workbench.py`

- [ ] **Step 1: Add a Save-path regression**

Add a mounted create-flow test:

```python
async def test_save_persists_staged_avatar_bytes(
    self, mock_app_instance, stub_characters, monkeypatch, tmp_path
):
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"\x89PNG staged avatar")
    created: list[dict[str, Any]] = []
    monkeypatch.setattr(
        character_handler_module,
        "create_character",
        lambda data: created.append(dict(data)) or 99,
    )
    app = PersonasTestApp(mock_app_instance)

    async with app.run_test() as pilot:
        screen = await _mounted(pilot)
        await pilot.pause()
        await pilot.click("#personas-library-new")
        await pilot.pause()
        screen.query_one("#personas-char-editor-name", Input).value = "Avatar Hero"
        await screen._stage_character_avatar_from_path(str(avatar))
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert created
    assert created[0]["name"] == "Avatar Hero"
    assert created[0]["image"] == b"\x89PNG staged avatar"
```

Expected: if Task 2 is incomplete, this fails because no image is staged or persisted.

- [ ] **Step 2: Run the Save-path regression**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q -k "avatar or ctrl_s_saves_from_editor"
```

Expected: avatar tests and the nearby Save keyboard regression pass.

- [ ] **Step 3: Commit Task 3**

Run:

```bash
git add Tests/UI/test_personas_workbench.py
git commit -m "test: cover staged avatar save"
```

Expected: commit succeeds.

---

### Task 4: Final Verification and Backlog Completion

**Files:**
- Modify: `backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md`

- [ ] **Step 1: Run focused UI suites**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_character_widgets.py Tests/UI/test_personas_workbench.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run diff whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect the implementation diff**

Run:

```bash
git diff origin/dev...HEAD --stat
git diff origin/dev...HEAD -- tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/UI/Screens/personas_screen.py
```

Expected: diff is scoped to the planned files, no unrelated refactors, and avatar image data is stored as bytes.

- [ ] **Step 4: Update Backlog acceptance criteria and implementation notes**

Edit `backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md` so both ACs are checked and add:

```markdown
## Implementation Notes

Restored character avatar upload in the ds-native Personas editor using the existing staged edit workflow. The editor now emits `CharacterImageUploadRequested`, stages raw image bytes in pending character data, updates avatar status, and marks the session dirty. `PersonasScreen` opens an image-filtered picker, validates selected paths with `validate_path_simple`, reads bytes off the UI thread, and persists the staged image through the existing Save flow.

ADR required: no
ADR path: N/A
Reason: scoped UI workflow restoration using existing editor, file picker, dirty-state, and character persistence boundaries; no schema, sync, storage, provider/runtime, or application-architecture change.
```

- [ ] **Step 5: Mark the task Done via Backlog CLI**

Run:

```bash
backlog task edit 100 -s Done
```

Expected: `TASK-100` status is `Done`.

- [ ] **Step 6: Commit task completion**

Run:

```bash
git add "backlog/tasks/task-100 - Wire-avatar-upload-in-the-ds-native-character-editor.md"
git commit -m "docs: complete avatar upload task"
```

Expected: commit succeeds.

- [ ] **Step 7: Final branch check**

Run:

```bash
git status --short --branch
```

Expected: clean branch, ahead of `origin/dev` by the planning and implementation commits.
