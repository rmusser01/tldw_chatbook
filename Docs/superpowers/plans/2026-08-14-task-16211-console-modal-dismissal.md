# TASK-16211 Safe Console Modal Dismissal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every modal transitively reachable from Console safely cancellable with Escape or a primary backdrop click without losing work, duplicating callbacks, popping the wrong screen, or breaking nested controls.

**Architecture:** Add one framework-neutral `SafeModalDismissMixin` under `Widgets` that owns hit classification, one-shot async cancellation, top-screen verification, and focus restoration. Each modal remains responsible for its typed cancel result and state-specific guard; an explicit test-only contract table inventories the transitive Console launch graph and records every content boundary, safe hook, guard, and focus destination.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Textual Pilot, Ruff, MyPy.

**ADR required:** yes; amend existing ADR

**ADR path:** `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`

**Reason:** The shared mixin and transitive modal grammar are a cross-module interface and long-lived UX rule. ADR-031 already owns TUI keybinding and truthful-hint conventions and has been amended by the approved design commit.

---

## File structure

**New shared boundary**

- Create `tldw_chatbook/Widgets/modal_dismissal.py`: pure backdrop classifier plus `SafeModalDismissMixin`; no Console imports.
- Create `Tests/UI/test_console_modal_dismissal.py`: shared behavior, transitive inventory, MRO, focus, and click-through contracts.

**Exceptional state machines**

- Modify `tldw_chatbook/Widgets/Console/console_prompts_modal.py`: Workbench Close/dirty/improvement transition contract.
- Modify `tldw_chatbook/Widgets/Console/console_settings_modal.py`: reset-token and compaction close guards.
- Modify `tldw_chatbook/Widgets/Console/console_video_capacity_modal.py`: non-destructive generic dismissal guard.

**Shared Console-reachable components**

- Modify `tldw_chatbook/UI/Workbench/help.py`.
- Modify `tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py`.
- Modify `tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py`.
- Modify `tldw_chatbook/Widgets/confirmation_dialog.py`.
- Modify `tldw_chatbook/Widgets/cancel_confirmation_dialog.py`.
- Modify `tldw_chatbook/Widgets/enhanced_file_picker.py`.
- Modify `tldw_chatbook/UI/Screens/video_player_screen.py`.
- Modify `tldw_chatbook/UI/Screens/change_review_screen.py`.

**Mechanical Console adoption**

- Modify the 24 non-exceptional `ModalScreen` classes under `tldw_chatbook/Widgets/Console/` listed in Tasks 2 and 3. Each change is limited to mixin inheritance, a stable content selector, Escape routing, and reuse of the existing visible Cancel path.

**Focused existing tests**

- Extend only the modal-specific files named under each task. Do not run broad test directories or the full suite.

---

### Task 1: Build the one-shot modal dismissal primitive

**Files:**

- Create: `tldw_chatbook/Widgets/modal_dismissal.py`
- Create: `Tests/UI/test_console_modal_dismissal.py`

- [x] **Step 1: Write pure classifier RED tests**

Add tests for primary outside, primary inside by DOM ancestry, primary inside by geometry, non-primary outside, and unknown provenance. Use a small fake content widget/region rather than constructing impossible coordinate-less Textual `Click` events.

```python
@pytest.mark.parametrize(
    ("button", "known", "descendant", "contains", "expected"),
    [
        (1, True, False, False, True),
        (1, True, True, False, False),
        (1, True, False, True, False),
        (2, True, False, False, False),
        (1, False, False, False, False),
    ],
)
def test_classify_modal_backdrop(...):
    assert is_modal_backdrop_click(...) is expected
```

- [x] **Step 2: Run the classifier tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_modal_dismissal.py -k classifier
```

Expected: collection/import failure because `modal_dismissal` does not exist.

- [x] **Step 3: Implement the pure classifier**

Implement a small function with no Textual screen-stack behavior:

```python
def is_modal_backdrop_click(
    *,
    button: int,
    provenance_known: bool,
    target_is_content_or_descendant: bool,
    point_is_in_content_region: bool,
) -> bool:
    return (
        button == 1
        and provenance_known
        and not target_is_content_or_descendant
        and not point_is_in_content_region
    )
```

- [x] **Step 4: Write mounted RED tests for single-shot cancellation**

Create a tiny `ModalScreen[bool | None]` using the mixin and an async cancellation gate. Assert:

- two Escape requests while the gate is pending invoke the callback once;
- a backdrop request while Escape is pending is consumed;
- if the callback pushes another modal, the stale outer request does not pop it;
- after that nested screen closes, retrying the outer request dismisses it
  without invoking the already-committed callback again;
- visible Cancel and backdrop return the same typed value;
- the opener's previously focused widget is restored;
- if the opener widget is removed, a screen exposing `_focus_console_composer_if_needed(force=True)` receives the fallback call.

- [x] **Step 5: Run the mounted tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_modal_dismissal.py -k 'single_shot or top_screen or restore_focus'
```

Expected: failures because `SafeModalDismissMixin` is absent.

- [x] **Step 6: Implement `SafeModalDismissMixin` minimally**

The implementation must expose these override seams and no registry/framework magic:

```python
class SafeModalDismissMixin:
    SAFE_MODAL_CONTENT: str | None = None

    async def action_request_safe_cancel(self) -> None:
        await self.request_safe_cancel(source="escape")

    async def request_safe_cancel(self, *, source: str) -> None:
        if self._safe_cancel_pending:
            return
        self._safe_cancel_pending = True
        try:
            await self._perform_safe_cancel(source=source)
        finally:
            if self.is_mounted:
                self._safe_cancel_pending = False

    async def _perform_safe_cancel(self, *, source: str) -> None:
        self.dismiss_safe_once(None)

    async def run_cancel_effect_once(self, effect: Callable[[], Awaitable[None]]) -> None:
        if self._safe_cancel_effect_committed:
            return
        self._safe_cancel_effect_committed = True
        await effect()

    def dismiss_safe_once(self, result: object) -> bool:
        if self._safe_dismiss_committed:
            return False
        if not self.is_mounted or self.app.screen is not self:
            return False
        self._safe_dismiss_committed = True
        # Capture app/opener before dismiss, then restore after refresh.
        self.dismiss(result)
        return True
```

`_safe_cancel_pending`, `_safe_cancel_effect_committed`, and
`_safe_dismiss_committed` are distinct states. The pending latch may reset after
an attempt; the effect commitment never resets, even when the callback raises
or a nested modal prevents terminal dismissal. A later request may retry only
`dismiss_safe_once` after the nested screen closes. `on_click` must derive
ancestry and `Widget.region.contains(screen_x, screen_y)`, call the pure
classifier, then `stop()` and `prevent_default()` before awaiting cancellation.
Record opener focus on mount without retaining it beyond modal lifetime. Restore
it after refresh if still mounted; otherwise call the optional Console composer
focus method on the revealed screen.

- [x] **Step 7: Run Task 1 GREEN and static checks**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_modal_dismissal.py
../../.venv/bin/ruff check tldw_chatbook/Widgets/modal_dismissal.py Tests/UI/test_console_modal_dismissal.py
../../.venv/bin/ruff format --check tldw_chatbook/Widgets/modal_dismissal.py Tests/UI/test_console_modal_dismissal.py
```

Expected: all pass.

- [x] **Step 8: Commit the primitive**

```bash
git add tldw_chatbook/Widgets/modal_dismissal.py Tests/UI/test_console_modal_dismissal.py
git diff --cached --check
git commit -m "feat(ui): add safe modal dismissal boundary"
```

---

### Task 2: Adopt low-state Console-owned modals

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_character_picker_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_citation_sources_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_context_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_cost_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_image_viewer_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_model_popover.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompt_picker_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_prompt_queue_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_run_log_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_scope_picker_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_skill_picker_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_style_picker_modal.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`
- Test: `Tests/UI/test_console_image_viewer.py`
- Test: `Tests/UI/test_console_context_modal.py`
- Test: `Tests/UI/test_console_cost_modal.py`
- Test: `Tests/UI/test_console_prompt_queue_modal.py`
- Test: `Tests/UI/test_console_scope_picker_modal.py`
- Test: `Tests/UI/test_console_skill_picker.py`
- Test: `Tests/Chat/test_console_style_picker.py`

- [x] **Step 1: Add a RED contract table for these 12 types**

Each row contains `modal_type`, `content_selector`, `cancel_result`, `opener`,
`pre_cancel_hook`, `guard`, and `focus_postcondition`. Assert mixin inheritance,
stable selector presence, and Escape binding to `request_safe_cancel`. The four
existing cleanup contracts are explicit, not treated as default `dismiss(None)`:

```text
ConsoleCharacterPickerModal -> _cancel_query_debounce
ConsoleCitationSourcesModal -> increment _request_generation
ConsoleStylePickerModal -> _cancel_search_debounce
all other Task 2 rows -> no explicit pre-cancel hook
```

- [x] **Step 2: Add representative mounted RED tests**

Mount at least one `None` result, one typed picker, and Image Viewer. Assert primary backdrop matches visible Cancel, inside clicks stay open, and Image Viewer invokes dismissal once despite its click-anywhere contract.

For Character, Citation Sources, and Style Picker, arm the real timer/request
generation state before each of visible Cancel, Escape, and backdrop. Assert
all three paths perform the exact cleanup once before dismissal.

- [x] **Step 3: Run Task 2 tests RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_console_image_viewer.py \
  Tests/UI/test_console_context_modal.py \
  Tests/UI/test_console_cost_modal.py \
  Tests/UI/test_console_prompt_queue_modal.py \
  Tests/UI/test_console_scope_picker_modal.py \
  Tests/UI/test_console_skill_picker.py \
  Tests/Chat/test_console_style_picker.py
```

Expected: new mixin/contract assertions fail.

- [x] **Step 4: Apply the mechanical adoption**

For each class:

1. inherit `SafeModalDismissMixin` before `ModalScreen`;
2. set `SAFE_MODAL_CONTENT` to its existing outer ID;
3. route Escape to `request_safe_cancel`;
4. route the visible Cancel/Close handler through the same safe request;
5. override `_perform_safe_cancel` where the contract table names a pre-cancel
   hook, invoke that hook once, then call `dismiss_safe_once`;
6. preserve non-cancel results unchanged.

Selectors are the existing outer IDs: `#console-character-picker`, `#console-citation-sources-modal`, `#console-context-modal`, `#console-cost-modal`, `#console-image-viewer`, `#console-model-popover`, `#console-prompt-picker-modal`, `#console-prompt-queue-dialog`, `#console-run-log-modal`, `#console-scope-picker-modal`, `#console-skill-picker-modal`, and `#console-style-picker-modal`.

Remove or delegate Image Viewer's old screen-level click dismissal so full MRO dispatch cannot call dismissal twice. Preserve its click-anywhere result `None` by overriding the backdrop classifier/action intentionally.

Do not replace Character, Citation Sources, or Style Picker cancellation with
the mixin default: their debounce/request invalidation is part of the existing
safe Cancel contract.

- [x] **Step 5: Run Task 2 GREEN**

Run the Step 3 command again. Expected: all pass.

- [x] **Step 6: Commit Task 2**

```bash
git add tldw_chatbook/Widgets/Console Tests/UI/test_console_modal_dismissal.py Tests/UI/test_console_image_viewer.py
git diff --cached --check
git commit -m "feat(console): dismiss ordinary modal surfaces safely"
```

---

### Task 3: Adopt form/action Console-owned modals and reconcile MRO handlers

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_composer_menu_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_generate_image_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rag_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rename_session_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rewind_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_save_as_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_system_prompt_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_switcher_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/prompt_variables_dialog.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`
- Test: `Tests/UI/test_console_composer_menu.py`
- Test: `Tests/UI/test_console_rag_settings_modal.py`
- Test: `Tests/Chat/test_console_edit_message_modal.py`
- Test: `Tests/Chat/test_console_rewind_modal.py`
- Test: `Tests/Chat/test_console_session_settings.py`
- Test: `Tests/UI/test_console_system_prompt.py`
- Test: `Tests/UI/test_prompt_variables_dialog.py`

- [x] **Step 1: Extend the RED contract table with these 11 files / 12 classes**

Include both `ConsoleWorkspaceSwitcherModal` and
`ConsoleWorkspaceRenameModal`. Use their exact existing outer IDs. Record
`ConsoleSessionSwitcherModal._cancel_query_debounce` as its pre-cancel hook;
all other Task 3 rows have no explicit cleanup hook. Assert typed success
results remain outside the generic cancel path.

- [x] **Step 2: Add RED exact-once MRO tests**

For Composer Menu and RAG Settings, dispatch a real mounted `events.Click` through Textual rather than calling one handler directly. Count the result callback once. Add a Settings-adjacent regression fixture proving the mixin pattern does not suppress unrelated click handlers.

Arm Session Switcher's real debounce timer and prove visible Cancel, Escape,
and backdrop each stop it once before dismissing.

- [x] **Step 3: Run focused RED tests**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_rag_settings_modal.py \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/Chat/test_console_rewind_modal.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/UI/test_console_system_prompt.py \
  Tests/UI/test_prompt_variables_dialog.py
```

Expected: contract/backdrop tests fail before adoption.

- [x] **Step 4: Apply mechanical adoption and remove duplicate backdrop branches**

Use selectors: `#console-composer-menu`, `#console-edit-message-modal`, `#console-generate-image-modal`, `#console-rag-settings`, `#console-rename-session-modal`, `#console-rewind-modal`, `#console-save-as-modal`, `#console-switcher-modal`, `#console-system-prompt-modal`, `#console-workspace-switcher-modal`, `#console-workspace-rename-modal`, and `#prompt-variables-dialog`.

Composer Menu and RAG Settings must delegate/remove their existing backdrop blocks. Preserve every existing non-cancel button and result unchanged.

Override Session Switcher's `_perform_safe_cancel` to call the existing
`_cancel_query_debounce` helper before `dismiss_safe_once(None)`; do not route
it through the default hook.

- [x] **Step 5: Run Task 3 GREEN and commit**

Run the Step 3 command, then:

```bash
git add tldw_chatbook/Widgets/Console Tests/UI Tests/Chat/test_console_edit_message_modal.py Tests/Chat/test_console_rewind_modal.py
git diff --cached --check
git commit -m "feat(console): unify form modal cancellation"
```

---

### Task 4: Cover shared and transitively reachable modal components

**Files:**

- Modify: `tldw_chatbook/UI/Workbench/help.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py`
- Modify: `tldw_chatbook/Widgets/confirmation_dialog.py`
- Modify: `tldw_chatbook/Widgets/cancel_confirmation_dialog.py`
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py`
- Modify: `tldw_chatbook/UI/Screens/video_player_screen.py`
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`
- Test: `Tests/UI/test_dictionary_picker.py`
- Test: `Tests/UI/test_world_book_picker.py`
- Test: `Tests/UI/test_enhanced_file_dialog_mount.py`
- Test: `Tests/UI/test_change_review_screen.py`
- Test: `Tests/Media_Playback/test_player_screen.py`

- [x] **Step 1: Write RED shared-result and transitive inventory tests**

Assert:

- Dictionary/World Book/Workbench/File Open/Video Player cancel with `None`;
- `ConfirmationDialog`, `CancelConfirmationDialog`, and `ChangeRevertConfirmModal` cancel with exact `False`;
- `ConfirmationDialog` awaits its cancel callback once under repeated input;
- `ChangeRevertConfirmModal` is reached transitively through `ChangeReviewScreen`;
- `EnhancedFileOpen` and `EnhancedFileSave` Escape still peel path/search/recent/bookmarks in order; their terminal Escape, backdrop, and visible Cancel all pass through `dismiss_safe_once(None)` exactly once;
- clicks on each open file-picker sub-surface remain inside;
- Video Player's whole screen is content, Escape runs the same stop/cleanup path as `q`, and frame/status/hint clicks are not backdrop.

- [x] **Step 2: Run shared tests RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_dictionary_picker.py \
  Tests/UI/test_world_book_picker.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  Tests/UI/test_change_review_screen.py \
  Tests/Media_Playback/test_player_screen.py
```

Expected: missing Escape/backdrop behavior fails.

- [x] **Step 3: Add stable content boundaries and typed cancel overrides**

Add IDs to currently bare outer wrappers: `#dictionary-picker-dialog`, `#world-book-picker-dialog`, `#confirmation-dialog`, `#cancel-confirmation-dialog`, and `#enhanced-file-dialog`. Use `#workbench-help-panel` and `#change-revert-confirm` as-is. Mark Video Player as full-screen content so it has no synthetic backdrop.

`ConfirmationDialog._perform_safe_cancel` must pass its existing callback to
`run_cancel_effect_once` and then call `dismiss_safe_once(False)`. If a nested
screen made that dismissal stale, the later retry skips the callback and only
retries `dismiss_safe_once(False)`. The other boolean dialogs call
`dismiss_safe_once(False)` without inventing `None`.

For the shared `EnhancedFileDialog` base used by `EnhancedFileOpen` and
`EnhancedFileSave`, inherit the mixin without disturbing its
`_get_dispatch_methods` filtering. Keep `action_smart_dismiss` bound to Escape
and preserve its overlay-peeling branches unchanged. Replace only its terminal
branch with `dismiss_safe_once(None)`, and route backdrop plus visible Cancel
through the same safe terminal path. `dismiss_safe_once` still calls the
dialog's existing `dismiss` override, so recent-location persistence remains
intact while focus restoration, top-screen verification, and terminal
single-shot behavior are enforced.

- [x] **Step 4: Run shared tests GREEN and commit**

Run Step 2, then:

```bash
git add tldw_chatbook/UI/Workbench/help.py tldw_chatbook/Widgets/Persona_Widgets \
  tldw_chatbook/Widgets/confirmation_dialog.py tldw_chatbook/Widgets/cancel_confirmation_dialog.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py tldw_chatbook/UI/Screens/video_player_screen.py \
  tldw_chatbook/UI/Screens/change_review_screen.py Tests/UI Tests/Media_Playback/test_player_screen.py
git diff --cached --check
git commit -m "feat(console): cancel shared modal components safely"
```

---

### Task 5: Make Prompt Workbench dismissal a tested state machine

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_prompts_modal.py`
- Test: `Tests/UI/test_console_prompts_modal.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`

- [x] **Step 1: Write the transition-matrix RED tests**

Cover clean root, clean nested mode, dirty edit, dirty recipe, guard visible, active improvement, cancelling improvement, and expanded descendant control. For every row assert gesture, visible state, result, callback count, and focused widget.

Key contracts:

```text
clean nested + Escape/backdrop -> close whole Workbench
dirty + Escape/backdrop -> reveal dirty guard
guard visible + Escape -> exact Keep Editing path + editor focus
guard visible + backdrop -> remain guarded
active improvement -> existing cancellation behavior, once
```

- [x] **Step 2: Run Workbench tests RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_modal_dismissal.py -k prompts
```

Expected: current Escape performs Back and guard Escape lacks focus restoration.

- [x] **Step 3: Implement the minimal Workbench overrides**

Inherit the mixin, set `SAFE_MODAL_CONTENT = "#console-prompts-modal"`, route clean Escape/backdrop to the existing Close request, and route dirty requests to the existing guard. Refactor guard Escape to call the same helper as visible Keep Editing so focus restoration is not duplicated.

Do not change the visible Back button or improvement transaction semantics.

- [x] **Step 4: Run Workbench tests GREEN and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_modal_dismissal.py -k prompts
git add tldw_chatbook/Widgets/Console/console_prompts_modal.py Tests/UI/test_console_prompts_modal.py Tests/UI/test_console_modal_dismissal.py
git diff --cached --check
git commit -m "fix(console): guard Prompt Workbench dismissal"
```

---

### Task 6: Protect Console Settings reset and compaction state

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Test: `Tests/Chat/test_console_session_settings.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`

- [x] **Step 1: Write reset-token guard RED tests**

After `_reset_current_branch_memory`, request close from visible Cancel, Escape, and backdrop. Assert the same three-choice guard appears:

- Undo and close invokes the existing optimistic undo and dismisses only on success;
- Keep reset and close clears/accepts the local undo opportunity and dismisses;
- Return hides the guard and restores focus to Undo;
- expired Undo leaves the guard open with the existing recovery message;
- repeated close gestures never stack guards.

- [x] **Step 2: Write active-compaction RED tests**

Gate `_compact_now` with an `asyncio.Event`, request close, and assert the two-choice acknowledgement. Return preserves progress/focus. Close anyway dismisses the modal, reports that work may continue/be billed, and never reports provider cancellation. Reopening Settings obtains fresh controller state rather than the abandoned widget state.

- [x] **Step 3: Add MRO regression RED tests**

Keep the existing redirected Textual-Web `Select` click cases green while adding backdrop handling. Dispatch the real event through the mounted modal, not one method directly.

- [x] **Step 4: Run Settings tests RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_session_settings.py Tests/UI/test_console_modal_dismissal.py -k 'settings or memory_reset or compaction or redirected'
```

Expected: current direct `dismiss(None)` strands the token and closes during work.

- [x] **Step 5: Implement one in-modal close guard**

Inherit the mixin and set `SAFE_MODAL_CONTENT = "#console-settings-modal"`. Add a hidden overlay region inside the modal rather than a new `ModalScreen`, with mode `reset` or `compaction` and these action IDs:

```text
#console-settings-close-undo
#console-settings-close-keep
#console-settings-close-anyway
#console-settings-close-return
```

Route visible Cancel, Escape, and backdrop through `_request_settings_close`. Keep Save and Save as default behavior unchanged. Store the focused widget before showing the guard and restore it on Return. Close anyway cancels only the modal-owned wait/worker and emits truthful status/notification copy; it must not claim the provider request was cancelled.

- [x] **Step 6: Run Settings tests GREEN and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_session_settings.py Tests/UI/test_console_modal_dismissal.py -k 'settings or memory_reset or compaction or redirected'
git add tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_console_session_settings.py Tests/UI/test_console_modal_dismissal.py
git diff --cached --check
git commit -m "fix(console): guard Settings close side effects"
```

---

### Task 7: Preserve staged generated video across generic dismissal

**Files:**

- Modify: `tldw_chatbook/Widgets/Console/console_video_capacity_modal.py`
- Test: `Tests/Chat/test_console_video_capacity.py`
- Test: `Tests/UI/test_console_modal_dismissal.py`

- [x] **Step 1: Replace the unsafe Escape oracle with RED guard tests**

Remove the old expectation that Escape returns `"discard"`. Mount the production resolver with a real `PendingVideoArtifact`; Escape and backdrop must push one `CancelConfirmationDialog` while the stream remains open and owned.

Assert:

- Continue returns to capacity choices and restores the safest reason-specific button;
- explicit confirmation alone returns `"discard"` and closes the artifact once through the existing resolver;
- repeated Escape/backdrop does not stack guards;
- Keep/Retry and Save to disk are unchanged;
- navigation/unmount still closes the artifact once.

- [x] **Step 2: Run capacity tests RED**

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_video_capacity.py Tests/UI/test_console_modal_dismissal.py -k 'capacity or staged_artifact'
```

Expected: Escape immediately discards before the fix.

- [x] **Step 3: Implement guarded capacity cancellation**

Inherit the mixin, set `SAFE_MODAL_CONTENT = "#video-capacity-dialog"`, and replace `action_discard` as the Escape target. `_perform_safe_cancel` pushes one `CancelConfirmationDialog` with explicit generated-video copy. Its `False` result refocuses `#video-capacity-save` for over-capacity and `#video-capacity-keep` for store failure; `True` calls `dismiss_safe_once("discard")`.

Do not map `None`, backdrop, or guard cancellation to any terminal `CapacityAction`.

- [x] **Step 4: Run capacity tests GREEN and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/Chat/test_console_video_capacity.py Tests/UI/test_console_modal_dismissal.py -k 'capacity or staged_artifact'
git add tldw_chatbook/Widgets/Console/console_video_capacity_modal.py Tests/Chat/test_console_video_capacity.py Tests/UI/test_console_modal_dismissal.py
git diff --cached --check
git commit -m "fix(console): confirm generated video discard"
```

---

### Task 8: Close the transitive inventory and focused verification

**Files:**

- Modify: `Tests/UI/test_console_modal_dismissal.py`
- Modify: `Docs/superpowers/plans/2026-08-14-task-16211-console-modal-dismissal.md`
- Modify: `backlog/tasks/task-16211 - Make-all-Console-modals-dismiss-safely-with-Escape-or-backdrop-click.md`
- Modify if incident-backed: `backlog/docs/lessons-testing-evidence.md`

- [x] **Step 1: Make the inventory assertion exhaustive**

Use Python runtime/AST inspection, not an anchored regex, to discover every `ModalScreen` type under `Widgets/Console`. Compare them with the explicit contract rows. Walk the explicit direct and nested launch edges to a fixed point and require both enhanced file dialog variants, `CancelConfirmationDialog`, and `ChangeRevertConfirmModal`. Assert `ConsoleSetupModal` is excluded because it is an embedded `Vertical`, not a screen.

- [x] **Step 2: Add final click-through and focus integration tests**

Use explicitly dispatched button-2/3 events for non-primary coverage. Exercise rapid primary clicks through normal Textual dispatch, count callbacks, and prove the revealed screen receives no action. Verify prior focus restoration and Console composer fallback. Keep full-screen Video Player cells inside.

- [x] **Step 3: Run only the related-file verification matrix**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_console_image_viewer.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_rag_settings_modal.py \
  Tests/UI/test_console_prompts_modal.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/UI/test_dictionary_picker.py \
  Tests/UI/test_world_book_picker.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  Tests/UI/test_change_review_screen.py \
  Tests/Media_Playback/test_player_screen.py \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/Chat/test_console_rewind_modal.py \
  Tests/Chat/test_console_video_capacity.py \
  Tests/UI/test_prompt_variables_dialog.py
```

Expected: all pass. Do not run full test directories or the full repository suite.

- [x] **Step 4: Run targeted static verification**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py \
  tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py \
  tldw_chatbook/Widgets/confirmation_dialog.py \
  tldw_chatbook/Widgets/cancel_confirmation_dialog.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Workbench/help.py \
  tldw_chatbook/UI/Screens/video_player_screen.py \
  tldw_chatbook/UI/Screens/change_review_screen.py \
  Tests/UI/test_console_modal_dismissal.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py \
  tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py \
  tldw_chatbook/Widgets/confirmation_dialog.py \
  tldw_chatbook/Widgets/cancel_confirmation_dialog.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Workbench/help.py \
  tldw_chatbook/UI/Screens/video_player_screen.py \
  tldw_chatbook/UI/Screens/change_review_screen.py \
  Tests/UI/test_console_modal_dismissal.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Persona_Widgets/dictionary_picker.py \
  tldw_chatbook/Widgets/Persona_Widgets/world_book_picker.py \
  tldw_chatbook/Widgets/confirmation_dialog.py \
  tldw_chatbook/Widgets/cancel_confirmation_dialog.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Workbench/help.py \
  tldw_chatbook/UI/Screens/video_player_screen.py \
  tldw_chatbook/UI/Screens/change_review_screen.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/UI/Workbench/help.py \
  tldw_chatbook/UI/Screens/video_player_screen.py
git diff --check
```

The listed legacy scope may already be formatter/MyPy-red. Capture the exact
pre-task result for any failing command, compare it with the final result, and
fix every new or changed-line diagnostic. New files and previously clean files
must pass outright; do not reformat unrelated legacy code.

- [x] **Step 5: Perform the required mutation proofs**

Temporarily and separately:

1. remove the backdrop outside branch;
2. remove the cancellation-pending latch;
3. remove the active-top-screen check;
4. change `False` confirmation cancellation to `None`.

Run each named discriminating test and require RED. Restore immediately and rerun GREEN.

- [x] **Step 6: Self-review against all nine acceptance criteria**

Audit the staged diff for accidental result changes, missing launch edges, MRO double handlers, unsafe destructive defaults, focus gaps, and unrelated formatting churn. Update the task plan only if implementation deviated.

- [x] **Step 7: Complete Backlog hygiene**

Check all acceptance criteria, add concise Implementation Notes with the exact related-file evidence and ADR-031 amendment, and set TASK-16211 to Done through the Backlog CLI. Add a lessons entry only if implementation produces incident-backed knowledge not already captured by the design/ADR.

**Task 8 closeout evidence (2026-08-14):**

- Runtime plus AST discovery found exactly 28 `Widgets/Console` `ModalScreen`
  classes on the rebased `dev`, including `AutoSpeakConsentModal`, and walked
  the direct/nested launch graph to exactly 37 contracted
  modal types. Final review hardened this to resolve the actual constructed
  runtime classes from AST imports, aliases, attributes, and same-module
  definitions, with exact equality per fixed-point edge. A synthetic extra
  constructed modal proved the assertion rejects uncontracted launch edges.
- Prompt Workbench apply now claims a mount-scoped transaction before its
  callback worker starts. Escape, backdrop, Close, Back, and duplicate Apply
  remain inert until the callback completes, while the applying status owns
  focus. Applied results use `dismiss_safe_once`; stale unmounts and nested top
  screens cannot be popped, and a blocked nested dismissal reports the
  committed state truthfully. Explicit lifecycle `super()` calls were removed,
  with mount/unmount/remount dispatch proven once per lifecycle.
- The final exact 15-file matrix passed 626 tests. The final correction-focused
  set passed 14 tests after the nested-top RED/GREEN proof; the complete two-file
  Prompt Workbench/modal dismissal sweep passed 212 before that final one-line
  status refinement.
- Ruff check, Ruff format, compileall, and `git diff --check` passed for all
  changed code/test files. Targeted MyPy retained only two pre-existing
  diagnostics on untouched 2026-08-02 callback sites; no changed-line
  diagnostic remains.
- The outside-backdrop branch, pending latch, top-screen check, and
  `False`-confirmation result each produced the named RED failure separately,
  then passed GREEN after immediate restoration.
- All nine acceptance criteria were reviewed against the inventory, typed
  result/callback, guard, MRO, focus, click-through, and video-player evidence.
  No new lessons entry was added because the observed Textual MRO behavior is
  already covered in `backlog/docs/lessons-testing-evidence.md`.
- Cumulative final review added cancellation-resistant same-instance remount
  proofs around async review validation. Both validation success and failure
  went RED: the former invoked the stale commit callback, while the latter
  cleared the new mount's latch. Mount generation is now checked before
  validation-side UI/latch mutation and again before apply coordination, whose
  own early generation gate protects every claimed apply worker.
- The launch fixed point now scans the defining class body of every reachable
  screen, treating owners without a declared edge as an explicit zero-launch
  contract. A synthetic reachable rowless owner constructing an aliased nested
  modal proved the inventory fails instead of silently skipping it.
- Every reachable screen's class-scoped defining-body scan is now unioned with
  its explicit helper/controller source scans before exact comparison. A
  synthetic owner with a valid declared helper edge and an undeclared aliased
  modal in its own class body produced RED before the union and GREEN after it;
  set union prevents duplicate constructors from changing the result.
- Rebase onto `origin/dev` added `AutoSpeakConsentModal`. The inventory caught
  the missing contract at RED (637 passed, one inventory failure). After the
  modal adopted the shared bounded-content contract, visible Cancel, Escape,
  and backdrop returned exact `False`; the final 16-file related matrix passed
  684 tests. Ruff lint, targeted MyPy, compileall, and diff checks passed. The
  upstream auto-speak module's pre-existing whole-file Ruff-format debt was
  left unchanged outside the new conforming hunks.
- Cumulative-review verification stayed bounded to the owned high-risk files:
  the final correction set passed 18 tests and the complete two-file matrix of
  217 tests
  passed. Ruff check/format, compileall, and `git diff --check` passed; targeted
  MyPy retained only the same two untouched baseline diagnostics.

- [x] **Step 8: Commit closeout**

```bash
git add Docs/superpowers/plans/2026-08-14-task-16211-console-modal-dismissal.md \
  "backlog/tasks/task-16211 - Make-all-Console-modals-dismiss-safely-with-Escape-or-backdrop-click.md" \
  backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md \
  backlog/docs/lessons-testing-evidence.md
git diff --cached --check
git commit -m "docs(console): complete safe modal dismissal task"
```

If no lessons file changed, omit it from `git add`.
