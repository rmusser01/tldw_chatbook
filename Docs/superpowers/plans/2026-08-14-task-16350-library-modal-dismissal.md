# TASK-16350 Safe Library Modal Dismissal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every modal transitively reachable from Library safely cancellable with Escape or a primary backdrop click without confirming destructive work, bypassing active mutations, popping the wrong screen, or losing focus across Library recomposition.

**Architecture:** Extend the existing `SafeModalDismissMixin` only where Library exposes a proven gap. Shared file pickers gain one intrinsic source-aware cancellation contract; each concrete Library modal keeps ownership of its typed negative result and mutation guard. A deliberately narrow, bidirectional test inventory names the supported direct, controller-injected, nested-widget, and modal-to-modal presenter edges and mounts every concrete reachable modal.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Textual Pilot, Ruff, MyPy.

**ADR required:** yes; amend existing ADR

**ADR path:** `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`

**Reason:** Library adoption changes the long-lived cross-module modal cancellation grammar, shared file-picker behavior, and stable focus restoration boundary. ADR-031 already owns this interaction contract and was amended with the approved design.

---

## Constraints and evidence rules

- Run only the named touched/related test files below. Do not run the full repository or broad test directories.
- Every behavior change starts with a focused failing test and records the observed RED cause before production edits.
- Every concrete Library-reachable modal gets its own mounted contract row; a sibling with the same result type is not a substitute.
- Real Textual dispatch is required for Escape, backdrop, visible controls, `SelectOverlay`, and `DirectoryNavigation.Changed` behavior.
- Keep the fixed-point inventory explicit and narrow. Do not create a general Python call-graph analyzer or runtime registry.
- Preserve user changes and stage only files owned by the current task.
- Before implementation, read `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-backlog-hygiene.md`, and the Impeccable craft floor referenced by the approved design workflow.

## File structure

**Shared boundary and picker base**

- Modify `tldw_chatbook/Widgets/modal_dismissal.py`: record a stable opener ID and restore only one eligible exact-ID replacement.
- Modify `tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py`: adopt the safe mixin, add a stable content boundary, route terminal cancellation through the shared one-shot path, and remove duplicate directory-change error clearing.
- Modify `tldw_chatbook/Third_Party/textual_fspicker/select_directory.py`: remove manual full-MRO lifecycle and decorated-parent dispatch duplication.
- Modify `tldw_chatbook/Widgets/enhanced_file_picker.py`: remove the now-redundant explicit mixin base while preserving enhanced smart-dismiss and handler suppression.

**Library modal owners**

- Modify `tldw_chatbook/UI/Screens/skills_screen.py`.
- Modify `tldw_chatbook/Widgets/ModelArtifacts/install_modal.py`.
- Modify `tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py`.
- Modify `tldw_chatbook/Widgets/Library/library_note_folder_dialog.py`.
- Modify `tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py`.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`.

**Focused tests**

- Create `Tests/UI/test_library_modal_dismissal.py`: concrete modal contract table, focus restoration, click ownership, lifecycle, and exact launch inventory.
- Modify `Tests/UI/test_console_modal_dismissal.py`: shared focus helper compatibility, including the existing Console-only fallback.
- Modify `Tests/UI/test_fspicker_keyboard_save.py`: base picker Escape/Cancel/backdrop precedence and typed results.
- Modify `Tests/UI/test_enhanced_file_dialog_mount.py`: enhanced MRO/import/smart-dismiss/persistence compatibility.
- Modify `Tests/UI/test_library_skills_canvas.py`.
- Modify `Tests/UI/test_model_artifact_widgets.py`.
- Modify `Tests/UI/test_prompt_delete_confirmation_modal.py`.
- Modify `Tests/Widgets/Library/test_library_note_folder_dialog.py`.
- Modify `Tests/UI/test_library_prompt_collections.py`.
- Modify `Tests/UI/test_library_file_notes_workspace.py`.
- Modify `Tests/UI/test_library_file_notes_git.py`.
- Modify `Tests/UI/test_library_file_notes_git_push.py`.

---

### Task 1: Restore focus by eligible stable identity

**Files:**

- Modify: `tldw_chatbook/Widgets/modal_dismissal.py`
- Modify: `Tests/UI/test_console_modal_dismissal.py`
- Create: `Tests/UI/test_library_modal_dismissal.py`

- [x] **Step 1: Write focus eligibility RED tests**

Add a small Library-like host screen that can recompose its opener while the modal is mounted. Cover:

1. the exact opener object remains mounted, attached, displayed, visible, enabled, and focusable;
2. the exact object becomes ineligible and one eligible replacement with the same non-empty ID exists;
3. the original is still mounted but hidden, disabled, or non-focusable and the replacement is eligible;
4. the ID is missing, duplicated, empty, or resolves only to ineligible widgets; and
5. the revealed screen exposes the Console composer fallback.

Assert the same eligibility predicate is used for the exact object and the ID replacement. Cases 4 must leave focus to the revealed screen rather than selecting an unrelated Library action. Case 5 must preserve the existing Console-only fallback.

- [x] **Step 2: Run the focus tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'stable_focus or opener_focus or console_composer_fallback'
```

Expected: the replacement-ID cases fail because the mixin currently stores only a weak reference to the original widget.

- [x] **Step 3: Add one shared eligibility helper and record the opener ID**

Keep the change local to `modal_dismissal.py`:

```python
def _is_safe_focus_target(widget: Widget | None) -> bool:
    return bool(
        widget is not None
        and widget.is_mounted
        and widget.is_attached
        and widget.display
        and widget.visible
        and not widget.disabled
        and widget.can_focus
    )
```

On mount, retain the existing weak reference and record only a non-empty `opener.id`. On dismissal:

- focus the exact opener only when `_is_safe_focus_target` passes;
- otherwise query the newly revealed screen for that exact ID and focus only when exactly one eligible match exists;
- otherwise invoke the existing Console composer hook only when the revealed screen provides it;
- never choose another Library control heuristically.

Clear both reference and ID on unmount. Do not add a Library import or focus registry.

- [x] **Step 4: Run Task 1 GREEN and shared regression checks**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'stable_focus or opener_focus or console_composer_fallback or single_shot or mount_generation'
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/modal_dismissal.py \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/ruff format --check \
  tldw_chatbook/Widgets/modal_dismissal.py \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/mypy --follow-imports=skip tldw_chatbook/Widgets/modal_dismissal.py
```

Expected: all selected tests and static checks pass.

- [x] **Step 5: Commit the focus boundary**

```bash
git add \
  tldw_chatbook/Widgets/modal_dismissal.py \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_library_modal_dismissal.py
git diff --cached --check
git commit -m "fix(ui): restore safe modal focus by stable identity"
```

---

### Task 2: Give shared file pickers one safe cancellation contract

**Files:**

- Modify: `tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py`
- Modify: `tldw_chatbook/Third_Party/textual_fspicker/select_directory.py`
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py`
- Modify: `Tests/UI/test_library_modal_dismissal.py`
- Modify: `Tests/UI/test_fspicker_keyboard_save.py`
- Modify: `Tests/UI/test_enhanced_file_dialog_mount.py`

- [x] **Step 1: Write base-picker contract and precedence RED tests**

For `FileOpen`, `FileSave`, and `SelectDirectory`, mount each concrete class and assert:

- the declared content selector resolves to the real bounded dialog;
- terminal Escape, visible `#cancel`, and a primary backdrop return exact `None` once;
- successful Open/Save/Select still return their existing `Path` values;
- non-primary and inside clicks stay open;
- a real expanded `SelectOverlay` receives first Escape and keeps the picker mounted;
- after the overlay closes, one combined state peels path editor, search, then recent locations in that order before terminal Escape dismisses;
- visible Cancel and backdrop are immediately terminal even while those transient surfaces are open; and
- clicks on path editor, search, recent list, directory options, and the expanded overlay remain inside.

Dispatch keys and clicks through Textual Pilot. Do not call `action_*` methods as the behavior oracle.

- [x] **Step 2: Write full-MRO and enhanced compatibility RED tests**

Instrument real `DirectoryNavigation.Changed` delivery in `SelectDirectory` and assert exactly one:

- breadcrumb update;
- current-path/recent hook update; and
- error clear.

Mount `EnhancedFileOpen` and `EnhancedFileSave` and assert their current path → search → recent → bookmarks smart-dismiss order, handler suppression, recent/last-directory persistence, visible Cancel/backdrop terminal result, and typed success results. Add an import smoke test for `tldw_chatbook.app` and `EnhancedFileDialog`.

- [x] **Step 3: Run picker tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  -k 'file_picker or fspicker or enhanced or directory_changed or application_import'
```

Expected failures:

- base pickers lack `SafeModalDismissMixin` and a stable content selector;
- base Escape dismisses instead of using source-aware transient precedence;
- `SelectDirectory` manually re-invokes full-MRO handlers; and
- leaving the explicit mixin on `EnhancedFileDialog` would become an inconsistent MRO after base adoption.

- [x] **Step 4: Adopt the mixin at `FileSystemPickerScreen`**

Make the shared base the single owner:

```python
class FileSystemPickerScreen(SafeModalDismissMixin, ModalScreen[Path | None]):
    SAFE_MODAL_CONTENT = "#file-system-picker-dialog"
```

Give the composed `Dialog` that stable ID. Replace the base Escape binding with `request_safe_cancel`. Implement source-aware cancellation with the existing state/actions:

- `source == "escape"`: close path editor, else search, else recent, else `dismiss_safe_once(None)`;
- `source in {"backdrop", "visible"}`: `dismiss_safe_once(None)` immediately.

Route the existing `#cancel` handler through `request_safe_cancel(source="visible")`. Keep selection and success paths unchanged.

- [x] **Step 5: Reconcile Textual full-MRO behavior minimally**

- Remove only the `@on(DirectoryNavigation.Changed)` decorator from the base `_clear_error` helper; retain the callable method because `BaseFileDialog._clear_error()` delegates to it for `Input.Changed`. `_on_directory_changed` remains the sole directory-change error-clear owner.
- Remove `SelectDirectory.on_mount()`'s manual `super().on_mount()` call. Retain only its subclass-specific `show_files = False` and input initialization in the subclass handler.
- Remove the decorated `SelectDirectory` handler's manual call to `super()._on_directory_changed(event)` and do not stop the shared `Changed` event before normal MRO dispatch.
- Preserve the base's no-op recent-location seam; do not add persistence.
- Change `EnhancedFileDialog(SafeModalDismissMixin, BaseFileDialog)` to `EnhancedFileDialog(BaseFileDialog)`.
- Keep `SAFE_MODAL_CONTENT = "#enhanced-file-dialog"`, `action_smart_dismiss`, `_SUPPRESSED_BASE_HANDLERS`, its persistence-aware `dismiss`, and typed results unchanged.

- [x] **Step 6: Run Task 2 GREEN and static checks**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  -k 'file_picker or fspicker or enhanced or directory_changed or application_import'
../../.venv/bin/ruff check \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py
../../.venv/bin/mypy --follow-imports=skip \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py
python -m compileall -q \
  tldw_chatbook/Third_Party/textual_fspicker \
  tldw_chatbook/Widgets/enhanced_file_picker.py
```

Expected: selected tests pass; new/changed lines introduce no Ruff, MyPy, or import diagnostics. If a legacy whole-file formatter or type baseline is red, compare the exact file at the task base and document only unchanged baseline findings rather than reformatting unrelated code.

- [x] **Step 7: Commit the shared picker contract**

```bash
git add \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py
git diff --cached --check
git commit -m "fix(ui): make shared file pickers safely dismissible"
```

---

### Task 3: Adopt ordinary Library modals with exact typed negatives

**Files:**

- Modify: `tldw_chatbook/UI/Screens/skills_screen.py`
- Modify: `tldw_chatbook/Widgets/ModelArtifacts/install_modal.py`
- Modify: `tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py`
- Modify: `tldw_chatbook/Widgets/Library/library_note_folder_dialog.py`
- Modify: `Tests/UI/test_library_modal_dismissal.py`
- Modify: `Tests/UI/test_library_skills_canvas.py`
- Modify: `Tests/UI/test_model_artifact_widgets.py`
- Modify: `Tests/UI/test_prompt_delete_confirmation_modal.py`
- Modify: `Tests/Widgets/Library/test_library_note_folder_dialog.py`

- [x] **Step 1: Add one mounted contract row per concrete class**

Add rows for:

| Concrete class | Stable content selector | Exact negative |
| --- | --- | --- |
| `SkillTrustPassphraseModal` | `#skill-trust-passphrase-modal` | `None` |
| `SkillTrustBootstrapModal` | `#skill-trust-bootstrap-modal` | `None` |
| `ModelInstallModal` | `.model-install-modal` | exact `False` |
| `PromptDeleteConfirmationModal` | `#prompt-delete-modal` | `PromptDeleteDecision(False, request.fingerprint)` |
| `LibraryNoteFolderNameDialog` | new `#library-note-folder-name-dialog` | `None` |
| `LibraryNoteFolderTargetDialog` | new `#library-note-folder-target-dialog` | `None` |

For every row, independently mount and test visible Cancel, terminal Escape, and primary backdrop. Assert selector existence, one callback, exact negative identity/value, and unchanged positive-result type. Also test one inside click and one non-primary backdrop click per concrete row.

- [x] **Step 2: Run ordinary-modal tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_prompt_delete_confirmation_modal.py \
  Tests/Widgets/Library/test_library_note_folder_dialog.py \
  -k 'library_modal_contract or skill_trust or model_install or prompt_delete or note_folder'
```

Expected: backdrop/contract failures because these classes have not adopted the mixin; folder selector rows fail because their outer containers are anonymous.

- [x] **Step 3: Apply mechanical safe adoption**

For each class:

- inherit `SafeModalDismissMixin` before `ModalScreen`;
- declare the exact `SAFE_MODAL_CONTENT` from the table;
- bind Escape to `request_safe_cancel`;
- route the visible Cancel path through `request_safe_cancel(source="visible")` or the modal's `_perform_safe_cancel` seam;
- implement `_perform_safe_cancel` only where the exact negative is not `None`;
- use `dismiss_safe_once` for the negative result;
- leave positive Submit/Install/Delete/Save/Choose behavior unchanged.

For `PromptDeleteConfirmationModal`, construct the negative using the current request fingerprint inside `_perform_safe_cancel`; do not cache a potentially stale decision. For `ModelInstallModal`, exact `False` must not toggle acknowledgement or start acquisition.

- [x] **Step 4: Prove lifecycle and result exactness GREEN**

Run the Step 2 command again, plus:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'ordinary_modal_lifecycle or repeated_input or exact_negative'
../../.venv/bin/ruff check \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py \
  Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/mypy --follow-imports=skip \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py
```

Expected: all focused behavior passes and Textual invokes mixin/class lifecycle handlers exactly once.

- [x] **Step 5: Commit ordinary Library adoption**

```bash
git add \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_prompt_delete_confirmation_modal.py \
  Tests/Widgets/Library/test_library_note_folder_dialog.py
git diff --cached --check
git commit -m "feat(library): dismiss ordinary Library modals safely"
```

---

### Task 4: Adopt File Notes and Git modal surfaces

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `Tests/UI/test_library_modal_dismissal.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify: `Tests/UI/test_library_file_notes_git_push.py`

- [x] **Step 1: Add concrete File Notes/Git contract RED tests**

Mount and independently exercise all three gestures for:

| Concrete class | Stable content selector | Exact negative |
| --- | --- | --- |
| `FileNotesRootDetailsDialog` | existing `#file-notes-root-details-dialog` | `None` |
| `FileNotesConflictCompareDialog` | existing `#file-notes-conflict-dialog` | `None` |
| `SessionGitTrustDialog` | inherited confirmation selector | exact `False` |
| `PushEndpointDetailsDialog` | existing `#file-notes-push-endpoint-details-dialog` | `None` |
| `PushDestinationAuthorizationDialog` | existing `#file-notes-push-auth-dialog` | exact `False` |

Add a real nested flow where authorization opens endpoint details. Cancel the top detail modal and assert only it closes and focus returns to its exact opener inside authorization; then cancel authorization and assert exact `False` once.

Record current explicit focus callbacks in the workspace/Git presenter paths and add an exact-once focus assertion before removing or retaining them.

- [x] **Step 2: Run File Notes/Git tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_library_file_notes_workspace.py \
  Tests/UI/test_library_file_notes_git.py \
  Tests/UI/test_library_file_notes_git_push.py \
  -k 'modal_dismissal or details_dialog or conflict_compare or trust_dialog or authorization_dialog or endpoint_details'
```

Expected: unadopted concrete classes lack backdrop behavior/stable selectors; nested focus may show duplicate presenter/mixin restoration.

- [x] **Step 3: Adopt the mixin without changing trust/push policy**

- Reuse the existing stable IDs on all five bounded outer containers; do not rename them because CSS and focused tests already depend on those selectors.
- Adopt `SafeModalDismissMixin` on direct `ModalScreen` classes and route exact negative values through `_perform_safe_cancel`/`dismiss_safe_once`.
- Preserve `SessionGitTrustDialog`'s inherited `ConfirmationDialog` contract rather than adding a second mixin base.
- Preserve `PushDestinationAuthorizationDialog.dismiss(False if result is None else result)` compatibility, but make generic safe requests explicitly return `False`.
- Keep endpoint-detail launch, positive authorization, and Git operations unchanged.
- Remove presenter-owned post-dismiss focus callbacks when the mixin is the sole owner. If one must remain for a non-modal workflow, retain it only with the mounted exact-once test.

- [x] **Step 4: Run Task 4 GREEN and static checks**

Run the Step 2 command again, then:

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_library_file_notes_workspace.py \
  Tests/UI/test_library_file_notes_git.py \
  Tests/UI/test_library_file_notes_git_push.py
../../.venv/bin/mypy --follow-imports=skip \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
```

Expected: all focused tests pass, nested top-screen ownership is exact, and no new changed-line diagnostics appear.

- [x] **Step 5: Commit File Notes/Git adoption**

```bash
git add \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_library_file_notes_workspace.py \
  Tests/UI/test_library_file_notes_git.py \
  Tests/UI/test_library_file_notes_git_push.py
git diff --cached --check
git commit -m "feat(library): dismiss File Notes modals safely"
```

---

### Task 5: Guard Prompt collection mutations without blocking input dispatch

**Files:**

- Modify: `tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py`
- Modify: `Tests/UI/test_library_modal_dismissal.py`
- Modify: `Tests/UI/test_library_prompt_collections.py`

- [x] **Step 1: Write real mutation-race RED tests**

Gate create, rename, and mutation-retry callbacks with `asyncio.Event`. Start each through a real mounted button press, then use `await asyncio.wait_for(started.wait(), timeout=1.0)` before delivering close input so a RED test cannot hang behind the currently blocked message pump. While the callback is blocked:

- assert mutation/selection/Done/retry controls are disabled;
- assert visible Cancel remains enabled;
- dispatch Escape, primary backdrop, and `pilot.click` on visible Cancel;
- assert the modal remains topmost and no result is emitted;
- assert the first rejected close changes `#prompt-collection-manager-outcome` to exactly `Finish the current collection change before closing.`;
- assert later rejected close requests do not append, toast, or otherwise change the line;
- release the gate and assert the mutation settles once and no queued close fires afterward.

Add same-instance unmount/remount tests for both success and failure completion. The callback must catch `asyncio.CancelledError`, wait on a second gate, and then deliberately return or raise. This proves a cancellation-resistant stale completion cannot clear in-flight state, repaint outcome, focus a control, or dismiss the new presentation.

- [x] **Step 2: Run mutation tests RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'collection_mutation_close or collection_mutation_remount or collection_modal_contract'
```

Expected: current async button handlers occupy the Textual message pump, Cancel is disabled, and close input can be delayed until `_mutation_in_flight` becomes false.

- [x] **Step 3: Claim synchronously, then run mutation work in a screen worker**

Adopt `SafeModalDismissMixin` with `SAFE_MODAL_CONTENT = "#prompt-collection-manager"`. Refactor only the create/rename/retry mutation path:

1. a synchronous button handler validates and claims `_mutation_in_flight` before returning to Textual;
2. it increments/captures a modal-owned mutation epoch alongside the current `_safe_mount_generation` and starts a screen-owned Textual worker for the awaited callback;
3. the worker uses the existing callback/error/result mapping;
4. before every completion mutation, compare both captured tokens and require this exact instance to remain mounted;
5. mutation completion restores existing success/error outcome and focus behavior only for the same presentation.

Add a local `on_unmount` handler that invalidates the mutation epoch and resets `_mutation_in_flight` plus the rejected-close feedback latch for a possible same-instance remount. Do not call `super().on_unmount()`; Textual dispatches matching lifecycle handlers across the full MRO, including the mixin's own cleanup. On a fresh mount, initialize the mutation-local state without allowing any old worker to reuse the new epoch.

Do not add a generalized operation manager. Leave catalog-loading token behavior unchanged unless its focused tests expose a shared defect.

- [x] **Step 4: Keep visible Cancel enabled as a guarded request**

During mutation recomposition:

- disable create, rename, selection/membership controls, Done, and retry;
- keep `#prompt-collection-manager-cancel` enabled;
- route Cancel, Escape, and backdrop into one `_perform_safe_cancel` implementation;
- when mutation is active, consume the request without dismissal and update the existing outcome line only if this mutation has not already reported a rejected close;
- when idle, return `None` through `dismiss_safe_once`.

Reset the feedback latch on the start/end boundary of each mutation. Do not add a toast, nested guard, or second status widget.

- [x] **Step 5: Run Task 5 GREEN and adjacent collection tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'collection or prompt_collection_manager'
../../.venv/bin/ruff check \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/ruff format --check \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/mypy --follow-imports=skip \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py
```

Expected: focused collection tests pass; real input remains dispatchable during the worker; no stale completion affects a remount.

- [x] **Step 6: Commit the mutation guard**

```bash
git add \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_modal_dismissal.py
git diff --cached --check
git commit -m "fix(library): guard Prompt collection modal dismissal"
```

---

### Task 6: Enforce the exact Library modal launch inventory

**Files:**

- Modify: `Tests/UI/test_library_modal_dismissal.py`

- [x] **Step 1: Define the explicit owner-edge contract table**

Separate modal behavior from graph ownership. Each concrete behavior row must record:

```python
@dataclass(frozen=True)
class LibraryModalContract:
    concrete_type: type[ModalScreen[Any]]
    factory: Callable[[], ModalScreen[Any]]
    content_selector: str
    visible_negative_selector: str
    negative_assertion: Callable[[object], None]
    positive_type: type[object] | tuple[type[object], ...]
    active_guard: str | None
    focus_postcondition: str
    non_dismissible_reason: str | None
```

Define launch ownership independently so one concrete modal can have multiple owners:

```python
@dataclass(frozen=True)
class LibraryModalLaunchEdge:
    owner_file: str
    owner_class: str
    presenter_name: str
    concrete_type: type[ModalScreen[Any]]
```

For example, declare separate `PushEndpointDetailsDialog` edges from `LibraryFileNotesWorkspace` and `PushDestinationAuthorizationDialog`.

Populate one row for every concrete type in the approved design, including already-safe reachable `WorkbenchHelpPanel`, `PromptVariablesDialog`, `ConfirmationDialog`, and every concrete reachable confirmation subclass. Keep `EnhancedFileOpen`/`EnhancedFileSave` in a separate compatibility table because they are not Library-reachable rows.

- [x] **Step 2: Add narrow bidirectional reachability RED tests**

Build a test-only walker that scans only the explicit supported owners:

- production `LibraryScreen` presenter methods;
- `LibraryPromptCollectionsController`'s declared `push_modal` seam, joined to a direct production-route test;
- `LibraryFileNotesWorkspace` and `LibraryFileNotesGitPanel` presenter methods; and
- modal-owned presenter methods such as authorization → endpoint details.

Resolve same-module classes, top/local imports, aliases, and attribute references only inside those named owner bodies. Compare discovered and declared `(owner, presenter, concrete modal)` edges exactly in both directions. Do not chase arbitrary calls or imports.

Add isolated synthetic mutation tests that inject one undeclared aliased modal into:

1. a controller-injected owner;
2. a nested widget owner; and
3. a modal-to-modal presenter.

Each mutation must make the exact-set assertion fail.

- [x] **Step 3: Run inventory RED and close missing rows/edges**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'library_modal_inventory or library_modal_contract_table'
```

Expected on first run: failures identify any concrete production edge or already-safe subclass omitted from the explicit table. Add contract rows or supported owner methods until production discovery and declaration are exactly equal. Do not weaken the assertion or broaden into a general analyzer.

- [x] **Step 4: Run every concrete row through all three gestures**

Parameterize the table so each concrete factory independently proves:

- content selector resolves in the mounted DOM;
- visible negative control returns the row's exact negative;
- terminal real Escape returns the same negative;
- primary real backdrop returns the same negative;
- one inside primary click and one outside non-primary click stay open;
- positive action/result type remains unchanged when feasible through the public UI;
- `non_dismissible_reason` is `None` for every current Library row.

Add full-MRO mount/unmount exact-once instrumentation for every class that defines its own lifecycle handler or inherits a decorated handler from another participating class.

- [x] **Step 5: Run inventory/contract GREEN**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  -k 'library_modal_inventory or library_modal_contract_table or concrete_library_modal or library_modal_lifecycle'
../../.venv/bin/ruff check Tests/UI/test_library_modal_dismissal.py
../../.venv/bin/ruff format --check Tests/UI/test_library_modal_dismissal.py
```

Expected: exact inventory equality, every concrete row green for all gestures, and all mutation-oracle tests green after restoring production source.

- [x] **Step 6: Commit the inventory closure**

```bash
git add Tests/UI/test_library_modal_dismissal.py
git diff --cached --check
git commit -m "test(library): enforce modal dismissal inventory"
```

---

### Task 7: Focused verification, mutation evidence, and task closeout

**Files:**

- Modify: `backlog/tasks/task-16350 - Make-all-Library-modals-dismiss-safely-with-Escape-or-backdrop-click.md`
- Modify: `Docs/superpowers/plans/2026-08-14-task-16350-library-modal-dismissal.md`
- Modify only if a real reusable incident occurred: `backlog/docs/lessons-testing-evidence.md` or another relevant lessons file.

- [x] **Step 1: Run the exact touched/related behavior matrix**

Run only:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_prompt_delete_confirmation_modal.py \
  Tests/Widgets/Library/test_library_note_folder_dialog.py \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_file_notes_workspace.py \
  Tests/UI/test_library_file_notes_git.py \
  Tests/UI/test_library_file_notes_git_push.py
```

Expected: all named related tests pass. If an unrelated pre-existing failure appears, reproduce the exact node at the task base before classifying it; do not expand to a broad directory run.

- [x] **Step 2: Run final targeted static checks**

Run Ruff check over every changed Python file, Ruff format check over new/already-formatted files, targeted MyPy over changed production files, `compileall` over changed production modules, and `git diff --check`.

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py \
  Tests/UI/test_library_modal_dismissal.py \
  Tests/UI/test_console_modal_dismissal.py \
  Tests/UI/test_fspicker_keyboard_save.py \
  Tests/UI/test_enhanced_file_dialog_mount.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/UI/test_model_artifact_widgets.py \
  Tests/UI/test_prompt_delete_confirmation_modal.py \
  Tests/Widgets/Library/test_library_note_folder_dialog.py \
  Tests/UI/test_library_prompt_collections.py \
  Tests/UI/test_library_file_notes_workspace.py \
  Tests/UI/test_library_file_notes_git.py \
  Tests/UI/test_library_file_notes_git_push.py
../../.venv/bin/ruff format --check \
  Tests/UI/test_library_modal_dismissal.py \
  tldw_chatbook/Widgets/modal_dismissal.py
../../.venv/bin/mypy --follow-imports=skip \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
python -m compileall -q \
  tldw_chatbook/Widgets/modal_dismissal.py \
  tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py \
  tldw_chatbook/Third_Party/textual_fspicker/select_directory.py \
  tldw_chatbook/Widgets/enhanced_file_picker.py \
  tldw_chatbook/UI/Screens/skills_screen.py \
  tldw_chatbook/Widgets/ModelArtifacts/install_modal.py \
  tldw_chatbook/Widgets/Library/prompt_delete_confirmation_modal.py \
  tldw_chatbook/Widgets/Library/library_note_folder_dialog.py \
  tldw_chatbook/UI/Library_Modules/prompt_collection_manager_modal.py \
  tldw_chatbook/Widgets/Library/library_file_notes_workspace.py \
  tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
git diff --check
```

Expected: no new/changed-line diagnostics. Record exact base provenance for any unchanged legacy whole-file formatting or type debt rather than rewriting unrelated code.

- [x] **Step 3: Perform and record mutation RED/GREEN checks**

Temporarily apply and restore these one-at-a-time:

1. disable the shared backdrop branch;
2. replace one exact negative (`False` or fingerprinted decision) with `None`;
3. remove the Prompt collection in-flight guard;
4. remove stable-ID focus restoration; and
5. remove one declared nested/controller/modal-to-modal inventory edge.

For each mutation, run its single focused test, record the expected failing assertion, restore the code, and rerun GREEN. Do not commit mutated source:

```bash
# 1. Disable the shared backdrop branch.
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py::test_library_modal_backdrop_mutation_oracle
# Expected RED: no negative result was emitted for the primary outside click.

# 2. Replace a fingerprinted or exact-False negative with None.
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py::test_library_modal_exact_negative_mutation_oracle
# Expected RED: result is not PromptDeleteDecision(False, fingerprint) / exact False.

# 3. Remove the Prompt collection in-flight guard.
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_prompt_collections.py::test_collection_mutation_close_guard
# Expected RED: the modal dismisses or emits a result while the callback gate is active.

# 4. Remove stable-ID focus restoration.
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py::test_library_modal_stable_id_focus_replacement
# Expected RED: the recomposed eligible opener does not receive focus.

# 5. Remove a declared nested/controller/modal-to-modal launch edge.
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_library_modal_dismissal.py::test_library_modal_inventory_detects_nested_edge
# Expected RED: discovered and declared LibraryModalLaunchEdge sets differ.
```

- [x] **Step 4: Self-review against all eight acceptance criteria**

Review the cumulative diff for:

- exact typed negative results and unchanged positive paths;
- real overlay/Escape/backdrop dispatch;
- source-aware picker precedence;
- one-shot/top-screen/mount-generation safety;
- mutation-time enabled Cancel and one status update;
- stable-ID eligibility symmetry;
- enhanced picker MRO/import/persistence compatibility; and
- exact bidirectional reachability.

Request one independent cumulative code review before closeout and resolve all Critical/Important findings. Add focused regressions for accepted findings; do not broaden test scope.

- [x] **Step 5: Complete Backlog and documentation hygiene**

In the task file:

- check every acceptance criterion only after its evidence is green;
- add concise `## Implementation Notes` with approach, trade-offs, exact focused test/static evidence, and mutation RED/GREEN evidence;
- record the ADR path and that no new ADR was needed because ADR-031 was amended;
- document any implementation-plan deviation; and
- add a lessons entry only if this task produced a new, evidenced reusable incident.

Because this repository has documented five-digit Backlog CLI quirks, verify the exact task before any status command. Use the CLI only if it resolves `TASK-16350` correctly; otherwise update the task file directly and record the verified CLI limitation rather than editing a different task.

- [x] **Step 6: Commit closeout documentation**

```bash
git add \
  'backlog/tasks/task-16350 - Make-all-Library-modals-dismiss-safely-with-Escape-or-backdrop-click.md' \
  Docs/superpowers/plans/2026-08-14-task-16350-library-modal-dismissal.md
git diff --cached --check
git commit -m "docs(library): complete safe modal dismissal task"
```

- [x] **Step 7: Verify final branch state**

Run:

```bash
git status --short
git log --oneline --decorate -10
```

Expected: clean worktree and the bounded task commits in dependency order. Do not create or merge a PR unless the user separately asks for integration.
