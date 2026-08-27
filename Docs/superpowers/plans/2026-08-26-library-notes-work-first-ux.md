# Library Notes Work-First UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Database Notes and Folder Files one calmer, work-first Notes experience with a collapsible Folder Files tree, more room for note bodies, boundary-only editor focus, and clearer progressive disclosure without losing any existing capability, authority, recovery, or state.

**Architecture:** Keep `LibraryScreen` as the single runtime preference and Notes-session coordinator, use `LibraryAdaptiveReaderShell` as the only Library/local-navigator/work geometry and focus-evacuation owner for both Notes authorities, and keep `LibraryFileNotesWorkspace` as the disk-authority/service owner while it supplies retained navigator and work regions to that shell. A small pure reducer owns only the transient `inactive | active | manually_cancelled` work-first lifecycle. Requested preferences, transient work-first requests, and responsive effective layout remain separate.

**Tech Stack:** Python 3.11+, Textual 8.x, TOML configuration, existing `ds-*` TCSS tokens and generated CSS bundle, pytest/pytest-asyncio with real Textual `run_test()` pilots.

---

## Governing records and scope gates

- Backlog task: `TASK-22513`
- Approved design: `Docs/superpowers/specs/2026-08-26-library-notes-ux-improvements-design.md`
- Existing architecture: `backlog/decisions/086-library-adaptive-reader-shell.md` and `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`
- ADR required: no
- ADR path: N/A
- Reason: this implements the already-approved adaptive-reader and progressive-disclosure boundaries; it adds no storage, service, ownership, security, dependency, or cross-module contract decision.

Before editing, inspect `git status --short` and the diff for every file in the task. This checkout contains unrelated user/concurrent changes. Never reset or overwrite them; stage only the exact task files after reviewing their complete diff.

The implementation must preserve this capability ledger:

- Database Notes: Edit, Preview, Info, filter/clear/sort, folders and placements, `#library-notes-new`, `#library-notes-add-from-files`, `#library-notes-manage-sync-folders`, `#library-notes-import-receipt`, whole-source `#library-notes-export`, selection and `#library-notes-export-selected`, undo, visible `#library-note-save`, primary `#library-note-use-in-console`, metadata/keywords/copy/export/delete in Info, conflict/recovery, and no invented bulk delete. Tests must continue to exercise the incumbent `LibraryScreen` handlers and their current guards/disabled reasons, not merely assert labels.
- Folder Files: autosave, `#file-notes-choose-root`, root details, tree/search, New, Move, Save Copy, Restore, Compare, Resolve, Reload, Protect, Refresh, Delete, session Git, large-file guard/export, conflict/reload recovery, and the retained exact-path authority. Tests must keep the existing service calls, transition/overwrite guards, disabled reasons, and focus return paths pinned. It gains Edit and Manage, not Markdown Preview or an ordinary Save button.
- Notes loses only the Notes-specific `Ctrl+S` binding and its Notes footer/F1 hints. The separately gated Skill-editor `Ctrl+S` remains.

Run only the targeted commands listed below. Do not run the full suite unless the user separately opts in.

## Task 1: Add the pure Notes work-session reducer

**Files:**

- Create: `tldw_chatbook/UI/Library_Modules/library_notes_work_session.py`
- Create: `Tests/UI/test_library_notes_work_session.py`

- [ ] **Step 1: Write the reducer contract as failing tests.**

  Cover these named transitions in a parameterized matrix:

  - `editable_item_opened` activates only from `inactive` at width `>= 120` after a Database note is loaded or a Folder file is successfully opened.
  - `manual_library_expand` changes `active` to `manually_cancelled`; it is otherwise idempotent.
  - selection/item changes, Edit/Preview/Info/Manage changes, save, conflict, recovery, resize, and Folder `back_to_navigator` preserve `active` and `manually_cancelled`.
  - `database_identity_cleared`, `folder_identity_cleared`, `authority_changed`, `folder_root_changed`, and `left_notes` reset to `inactive`.
  - another item open in the same session never rearms `manually_cancelled`.

  Keep event names explicit; do not encode reset semantics as a generic Boolean.

- [ ] **Step 2: Run the focused test and confirm RED.**

  Run: `python -m pytest Tests/UI/test_library_notes_work_session.py -q`

  Expected: collection/import failure because the reducer module does not exist.

- [ ] **Step 3: Implement the smallest pure reducer with this public seam.**

  ```python
  class NotesWorkSessionPhase(StrEnum):
      INACTIVE = "inactive"
      ACTIVE = "active"
      MANUALLY_CANCELLED = "manually_cancelled"

  class NotesWorkSessionEvent(StrEnum):
      # Use the explicit event names from Step 1.
      ...

  def reduce_notes_work_session(
      phase: NotesWorkSessionPhase,
      event: NotesWorkSessionEvent,
      *,
      reader_width: int | None = None,
  ) -> NotesWorkSessionPhase:
      ...
  ```

  `phase is ACTIVE` is the only force-closed signal consumed by `LibraryScreen`; no second Boolean is stored. The reducer must not import Textual, query widgets, read configuration, or write preferences.

- [ ] **Step 4: Run the reducer tests and confirm GREEN.**

  Run: `python -m pytest Tests/UI/test_library_notes_work_session.py -q`

- [ ] **Step 5: Commit the reducer only.**

  ```bash
  git add tldw_chatbook/UI/Library_Modules/library_notes_work_session.py Tests/UI/test_library_notes_work_session.py
  git commit -m "feat(library): add Notes work-session reducer"
  ```

## Task 2: Normalize and stage independent Folder-tree preferences

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_appearance_defaults.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/test_config_library_defaults.py`
- Modify: `Tests/UI/test_settings_appearance_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Add failing normalization tests.**

  Extend `Tests/test_config_library_defaults.py` to prove:

  - `[library.notes_reader].items_open/items_width` and `files_tree_open/files_tree_width` normalize independently.
  - Folder-tree defaults are open and `ITEMS_TARGET_WIDTH`.
  - `TLDW_LIBRARY_NOTES_READER_FILES_TREE_OPEN` and `TLDW_LIBRARY_NOTES_READER_FILES_TREE_WIDTH` override TOML per key.
  - malformed Folder-tree values fall back without rewriting or borrowing Database `items_*` values.

- [ ] **Step 2: Add failing Settings model and UI tests.**

  Extend the Appearance tests for `library_notes_files_tree_open` and `library_notes_files_tree_width`: load, strict validation, default reset, deep-merge save, widget sync, field search, dirty tracking, and labels `Folder Files tree pane` / `Folder Files tree width`.

- [ ] **Step 3: Run the focused configuration/Settings batch and confirm RED.**

  Run:

  ```bash
  python -m pytest Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py -q
  ```

  Expected: new assertions fail because the keys and Settings controls do not exist.

- [ ] **Step 4: Implement normalization and Settings ownership.**

  In `config.py`, extend only `library.notes_reader` with the two Folder-tree keys and environment overrides. In `SettingsAppearanceDefaults`, add the two typed fields and include them in load, validation, reset, and save-section construction. In `settings_screen.py`, render the two explicit Notes Folder-tree controls beside Notes Items; do not add Folder Files as a fake destination to `LIBRARY_READER_DESTINATIONS`, because both preference pairs intentionally share one `[library.notes_reader]` section.

  Settings remains the only width mutation owner. Runtime grips may later write only open/closed state.

- [ ] **Step 5: Re-run the focused batch and confirm GREEN.**

  Run the Step 3 command again.

- [ ] **Step 6: Commit only the configuration and Settings slice.**

  ```bash
  git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py
  git commit -m "feat(settings): add Folder Files tree preferences"
  ```

## Task 3: Make shared-shell focus restoration manual and authority-neutral

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`
- Modify: `Tests/UI/test_library_adaptive_reader_shell.py`
- Verify unchanged consumer contracts: `Tests/UI/test_library_conversation_reader.py`
- Verify unchanged consumer contracts: `Tests/UI/test_library_media_reader_shell.py`

- [ ] **Step 1: Add failing focus-memory tests.**

  Extend the existing `_ProbeApp` with two focusable controls per optional pane. Assert:

  - collapsing a focused pane records the last valid focused descendant and evacuates focus to that pane's five-column grip;
  - manual reopen restores the recorded control;
  - if that control is removed or disabled, manual reopen chooses the first valid focus target in the pane;
  - a responsive-only reopen never steals focus from work;
  - retained child identities and the fixed five-column grip geometry do not change.

- [ ] **Step 2: Run the shell tests and confirm RED.**

  Run: `python -m pytest Tests/UI/test_library_adaptive_reader_shell.py -q`

- [ ] **Step 3: Extend `sync_layout` with this explicit manual-restore signal.**

  ```python
  def sync_layout(
      self,
      layout: AdaptiveReaderEffectiveLayout,
      *,
      manual_reopen: PaneName | None = None,
  ) -> None:
      ...
  ```

  Retain one last-focused descendant per optional pane in `_last_focused_descendant: dict[PaneName, Widget | None]`. Record it before hiding. Preserve the existing focus-to-grip evacuation. Only `manual_reopen="library"` or `manual_reopen="items"` may restore focus. Normal mount, resize, hysteresis, and priority resolution omit the argument and therefore cannot steal focus.

  Keep the shell authority-neutral: it knows only `library`, `items`, and `work`, never Database Notes or Folder Files.

- [ ] **Step 4: Re-run the shell and unchanged-consumer tests and confirm GREEN.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_media_reader_shell.py -q
  ```

- [ ] **Step 5: Commit the shell slice.**

  ```bash
  git add tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py Tests/UI/test_library_adaptive_reader_shell.py
  git commit -m "feat(library): restore adaptive pane focus on manual reopen"
  ```

## Task 4: Put Folder Files in the shared adaptive shell

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify: `Tests/UI/test_library_notes_reader.py`

- [ ] **Step 1: Add failing authority-scaffold, structural, and retention tests.**

  Add these exact tests:

  - `Tests/UI/test_library_notes_reader.py::test_folder_files_reader_authority_scaffold_is_distinct`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_builds_shared_adaptive_reader_roles`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_shared_shell_retains_state_across_breakpoints`
  - `Tests/UI/test_library_file_notes_workspace.py::test_notes_authority_round_trip_retains_both_workspaces`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_routes_rail_descendants_to_visible_authority`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_low_height_rail_scrolls_to_last_action`

  Add production-shell pilots proving that Folder Files has exactly one `LibraryAdaptiveReaderShell` with this mapping:

  - `library`: the existing Library rail;
  - `items`: the retained Folder tree/search navigator;
  - `work`: the retained incumbent Folder authority/editor/action/recovery surface; Task 6 later reorganizes this same retained region into Edit/Manage without changing the Task 4 shell contract.

  Assert `LibraryScreen` has separate Database and Folder requested/effective projections before Files mode handles a grip: `_library_notes_reader_preferences/_layout` and `_library_file_notes_reader_preferences/_layout`. Assert that both grips remain five columns, the Folder items grip is independently operable as `notes_file_items`, and the old `LibraryFileNotesWorkspace._narrow` display toggles no longer own navigator/editor geometry.

  Across manual collapse/reopen and widths 160, `120 → 119 → 160 restored`, 100, 80, 79, and 60, retain editor identity, text, cursor/selection, undo, tree expansion/selection, search results, Git widget/operation state, recovery state, scroll, and autosave. Add a Database → Folder → Database → Folder round trip that preserves each authority's independent selection, draft, cursor/undo/scroll, navigator state, and Folder Git/recovery state. Reducer reset behavior is integrated and asserted in Task 5, after the workspace notification seam exists.

- [ ] **Step 2: Run the structural tests and confirm RED.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py -q
  ```

- [ ] **Step 3: Add the Folder reader projection and authority mapping, then run its focused tests GREEN.**

  Add `LibraryReaderDestination = Literal["media", "conversations", "notes", "notes_files"]` and centralize the mapping with one immutable target record (name it `_LibraryReaderPersistenceTarget`) containing `section`, `config_key`, `authority`, and `preferences_attribute`. The `notes_files/items` entry must be exactly:

  ```python
  _LibraryReaderPersistenceTarget(
      section="library.notes_reader",
      config_key="files_tree_open",
      authority="notes_file_items",
      preferences_attribute="_library_file_notes_reader_preferences",
  )
  ```

  Create the Folder preference/layout projection plus distinct generation, durable-value, and lock entries before any Files grip can fire. Share only Library-pane preference fields. At this checkpoint, run:

  ```bash
  python -m pytest Tests/UI/test_library_notes_reader.py::test_folder_files_reader_authority_scaffold_is_distinct -q
  ```

  It must pass before changing Folder DOM structure.

- [ ] **Step 4: Split Folder presentation into retained navigator and work regions, then run retention tests GREEN.**

  Keep `LibraryFileNotesWorkspace` mounted as the disk session/service/timer/event owner. Refactor its composition so the Library rail, its navigator region, and its work region are descendants of one `LibraryAdaptiveReaderShell`; reuse the current tree, editor, Git, recovery, and action widget instances rather than recreating them during mode or layout changes. Use this narrow workspace seam:

  ```python
  def configure_reader_shell(
      self,
      *,
      library_pane: Widget,
      layout: AdaptiveReaderEffectiveLayout,
  ) -> None:
      ...

  def sync_reader_layout(
      self,
      layout: AdaptiveReaderEffectiveLayout,
      *,
      manual_reopen: PaneName | None = None,
  ) -> None:
      ...
  ```

  `configure_reader_shell` is called while the retained workspace is detached/before compose; mounted changes use `sync_reader_layout`. Run:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_folder_files_builds_shared_adaptive_reader_roles -q
  ```

  Make it green before deleting legacy geometry.

- [ ] **Step 5: Remove the competing Folder geometry owner, then run breakpoint tests GREEN.**

  Move the existing `<80` Folder behavior into adaptive-layout `items` priority. Remove the workspace's competing `_narrow` width/display mutations only after the shared-shell tests pass. Prove the `120 → 119 → 160` restoration and the existing 80/79/60 behavior. Do not change root binding, filesystem validation, autosave, transition admission, or session-Git services. Run:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_folder_files_shared_shell_retains_state_across_breakpoints -q
  ```

- [ ] **Step 6: Route Folder shell resize/toggle messages through `LibraryScreen`, then run authority round trips GREEN.**

  Files mode should use the same shared-shell message path as Database Notes. Manual Folder tree toggles identify the distinct `notes_file_items` authority. Automatic `<80`, focus-priority, and responsive resolutions update only effective layout and never write configuration. Run:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_notes_authority_round_trip_retains_both_workspaces -q
  ```

- [ ] **Step 7: Re-run the complete structural batch and confirm GREEN.**

  Run the Step 2 command again.

- [ ] **Step 8: Commit the shell migration.**

  ```bash
  git add tldw_chatbook/Widgets/Library/library_file_notes_workspace.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_notes_reader.py
  git commit -m "refactor(library): share adaptive shell with Folder Files"
  ```

## Task 5: Wire independent persistence and work-first behavior

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_notes_reader.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_shell.py`

- [ ] **Step 1: Add failing persistence-authority tests.**

  Prove `notes_items` writes `[library.notes_reader].items_open` and `notes_file_items` writes `[library.notes_reader].files_tree_open`. Cover optimistic mirror, success, write failure rollback, readback/configuration failure copy, and opposite-order generation races. Assert neither authority can roll back or overwrite the other, and neither grip writes width.

- [ ] **Step 2: Add failing work-session integration pilots.**

  Add these exact tests:

  - `Tests/UI/test_library_notes_reader.py::test_database_notes_work_session_activates_once_and_resets_exactly`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_emits_only_admitted_work_session_events`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_notes_work_session_activates_once_and_resets_exactly`
  - `Tests/UI/test_library_file_notes_workspace.py::test_notes_authority_round_trip_resets_only_transient_work_session`

  For Database Notes and Folder Files, assert:

  - entering Notes setup/list/root selection does not auto-collapse;
  - successfully opening editable work at width `>=120` closes only the effective Library request once;
  - when the saved Library request is already open, manual expansion cancels work-first with zero configuration writes;
  - when the saved Library request is closed, manual expansion cancels work-first and produces exactly one `library_open=true` write;
  - switching item or Edit/Preview/Info/Manage, saving, conflict, recovery, and resize do not rearm it;
  - Database selected/loaded identity clear, Folder opened-file identity explicit clear, source/root change, and leaving Notes reset it;
  - Folder `Back to navigator` at narrow width retains the opened file and does not reset;
  - automatic work-first and responsive changes produce zero configuration writes.

- [ ] **Step 3: Run the focused integration tests and confirm RED.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_library_notes_work_session.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_shell.py -q
  ```

- [ ] **Step 4: Complete persistence behavior on the authority scaffold from Task 4.**

  Use Task 4's `_LibraryReaderPersistenceTarget` mapping for optimistic mirror, serialized write, authoritative readback, rollback, and generation reconciliation. On Settings refresh, reload both projections. A Folder grip writes only `files_tree_open`; a Database grip writes only `items_open`; neither writes width. Manual reopen passes `manual_reopen` to the shell; responsive reopen does not.

- [ ] **Step 5: Integrate the pure reducer at named lifecycle boundaries.**

  Add these workspace-to-screen messages in `library_file_notes_workspace.py`:

  ```python
  class FileNotesEditableOpened(Message):
      def __init__(self, identity: str) -> None: ...

  class FileNotesIdentityCleared(Message):
      pass

  class FileNotesRootChanged(Message):
      def __init__(self, root: Path) -> None: ...
  ```

  Emit `FileNotesEditableOpened` only after the current open request successfully applies its document; failed, cancelled, and stale opens emit nothing. Emit `FileNotesIdentityCleared` only when the opened-file identity is explicitly closed/cleared; narrow `Back to navigator` emits nothing. Emit `FileNotesRootChanged` only after an admitted root transition becomes the current binding; rejected/stale root changes emit nothing. `LibraryScreen` handles these messages and dispatches the corresponding reducer events.

  Dispatch Database events only after successful load and at its exact identity-clear boundaries. Derive work-first layout from the reducer without mutating requested preferences. A manual Library grip expansion during `active` transitions the reducer to `manually_cancelled`: saved-open requests perform no write; saved-closed requests perform exactly one open write.

  Run the message and lifecycle checkpoints directly:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_folder_files_emits_only_admitted_work_session_events Tests/UI/test_library_notes_reader.py::test_database_notes_work_session_activates_once_and_resets_exactly Tests/UI/test_library_file_notes_workspace.py::test_folder_notes_work_session_activates_once_and_resets_exactly Tests/UI/test_library_file_notes_workspace.py::test_notes_authority_round_trip_resets_only_transient_work_session -q
  ```

- [ ] **Step 6: Re-run the integration batch and confirm GREEN.**

  Run the Step 3 command again.

- [ ] **Step 7: Commit the runtime policy slice.**

  ```bash
  git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_shell.py
  git commit -m "feat(library): apply work-first Notes sessions"
  ```

## Task 6: Reorganize Notes controls, modes, and status hierarchy

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_notes_reader.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_file_notes_git.py`
- Modify: `Tests/UI/test_library_honesty_accessibility.py`

- [ ] **Step 1: Write failing capability-inventory and mode tests.**

  Add these exact tests:

  - `Tests/UI/test_library_notes_reader.py::test_database_notes_capability_inventory_and_modes`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_capability_inventory_and_modes`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_primary_safe_recovery_actions_bypass_manage`
  - `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_edit_manage_retains_editor_git_and_recovery`

  Add an explicit assertion for every selector/handler in the governing ledger, including current guard and disabled-reason behavior. Prove Database modes are exactly Edit/Preview/Info and Folder modes are exactly Edit/Manage. Assert `#library-note-use-in-console` remains a primary Database task action, Folder has no Markdown Preview and no ordinary Save control, Database has no bulk delete, and visible Database Save remains keyboard reachable through ordinary focus/F6 navigation.

- [ ] **Step 2: Add failing Folder named-path-task tests.**

  Add `Tests/UI/test_library_file_notes_workspace.py::test_folder_files_path_tasks_are_exclusive_execute_and_restore_focus`.

  Model only `none | new | move | save_copy`. Assert opening another task respects the existing guard, Cancel/guarded Esc closes it and returns focus to its opener, and Save Copy remains immediately reachable in conflict/reload recovery. The target-path input should not consume editor height until one of those tasks is active.

  Pin this workspace API:

  ```python
  FileNotesPathTask = Literal["none", "new", "move", "save_copy"]

  async def _open_path_task(
      self,
      task: FileNotesPathTask,
      *,
      opener_id: str,
  ) -> bool:
      ...

  def _close_path_task(self, *, restore_focus: bool = True) -> None:
      ...

  async def _submit_path_task(self) -> bool:
      ...
  ```

  Use stable selectors `#file-notes-path-task`, existing `#file-notes-path`, `#file-notes-path-submit`, and `#file-notes-path-cancel`. Existing `#file-notes-new`, `#file-notes-move`, Manage's `#file-notes-save-copy`, and contextual recovery's new `#file-notes-recovery-save-copy` open the named task after `flush_pending_work()`/the incumbent transition guard admits replacement. Both Save Copy controls call `_open_path_task("save_copy", opener_id=event.button.id)` so cancellation returns focus to the actual opener. Submit maps exactly as follows:

  - `new` → the validation and `service.create_file` path currently in `_new_file`;
  - `move` → the validation and `service.move_file` path currently in `_move_file`;
  - `save_copy` → `_save_editor_copy`, preserving exact-file export and no-clobber draft-copy behavior.

  Extract service execution from the current button handlers rather than synthesizing a second implementation. Cancel and guarded Esc call `_close_path_task(restore_focus=True)`; successful submit closes the task through the same focus-safe seam.

- [ ] **Step 3: Add failing header/status matrix tests.**

  Add `Tests/UI/test_library_notes_reader.py::test_notes_header_status_channels_follow_approved_precedence` and parameterize both Database and Folder projections.

  Test the two independent status channels and precedence:

  - content/recovery: conflict → unavailable/read-only blocker → save failure → saving → dirty → saved/clean;
  - authority/Git: always name Database authority or Folder root; then failure/uncertain → running → changes; omit ordinary clean Git success.

  Git must never replace a content conflict, failed save, or safe next action. At wide size, headers use no more than two logical rows. At constrained size, consequential content/recovery copy remains complete; exact authority/path and secondary Git detail remain available in Info/Manage.

  Drive both headers through named pure projections rather than concatenating status in async handlers:

  ```python
  @dataclass(frozen=True)
  class NotesStatusChannels:
      content_recovery: str
      authority_git: str
      safe_next_action: str | None = None

  def resolve_database_note_status_channels(...) -> NotesStatusChannels:
      ...

  def resolve_file_note_status_channels(...) -> NotesStatusChannels:
      ...
  ```

- [ ] **Step 4: Run the focused mode/action/status tests and confirm RED.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_honesty_accessibility.py -q
  ```

- [ ] **Step 5: Reorganize Database Notes without deleting handlers, then run Database capability nodes GREEN.**

  Keep visible Save and Use in Console in the primary task group. Move metadata, copy/export, destructive, and maintenance details into Info while preserving their current events, guards, disabled reasons, and recovery focus. Keep all navigator workflows in their approved locations and retain widget/session identity across Edit/Preview/Info. Run:

  ```bash
  python -m pytest Tests/UI/test_library_notes_reader.py::test_database_notes_capability_inventory_and_modes -q
  ```

- [ ] **Step 6: Add Folder Edit/Manage progressive disclosure, then run identity/capability nodes GREEN.**

  Keep editor, contextual recovery, and consequential status in Edit. Keep Choose folder and root details in the browse/setup header. Keep Restore contextual to a deleted-file selection. Keep Compare and Resolve in contextual conflict recovery, with `#file-notes-recovery-save-copy` immediately reachable there and `#file-notes-save-copy` in Manage; both route through the same named task/service seam. Put exact-path details, Move/Save Copy/Reload/Protect/Refresh, Session Git, and Danger/Delete into stable Manage groups. Keep New in the tree header. Safe recovery must not require entering Manage. Toggle display only; do not remount the editor, Git panel, or active recovery widgets.

  Run:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_folder_files_capability_inventory_and_modes Tests/UI/test_library_file_notes_workspace.py::test_folder_files_primary_safe_recovery_actions_bypass_manage Tests/UI/test_library_file_notes_workspace.py::test_folder_files_edit_manage_retains_editor_git_and_recovery -q
  ```

- [ ] **Step 7: Implement named path tasks, then run path/recovery nodes GREEN.**

  Reuse existing validation, save-copy non-overwrite, reload, deletion, and transition guards. Replace the always-visible target-path row with the one active task presentation through `_open_path_task`, `_submit_path_task`, and `_close_path_task`. Run:

  ```bash
  python -m pytest Tests/UI/test_library_file_notes_workspace.py::test_folder_files_path_tasks_are_exclusive_execute_and_restore_focus -q
  ```

- [ ] **Step 8: Implement the deterministic status projections, then run the full status matrix GREEN.**

  Have async state changes update inputs only; render each header through the named pure resolver so partial updates cannot reorder precedence. Always include authority/root in `authority_git`, omit ordinary clean Git success, and keep recovery callouts in the form “what happened / draft impact / safest next action.” Run:

  ```bash
  python -m pytest Tests/UI/test_library_notes_reader.py::test_notes_header_status_channels_follow_approved_precedence -q
  ```

- [ ] **Step 9: Re-run the focused batch and confirm GREEN.**

  Run the Step 4 command again.

- [ ] **Step 10: Commit the information-architecture slice.**

  ```bash
  git add tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_honesty_accessibility.py
  git commit -m "feat(library): simplify Notes work surfaces"
  ```

## Task 7: Apply boundary-only note-body focus and remove Notes Ctrl+S

**Files:**

- Modify: `tldw_chatbook/css/components/_forms.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Regenerate if changed by the builder: `tldw_chatbook/css/.css-build-manifest.json`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_focus_accessibility.py`
- Modify: `Tests/UI/test_library_honesty_accessibility.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_shell.py` only to migrate stale Notes-specific `Ctrl+S` behavior/footer assertions to the visible Save and no-shortcut contract; preserve unrelated dirty hunks.
- Modify: `Tests/UI/test_css_staleness_manifest.py` only if a builder-contract test is required; do not change it merely to bless stale output.

- [ ] **Step 1: Add failing focus-style tests.**

  Under the real generated bundle and every shipped theme, measure `#library-note-body` and `#file-notes-editor` before and after focus. Assert identical computed background, unchanged region geometry, a `heavy` outline using an existing semantic `ds-*` focus token, and boundary contrast of at least 3:1 against adjacent surfaces. Also assert a compact Input and an unrelated TextArea retain the current focused-fill behavior.

- [ ] **Step 2: Add failing shortcut/honesty tests.**

  Assert Notes no longer registers, advertises in the footer, or lists in F1 any `Ctrl+S` Save action. Assert the Skill editor still does. Assert Database Save is visible/focusable and Folder Files still has no manual Save.

- [ ] **Step 3: Run the focused accessibility tests and confirm RED.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_focus_accessibility.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_library_file_notes_workspace.py -q
  ```

- [ ] **Step 4: Add exact-ID TCSS exceptions and remove only the Notes binding/hints.**

  Scope focus exceptions exactly to `#library-note-body:focus` and `#file-notes-editor:focus`. Preserve resting background and geometry, add the heavy semantic outline, and do not change title/path/search fields or global TextArea rules. Remove only `library_notes_save` from `LibraryScreen.BINDINGS` and the Notes editor shortcut tuples/help projection; keep the visible Save handler and the separately gated Skill binding.

- [ ] **Step 5: Regenerate CSS from source.**

  Run: `python tldw_chatbook/css/build_css.py`

  Never edit `tldw_chatbook/css/tldw_cli_modular.tcss` by hand. Review the generated diff and confirm it contains the exact-ID rules once.

- [ ] **Step 6: Re-run accessibility and CSS integrity tests.**

  Run:

  ```bash
  python -m pytest Tests/UI/test_focus_accessibility.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_widget_css_consolidation.py Tests/UI/test_css_staleness_manifest.py -q
  ```

- [ ] **Step 7: Commit the focus and shortcut slice.**

  ```bash
  git add tldw_chatbook/css/components/_forms.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/.css-build-manifest.json tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_focus_accessibility.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_shell.py Tests/UI/test_css_staleness_manifest.py
  git commit -m "fix(library): quiet Notes editor focus"
  ```

  If the manifest path or a listed test file is unchanged, omit it from `git add` rather than forcing it into the commit.

## Task 8: Document, verify live behavior, and close TASK-22513

**Files:**

- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `Docs/User_Guide/library/file-notes.md`
- Modify: `backlog/tasks/task-22513 - Polish-Library-Notes-work-first-editors-and-Folder-Files-shell.md`
- Modify only if an actual generalizable incident occurred: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`
- Modify for the shared-index incident encountered during Task 7: `backlog/docs/lessons-backlog-hygiene.md`

- [ ] **Step 1: Update both user guides.**

  Document Database Edit/Preview/Info, Folder Edit/Manage, both independent collapsible local navigators, once-per-work-session collapse/manual override/reset behavior, autosave versus visible Database Save, boundary-only body focus, named Folder path tasks, recovery/Git locations, and the removal of Notes `Ctrl+S` without implying a replacement shortcut.

- [ ] **Step 2: Run the complete targeted automated verification.**

  ```bash
  python -m pytest Tests/UI/test_library_notes_work_session.py Tests/Library/test_library_adaptive_reader_state.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_focus_accessibility.py Tests/UI/test_widget_css_consolidation.py Tests/UI/test_css_staleness_manifest.py -q
  ```

  Expected: all targeted tests pass. If an unrelated failure appears, reproduce the exact node on the unchanged base before classifying it as pre-existing; do not substitute a different command as evidence.

- [ ] **Step 3: Perform an isolated-profile live Textual walkthrough.**

  Read `backlog/docs/lessons-live-verification.md` immediately before the run. Create a scratch root and set absolute `HOME`, `USERPROFILE`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `TLDW_CONFIG_PATH`, `TMPDIR`, and `[paths].data_dir` before Python imports. Verify the effective paths are scratch paths, seed synthetic Database Notes and a temporary Folder Files root, and use no real user profile or real notes folder.

  Walk both authorities at 160x40, then `120x35 → 119x35 → 160x40 restored`, plus 100x30, 80x24, 79x24, and 60x20. Verify:

  - initial/list/setup states do not collapse unexpectedly;
  - first editable open at wide size collapses Library once, both grips remain usable, manual reopen wins, and each declared reset starts a new session;
  - Folder tree and Database list preferences remain independent across restart;
  - resize/hysteresis and Folder narrow Back do not persist or reset state;
  - Database → Folder → Database → Folder preserves both selections, drafts, cursor/undo/scroll, navigator state, and Folder Git/recovery state while the work-session reducer alone resets;
  - editor text, cursor, undo, tree/search, Git, recovery, autosave, and focus survive collapse/mode/resize;
  - headers remain bounded and consequential status/recovery actions remain complete;
  - only the two note bodies use boundary-only focus and there is no layout shift;
  - every capability in the ledger is reachable, Folder has no Preview/Save, Database has no bulk delete, and Notes has no `Ctrl+S` hint/binding.

  Record the exact commands, terminal sizes, scratch paths (never secrets), and observed results in the Backlog Implementation Notes.

- [ ] **Step 4: Perform final static checks.**

  Run:

  ```bash
  git diff --check
  python -m compileall -q tldw_chatbook/UI/Library_Modules/library_notes_work_session.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
  python -m ruff check tldw_chatbook/config.py tldw_chatbook/UI/Library_Modules/library_notes_work_session.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_library_notes_work_session.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_focus_accessibility.py Tests/UI/test_css_staleness_manifest.py
  python -m ruff format --check tldw_chatbook/config.py tldw_chatbook/UI/Library_Modules/library_notes_work_session.py tldw_chatbook/UI/Screens/settings_appearance_defaults.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/test_config_library_defaults.py Tests/UI/test_settings_appearance_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_library_notes_work_session.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_focus_accessibility.py Tests/UI/test_css_staleness_manifest.py
  ```

  Inspect `git status --short` and every task-file diff. Confirm unrelated dirty files remain untouched and no generated CSS source/bundle mismatch remains.

- [ ] **Step 5: Complete Backlog hygiene.**

  In the task file:

  - check all acceptance criteria only after their evidence exists;
  - add concise Implementation Notes with approach, trade-offs, exact files, test commands/results, live walkthrough evidence, and `ADR required: no` / `ADR path: N/A`;
  - add a lesson only if implementation exposed a real, generalizable incident;
  - set status to Done with `backlog task edit 22513 -s Done` only after every Definition-of-Done item is satisfied.

- [ ] **Step 6: Commit documentation and task closure.**

  ```bash
  git add Docs/User_Guide/library/notes.md Docs/User_Guide/library/file-notes.md 'backlog/tasks/task-22513 - Polish-Library-Notes-work-first-editors-and-Folder-Files-shell.md'
  git commit -m "docs(library): document work-first Notes UX"
  ```

## Final review gate

Before claiming completion, use `superpowers:requesting-code-review`, address findings through `superpowers:receiving-code-review`, and then use `superpowers:verification-before-completion`. The final report must distinguish targeted automated evidence, isolated live evidence, and any unrun full-suite evidence; never imply the full suite passed unless the user separately authorized and it actually ran.
