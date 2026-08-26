---
id: TASK-17962
title: >-
  Workspace create modal: follow-up tests and typed seams
status: Done
assignee: []
created_date: '2026-08-17 21:00'
labels:
  - workspaces
  - testing
priority: medium
dependencies:
  - TASK-18704
---

## Description (the why)

The TASK-18704 final whole-branch review closed with a set of deliberately
deferred coverage and typing gaps (its Recommendations §3). They are known,
ledgered, and none blocks the shipped behavior — but PR B (TASK-18705) reads
more fields off `WorkspaceCreateResult`, so these seams should be pinned and
typed before that work builds on them.

## Acceptance Criteria (the what)

- [x] Activation-failure path (`set_active_workspace` raising after a successful create) has a seam test on each surface: Console handler, Settings `_done`, Library `_done` (Library's reorder is already pinned; Console/Settings are not)
- [x] The Enter-submit fast path (`Input.Submitted` on `#workspace-create-name` → `_create`) has a pilot test using a real keypress (not a direct method call), per the task-17961 blind-spot lesson
- [x] A test produces `failed_folders` through the real TOCTOU path (folder deleted between Add and Create), not a synthetic result object
- [x] `_remove_folder` and unchecked-checkbox-survives-recompose have pilot coverage
- [x] The three `_done`/handler callbacks and `WorkspaceCreateResult.project_skills` carry real type annotations (`WorkspaceCreateResult | None`; a typed tuple for `project_skills`)
- [x] Decide (and implement or explicitly reject) restoring per-surface `description` provenance ("Created from Console/Settings/Library") lost to the uniform modal description

## Notes

Source: final review of `feat/workspace-create-modal` (see
`Docs/superpowers/plans/2026-08-17-workspace-create-modal.md` and the spec's
§7). Related open defect: TASK-17961 (focused compact-widget rendering).

## Implementation Plan (the how)

1. Reconcile each AC against the current tree (PR #1809/#1810 final-fix
   waves landed some of this already) before writing anything new; cite
   file::test evidence for anything already covered.
2. Library's activation-failure seam was already pinned
   (`test_create_workspace_recomposes_after_activation_failure`); add the
   matching seam for Console (unit-level, via the existing
   `_Stub`/`ConsoleWorkspaceController._handle_workspace_create_result`
   harness) and Settings (pilot-level, via the existing
   `test_create_rename_archive_unarchive_flow`-style harness), since
   neither had one.
3. Add a real-keypress Enter-submit test, a real-TOCTOU `failed_folders`
   test (delete the folder from disk between Add and Create), and a
   real-click checkbox-survives-recompose test to
   `Tests/Workspaces/test_workspace_create_modal.py`.
4. Add `WorkspaceCreateResult | None` annotations to the three surfaces'
   result callbacks (`TYPE_CHECKING`-gated imports to avoid real import
   cycles), and confirm `WorkspaceCreateResult.project_skills` is already
   a typed tuple.
5. Implement the controller-ruled `description` provenance restore:
   optional `description` param on `WorkspaceCreateModal.__init__`, each
   surface passing its own wording, one test pinning it end-to-end plus
   one pinning the modal's own default.
6. Run the full gate (`Tests/Workspaces/`, `Tests/Skills/`, the three
   named `Tests/UI/` files) and a repo-wide `--collect-only` sweep.

## Implementation Notes

**Reconciliation (ticked-by-evidence vs. newly implemented):**

- Activation-failure seam: Library was ALREADY covered by
  `test_create_workspace_recomposes_after_activation_failure` in
  `Tests/UI/test_post_release_workspaces_library_depth.py:553`. Console and
  Settings were NOT covered — implemented
  `test_activation_failure_notifies_error_and_skips_sync` in
  `Tests/Workspaces/test_console_workspace_create_handler.py` (unit-level,
  reusing the file's existing `_Stub` harness with a
  `_RaisingActivateRegistry` wrapper) and
  `test_activation_failure_surfaces_inline_but_creates_workspace` in
  `Tests/UI/test_settings_workspaces_category.py` (pilot-level, monkeypatching
  the real registry's `set_active_workspace`).
- Enter-submit fast path: the existing
  `test_double_submit_creates_exactly_one_workspace_no_crash` drives
  `modal._create()` directly, not a real keypress. Added
  `test_enter_key_on_name_input_submits_via_real_keypress`
  (`Tests/Workspaces/test_workspace_create_modal.py`), which asserts
  `AUTO_FOCUS` landed on the name input and drives `pilot.press("enter")`.
- Real TOCTOU `failed_folders`: the existing Qodo-round retry tests
  (`test_folder_binding_failure_keeps_modal_open_and_retries`, `test_escape_
  after_partial_create_returns_partial_result`) use a raising registry
  stub (`_FlakyBindRegistry`), not a real filesystem race. Added
  `test_real_toctou_folder_deleted_before_create_shows_inline_failure`: adds
  a real folder, `shutil.rmtree`s it, presses Create, and asserts the
  modal's real `add_folder_binding()` call raises and the inline-failure
  state (Finding 7) is reached with that folder's path in the error text.
- `_remove_folder` + checkbox-preservation: `_remove_folder`'s stale-error
  behavior was already pinned (`test_remove_folder_clears_stale_error`);
  checkbox-survives-recompose was NOT covered — added
  `test_make_active_checkbox_survives_folder_add_recompose` (real
  `pilot.click` on the checkbox, then a real folder-add recompose).
- Typed seams: `WorkspaceCreateResult.project_skills` was already
  `tuple[ProjectSkillsDiscovery, ...]` (verified, no change needed). The
  three surface callbacks were NOT annotated — added
  `result: WorkspaceCreateResult | None` (or a quoted string form where the
  module lacks `from __future__ import annotations`, i.e.
  `settings_screen.py`) to `ConsoleWorkspaceController._handle_workspace_
  create_result`, `SettingsScreen.handle_workspace_create`'s `_done`, and
  `LibraryScreen.create_local_workspace`'s `_done`, each behind a
  `TYPE_CHECKING`-gated import of `WorkspaceCreateResult` to avoid a real
  import cycle (matching each file's existing house style for such
  type-only imports).
- Description provenance — **controller ruling implemented**:
  `WorkspaceCreateModal.__init__` gained an optional
  `description: str = "Created from the workspace setup dialog."` param,
  threaded into `_create()`'s `create_workspace(...)` call (replacing the
  old hardcoded literal). Each surface now passes its own pre-modal
  wording, recovered from git history
  (`git log --all -p -S"Local workspace created from Console."`):
  Console → `"Local workspace created from Console."`, Settings →
  `"Local workspace created from Settings."` (no historical precedent
  existed for Settings specifically — Settings never had a distinctive
  description pre-modal — so this follows the controller-ruling's own
  explicit wording), Library → `"Local workspace created from Library."`.
  Added `test_surface_description_is_carried_onto_created_record` and
  `test_default_description_used_when_surface_passes_none`.

**Trap hit while writing the description-provenance test:** subclassing
the test file's `_HarnessApp` and overriding `on_mount` to push a modal
with a custom `description` silently failed — Textual's
`MessagePump._on_message` dispatches a message handler by walking the
**whole MRO** (`_get_dispatch_methods`), not just the most-derived
override, so BOTH the subclass's `on_mount` (correct modal, with
description) AND the base class's `on_mount` (default modal, no
description) ran, each pushing its own screen; the base class's push
landed second and became `app.screen`, so the pilot interacted with the
UN-described modal. Fixed by giving `_HarnessApp` itself an optional
`description` constructor param instead of subclassing — a single-class
harness, no MRO surprise. Generalizable lesson: **a plain Python subclass
override of a Textual message handler (`on_mount`, `on_key`, etc.) does
NOT shadow the base class's own handler — both run** (this differs from
ordinary Python method dispatch and is easy to assume away).

**Files touched:** `tldw_chatbook/Widgets/workspace_create_modal.py`,
`tldw_chatbook/UI/Console_Modules/workspace.py`,
`tldw_chatbook/UI/Screens/settings_screen.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/Workspaces/test_workspace_create_modal.py`,
`Tests/Workspaces/test_console_workspace_create_handler.py`,
`Tests/UI/test_settings_workspaces_category.py`.

**Gate:** `Tests/Workspaces/` 279 passed; `Tests/Skills/` 436 passed; the
three named `Tests/UI/` files 29 passed + 1 pre-existing, unrelated
failure (`test_single_item_handoff_gates_on_the_selected_row_not_the_
aggregate`, fails in complete isolation with a shifting root cause across
runs — "Local study backend is unavailable" vs. "Local prompt backend is
unavailable" — in a file this task never touched); repo-wide
`--collect-only` sweep: 52305 tests collected, zero collection errors.
