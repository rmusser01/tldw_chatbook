# Notes and Folder Files Presentation Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Library Notes authority, Folder Files actions, and Session Git reviews easier to understand without changing any storage or Git behavior.

**Architecture:** Keep the existing retained `LibraryScreen` and widget ownership. Add authority presentation at each widget's existing update choke points, reorder existing File Notes actions, and restructure existing Git review projections. Reuse current services and messages; add no event bus, state-machine framework, dependency, or new shortcut.

**Tech Stack:** Python 3.11, Textual 8.x, pytest/Pilot, existing modular TCSS build.

---

## Governance and execution order

ADR required: no new ADR

ADR paths:

- `backlog/decisions/021-file-backed-notes-disk-authority-and-recovery.md`
- `backlog/decisions/029-local-private-data-boundary.md`
- `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
- `backlog/decisions/035-file-notes-session-git-index-controls.md`
- `backlog/decisions/038-file-notes-guarded-session-commit.md`
- `backlog/decisions/039-file-notes-guarded-session-push.md`

Reason: the plan changes information hierarchy, copy, contrast, and progressive disclosure only. Existing authority, recovery, and Git authorization decisions remain unchanged.

Execute in order because all three tasks touch `library_file_notes_workspace.py` or its focused tests:

1. `TASK-19000`
2. `TASK-19001`
3. `TASK-19002`

Before each task: `backlog task edit <id> -a @codex -s "In Progress"`, then add the task's `## Implementation Plan` with a link to this file. Do not mark Done until its own ACs and Definition of Done are complete.

## TASK-19000 — Clarify Notes source authority and persistent status

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Test: `Tests/UI/test_library_file_notes_workspace.py`
- Test: `Tests/UI/test_library_shell.py`
- Test: `Tests/UI/test_library_honesty_accessibility.py`
- Test: `Tests/UI/test_css_build_integrity.py`

- [ ] Write failing source-strip and authority-row tests.

  Extend `test_library_notes_source_choices_render_and_switch_by_keyboard` to require `Library notes` and `Folder files`. Add mounted assertions that `LibraryNotesCanvas.compose()` emits a first-child, `markup=False` authority row in loading, list, editor, create, and legacy-sync modes. Replace `test_files_mode_carries_a_placement_sentence_relating_it_to_sync` with an authority-row contract driven by root, save, and Session Git state.

  Run:

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_file_notes_workspace.py
  ```

  Expected: FAIL because the strip still says `Database | Files` and authority rows are absent or list-only.

- [ ] Implement the smallest authority projection.

  In `LibraryScreen.compose_content()`, change display labels only; retain `LIBRARY_NOTES_SOURCE_DATABASE`, `LIBRARY_NOTES_SOURCE_FILES`, route values, handlers, and focus IDs. In `LibraryNotesCanvas.compose()`, emit one authority `Static` before mode dispatch so the row persists across loading/list/editor/create/legacy sync. Do not add a reactive store or event bus.

  In `LibraryFileNotesWorkspace.compose()`, replace `#file-notes-purpose` with one authority row. Update it only from `_update_root_surface()`, `_set_save_state()`, and `_render_session_git_label()`, which already own the displayed facts.

- [ ] Pin truthful navigation and compact behavior.

  Add tests that currently available operation status is not reset to idle by an in-surface route change; the row must keep text-explicit state and a next action. Do not add persistence or new writes to the legacy engine. Extend the 60x20 Notes matrix in `Tests/UI/test_library_shell.py` to account for the extra row and assert painted text/containment rather than style geometry alone.

- [ ] Fix scoped readable error ink and regenerate CSS.

  Add app-tier selectors for `.file-notes-git-commit-error` and `#file-notes-save-status.-error` using `$ds-status-error-readable`. Preserve the existing disabled-button opacity override. Do not edit generated bundles by hand.

  Run:

  ```bash
  ../../.venv/bin/python tldw_chatbook/css/build_css.py
  ../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py
  ```

  Expected: PASS, including exact bundle parity and theme contrast assertions.

- [ ] Run the task gate and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py
  git diff --check
  ```

  Commit: `feat(library): clarify Notes source authority`

## TASK-19001 — Refine Folder Files target path and action hierarchy

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Modify: `Tests/UI/test_library_file_notes_workspace.py`

- [ ] Write failing path-label and state-matrix tests.

  Require `Target path · New / Move / Save copy` above the input in every editor state. Add a table-driven DOM test for normal, dirty, conflict, error, deleted, protected, and excerpt states. Assert normal primary order `New`, `Move`, `Delete`, `More file actions`; promote `Restore`, `Reload from disk`, and `Save copy` only when their state requires them.

  Run:

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_file_notes_workspace.py -k "field_labels or discloses_actions or maintenance_disclosure"
  ```

  Expected: FAIL against the changing 17-cell label and current flat action rows.

- [ ] Implement layout-only action projection.

  Update `_path_field_label_copy()`, `compose()`, `_update_controls()`, `_sync_editor_action_visibility()`, `_toggle_maintenance_actions()`, and `_sync_editor_action_layout()`. Keep `_move_file()`, `_save_copy()`, `_reload_file()`, delete/recovery handlers, service calls, and message types unchanged. Use the existing maintenance disclosure; add only an explicit text/glyph state if required.

- [ ] Prove focus redirection and compact containment.

  Add tests that hiding a secondary action redirects focus to the disclosure or nearest valid primary action. Render wide and 40-column compact cases with the production widget hierarchy and stylesheet. Assert input/action containment and compositor text.

- [ ] Run the task gate and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_file_notes_workspace.py
  git diff --check
  ```

  Commit: `feat(file-notes): simplify editor actions`

## TASK-19002 — Make Session Git reviews decision-first

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Test: `Tests/UI/test_library_file_notes_git.py`
- Test: `Tests/UI/test_library_file_notes_git_push.py`

- [ ] Write failing decision-fact placement tests.

  Extend `test_commit_review_is_literal_complete_and_discloses_included_notes` and `test_push_review_is_complete_immutable_and_keyboard_safe`. Require visible `What`, `Where`, `Impact`, and `Recovery` sections. Assert exact destination ref, endpoint, candidate/transition, lease, hook behavior, transport/authentication, and object-publication scope are not descendants of the technical `Collapsible`.

  Run:

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_git_push.py -k "review"
  ```

  Expected: FAIL because authorization-changing push facts are currently inside technical details.

- [ ] Reshape presentation projections without changing domain services.

  Extend `CommitPanelReviewProjection` and `PushPanelReviewProjection` only with the already-owned `RepositoryIdentity` facts needed to render the four decision sections. Update `_build_commit_review_projection()` and `_publish_push_settlement()` to pass existing immutable snapshot data. Do not change `Notes/file_notes_git_commit.py`, `Notes/file_notes_git_push.py`, trust resolution, staging, lease, hook, transport, or uncertainty policy.

- [ ] Restrict technical disclosure to audit evidence.

  Keep repository/worktree identity tuples and duplicate/internal evidence under `Show technical details`. Move `Endpoint Details` outside that disclosure while reusing `PushEndpointDetailsDialog` and its existing workspace handler. Preserve default collapse, focus restoration, Edit/Cancel/Confirm order, and safe initial action.

- [ ] Verify 40x20 keyboard behavior and commit.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_git_push.py
  git diff --check
  ```

  Expected: PASS for compact scrolling, disclosure focus, endpoint-details return, and unchanged operation snapshots.

  Commit: `feat(file-notes): make Git reviews decision-first`

## Plan-level verification

- [ ] Run all presentation tests together after the three commits.

  ```bash
  ../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_git_push.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py
  git diff --check
  ```

- [ ] Complete each Backlog task independently with checked ACs, exact evidence, implementation notes, and its ADR check. Do not defer all three closures to the final programme task.
