# Lasting Notes Sync UI and Atomic Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Present the lasting-sync runtime safely, replace the legacy writer atomically, and verify the complete Notes/Files/Sync journey.

**Architecture:** Build inert Textual state/canvases behind a focused `LibraryNotesSyncController` and a narrow structural runtime port first. `LibraryScreen` owns routing and forwards typed messages only. The final cutover is one integration changeset: migrate paused candidates, swap entry points, remove every legacy mutation path, record the cutover marker, then enable reviewed local-root activation. A final evidence task validates production-shaped rendering, accessibility, lifecycle, recovery, and documentation.

**Tech Stack:** Python 3.11, Textual 8.x, pytest/Pilot, existing Notes sync foundation, modular TCSS build, tmux-based isolated TUI verification.

---

## Governance and gates

ADR required: no new ADR

ADR paths:

- `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this plan renders and activates the accepted local-only architecture. Server-backed lasting sync stays disabled until `tldw_server` publishes its separately allocated ADR and versioned capability. No local task or placeholder contract may claim that work.

Dependencies:

```text
TASK-19003 + TASK-19004 + TASK-19005 -> TASK-19010 inert UI
TASK-19000 + TASK-19003 + TASK-19006 + TASK-19007 + TASK-19008 + TASK-19009 + TASK-19010
  -> TASK-19011 atomic cutover
TASK-19001 + TASK-19002 + TASK-19011 -> TASK-19012 final evidence
```

## TASK-19010 — Build lasting sync setup and attention surfaces

**Files:**

- Create: `tldw_chatbook/Library/library_notes_lasting_sync_state.py`
- Create: `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py`
- Create: `tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py`
- Create: `tldw_chatbook/Widgets/Library/library_notes_sync_roots_canvas.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/widget_defaults_scoped.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/Library/test_library_notes_lasting_sync_state.py`
- Create: `Tests/UI/Library_Modules/test_library_notes_sync_controller.py`
- Create: `Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py`
- Create: `Tests/Widgets/Library/test_library_notes_sync_roots_canvas.py`
- Create: `Tests/UI/test_library_notes_lasting_sync_flow.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Modify: `Tests/UI/test_library_shell.py`

- [x] Start TASK-19010 and write RED pure-state tests.

  Cover relationship choice, local/server destination capability, direction, setup validation, checking, reviewed action groups, stale review, activation receipt, root list/status, manual reconciliation, attention choices, pause/resume, retarget, disconnect, pagination, redacted diagnostics, and explicit next actions.

  The pure state must reject global conflict-winner fields and an auto-sync-every-N-minutes field.

- [x] Implement frozen UI projections and a narrow controller port only.

  Define a structural `LastingSyncRuntimePort` in `library_notes_sync_controller.py` with only the `snapshot`, `check_root`, `apply_reviewed`, `activate_root`, `pause_root`, `resume_root`, `retarget_root`, and `disconnect_root` methods the UI needs. `LibraryNotesSyncController` receives that port plus `LibraryNoteImportController`, owns chooser/root-review task state, and publishes immutable snapshots. Translate public sync models into bounded presentation rows. Keep absolute paths bounded to the explicit root-detail surface; ordinary status uses display name and opaque IDs. Do not import the concrete runtime, private store, filesystem, coordinator, executor, legacy engine, or ChaChaNotes. Do not add a generic controller base class.

- [x] Write RED canvas/message tests for all phases.

  `library_notes_add_from_files_canvas.py` is the single authoritative relationship chooser and lasting setup surface. It renders:

  ```text
  choose relationship -> configure -> checking -> review -> activating -> receipt
  ```

  `library_notes_sync_roots_canvas.py` renders decorated roots, contextual root actions, manual `Check changes`, attention review, recovery, pause/resume, retarget, and disconnect. Test physical button messages, safe initial focus, disabled reasons, text/glyph states, bracket-safe copy, paging, and 60x20 compositor containment.

- [x] Keep server-backed setup visibly unavailable.

  Render `Unavailable - server sync-folder capability not installed`. The disabled control has readable contrast and the nearest valid action. Do not add a server adapter, feature-flag fallback, fake capability, claim token, or flat server write.

- [x] Integrate the inert UI behind an explicit availability gate.

  Add route/view projection and forward typed child messages to `LibraryNotesSyncController`. Until TASK-19011, the existing toolbar still shows legacy `Sync` and `Import`; expose the new surfaces only through a test-only/inert gate or direct widget mounting. The production `Keep a folder synced` path must remain unavailable and cannot activate roots. The chooser's `Import once` branch hands off to the existing TASK-19003 `LibraryNoteImportController`; do not implement a second import workflow or chooser.

- [x] Prove manual and attention semantics against a fake runtime.

  Test exact safe/attention/skip counts, reviewed observation token, stale review -> `Check again`, note-implied filesystem move preview, Keep file/note/both, bounded deletion choices, partial/recovery actions, retarget no-deletion inference, and disconnect no-delete copy. Canvas code emits messages only; fake runtime records calls.

- [x] Regenerate CSS, run, and commit.

  ```bash
  ../../.venv/bin/python tldw_chatbook/css/build_css.py
  ../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Library/test_library_notes_lasting_sync_state.py Tests/UI/Library_Modules/test_library_notes_sync_controller.py Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py Tests/Widgets/Library/test_library_notes_sync_roots_canvas.py Tests/UI/test_library_notes_lasting_sync_flow.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_shell.py Tests/UI/test_css_build_integrity.py
  git diff --check
  ```

  Commit: `feat(notes): build inert lasting-sync surfaces`

## TASK-19011 — Cut over atomically from legacy to lasting Notes sync

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py`
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py`
- Delete: `tldw_chatbook/Notes/sync_engine.py`
- Delete: `tldw_chatbook/Notes/sync_service.py`
- Delete: `tldw_chatbook/Library/library_notes_sync_state.py`
- Modify: `Docs/Features/notes_bidirectional_sync.md`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `Docs/superpowers/specs/2026-08-12-notes-folder-import-sync-design.md`
- Modify: `Docs/superpowers/specs/2026-08-19-library-notes-files-reviewed-sync-redesign-design.md`
- Create: `Tests/Notes/test_notes_sync_cutover.py`
- Modify: `Tests/UI/test_library_notes_lasting_sync_flow.py`
- Modify: `Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py`
- Modify: `Tests/Library/test_library_local_rag_search_service.py`
- Modify: `Tests/test_remaining_diagnostic_sentinel_matrix.py`
- Modify/delete legacy expectations: `Tests/Library/test_library_notes_sync_state.py`
- Modify/delete legacy expectations: `Tests/Notes/test_sync_engine.py`
- Modify/delete legacy expectations: `Tests/Notes/test_library_notes_sync_integration.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Modify: `Tests/UI/test_library_canvas_scoped_sync.py`
- Modify: `Tests/UI/test_library_shell.py`

- [x] Write the dual-owner prevention tests before production edits.

  In `Tests/Notes/test_notes_sync_cutover.py`, add source/AST/runtime guards proving:

  - no production import or construction of `NotesSyncEngine` or legacy `NotesSyncService`;
  - no `_library_notes_sync_*` timer, worker group, or mutating handler;
  - no writes to legacy sync config keys;
  - only `notes_sync_legacy.py` reads legacy sync metadata/config;
  - lasting runtime activation is impossible until the cutover marker exists and no already-running profile process is reported;
  - unavailable lasting runtime never falls back to legacy mutation.

  Expected: FAIL against current production wiring.

- [x] Add an explicit cutover startup barrier.

  Cutover occurs across a normal application restart; do not invent a hot-swap controller between screen-owned legacy code and the app-owned runtime. At startup of the cutover release:

  1. production code contains no constructor or admission path for the legacy writer;
  2. incomplete legacy evidence is read only into paused candidates, never resumed as a mutation;
  3. candidate migration completes and records its source fingerprint;
  4. a private cutover marker is recorded;
  5. only after legacy production paths are removed, change `tldw_chatbook/app.py` from TASK-19009's temporary private `cutover_admitted=False` to the code-owned `cutover_admitted=True`, keep the builder argument required, and prove that exact production lifecycle wiring in `Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py`;
  6. reviewed local-root activation becomes eligible only when both cutover gates pass and `_instance_lock_status` does not report another already-running profile process.

  If another process is already open, show `Close the other Chatbook process and restart before activating folder sync`; do not activate. The current app cannot retroactively fence an older binary launched after it, so documentation and tests must scope the no-dual-owner guarantee to production paths in the cutover release and require old versions to be closed before activation. The marker is not set and activation stays disabled on migration failure. Do not run both engines during a comparison window.

- [x] Swap toolbar and retained canvas entry points in one changeset.

  Replace adjacent legacy `Sync` and `Import` with one `Add from files…`; add `Manage sync folders` only when roots/candidates exist. The one TASK-19010 chooser routes `Import once` into TASK-19003's controller and `Keep a folder synced` into lasting setup. Decorate top-level root nodes with text-explicit state and contextual actions.

- [x] Remove every legacy mutation path.

  Delete the legacy engine/service/state modules. Before deleting `library_notes_sync_state.py`, remove its unrelated `count_noun` dependency from `library_local_rag_search_service.py`: use the direct singular/plural expression at the one conversation-row call site and preserve the existing exact-output cases in `test_library_local_rag_search_service.py`; do not create a new shared utility for one use. Remove sync fields, timer, config loading/writes, `_compose_sync`, focus roles, handlers, auto tick, run worker, and legacy panel CSS. Retain legacy config keys, note columns, `sync_sessions`, and `sync_conflicts` as read-only compatibility inputs; do not drop schema or history in this task. Retarget the `notes_sync` entry in `Tests/test_remaining_diagnostic_sentinel_matrix.py` from the deleted legacy engine to `tldw_chatbook.Notes.notes_sync_runtime` so diagnostic redaction remains owned by the live runtime.

- [x] Test fail-closed startup, migration review, and activation ordering.

  Cover clean install, legacy config only, incomplete legacy evidence, migration failure, paused candidates, missing cutover marker, another already-running profile process, no runtime backend, passive root lease, offline root, and successful reviewed activation. Prove the UI shows `Review migration` and requires a current dry-run; it never honors legacy conflict winners or auto-sync.

- [x] Update documentation and run the atomic gate.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_notes_sync_cutover.py Tests/Notes/test_notes_sync_runtime.py Tests/Notes/test_notes_sync_legacy_migration.py Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py Tests/Library/test_library_local_rag_search_service.py Tests/test_remaining_diagnostic_sentinel_matrix.py Tests/UI/test_library_notes_lasting_sync_flow.py Tests/UI/test_library_canvas_scoped_sync.py Tests/UI/test_library_shell.py Tests/Widgets/Library/test_library_notes_canvas.py
  rg -n "NotesSyncEngine|NotesSyncService|_library_notes_auto_sync_timer|handle_library_notes_sync" tldw_chatbook
  git diff --check
  ```

  Expected grep result: references only in explicitly retained migration/history documentation or none in production Python; the AST test is authoritative.

- [x] Commit and close only after the no-dual-owner gate passes.

  Commit: `feat(notes): cut over to lasting folder sync`

  Check every TASK-19011 AC, record exact cutover/lifecycle evidence, link ADR-059/073, and set Done. If any legacy path remains reachable, leave the task In Progress.

## TASK-19012 — Verify the reviewed Notes, Files, and Sync journey

**Files:**

- Create: `Tests/UI/test_library_notes_files_sync_journey.py`
- Create: `Helper_Scripts/verify_notes_files_sync_tui.py`
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `Docs/Features/notes_bidirectional_sync.md`
- Modify: relevant text captures/screenshots only if they can be regenerated from the isolated profile
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if a real new incident warrants a lesson

- [x] Write one production-shaped journey matrix.

  Mount the real `LibraryScreen` hierarchy and exact `TldwCli.CSS_PATH`. Cover Library notes authority, Folder files authority/actions, Import once, lasting setup/review/attention/recovery, and Session Git. Render wide and 60x20 Notes cases; render Folder Files/Session Git at their supported 40x20 alternate navigator/editor layout.

- [x] Assert pixels/compositor text and focus, not style properties alone.

  Capture `app.export_screenshot()` SVG or compositor strips. Check labels, authority, non-ready next actions, disabled/error contrast, disclosure containment, scroll owner, Escape behavior, focus restoration, and truthful footer hints. Add representative theme measurements for disabled and error text.

- [x] Prove lifecycle and recovery across restart with isolated files.

  Use temporary config and data directories. Create an incomplete journal in the scratch private store, start the real app runtime, and assert it becomes resumed or Needs attention before watcher admission. Confirm the decoy/default config and real user data paths remain byte-identical.

- [x] Add a safe live-TUI helper and run it.

  `Helper_Scripts/verify_notes_files_sync_tui.py` creates its own temporary directory, writes an explicit scratch `TLDW_CONFIG_PATH` whose `[paths].data_dir` is also scratch, disables model-catalog networking, seeds only fixture notes/files/Git repositories, launches the app under a unique tmux socket, captures frames, and tears down. It must never rely on the caller's real profile or run a schema migration against it.

  Run:

  ```bash
  ../../.venv/bin/python Helper_Scripts/verify_notes_files_sync_tui.py
  ```

  Expected: a bounded evidence directory path, checksums showing decoy/default files unchanged, and PASS summaries for the reviewed journeys. Inspect the captured frames top-to-bottom before accepting the result.

- [x] Run the broad programme gate.

  ```bash
  ../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_execution_models.py Tests/Notes/test_note_import_receipts.py Tests/Notes/test_note_import_executor.py Tests/Notes/test_note_import_planner.py Tests/Notes/test_notes_device_state_store.py Tests/Notes/test_notes_sync_models.py Tests/Notes/test_notes_sync_filesystem.py Tests/Notes/test_notes_sync_reconciler.py Tests/Notes/test_notes_sync_coordinator.py Tests/Notes/test_notes_sync_executor.py Tests/Notes/test_notes_sync_watcher.py Tests/Notes/test_notes_sync_runtime.py Tests/Notes/test_notes_sync_legacy_migration.py Tests/Notes/test_notes_sync_cutover.py Tests/UI/test_library_notes_files_sync_journey.py Tests/UI/test_library_note_import_flow.py Tests/UI/test_library_notes_lasting_sync_flow.py Tests/UI/test_library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py Tests/UI/test_library_file_notes_git_push.py Tests/UI/test_library_shell.py Tests/UI/test_library_honesty_accessibility.py Tests/UI/test_css_build_integrity.py Tests/ProductionApp/test_notes_sync_runtime_lifecycle.py
  ../../.venv/bin/python -m compileall -q tldw_chatbook
  git diff --check
  ```

- [x] Reproduce any inherited failures on the untouched base before documenting them.

  A green focused run is not evidence for an unreachable product flow; a failing broad run is not automatically caused by this branch. Record exact commands, counts, and base comparison in TASK-19012 Implementation Notes.

- [x] Close the programme.

  Update guides and design status, check every TASK-19012 AC, add exact automated/live evidence and the ADR-059/073 check, and set TASK-19012 Done. Audit TASK-19000 through TASK-19012 status from the current branch; do not claim the programme complete while any prerequisite remains open.

  Commit: `test(notes): verify reviewed sync journey`
