# Notes Import Once UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the existing immutable one-time Notes import planner and durable executor through a safe, production Library workflow while legacy Sync remains unchanged.

**Architecture:** A focused `LibraryNoteImportController` in `UI/Library_Modules` owns selection, planning, worker admission, cancellation, execution, and durable same-session workflow state through named late-bound dependencies. `LibraryScreen` owns the retained route and forwards typed child messages. A pure state module projects planner/executor models into bounded pages, and a child canvas renders phases. Existing TASK-16230/16309 planner, approval, executor, target, and receipt contracts remain authoritative.

**Tech Stack:** Python 3.11, Textual 8.x, SQLite, pytest/Pilot, existing Textual file picker and Notes services.

---

## Governance and scope

Backlog task: `TASK-19003`

Dependencies: `TASK-16230`, `TASK-16309`

ADR required: no new ADR

ADR paths:

- `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`
- `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

Reason: this plan exposes the already-accepted one-time planner/executor and adds a genuinely read-only pre-approval receipt lookup. It does not create lasting roots or server authority.

Explicit limits:

- Receipt revisit means the current application session; the existing ledger has no restart-safe session-list API or source-payload reconstruction.
- Cancellation reports partial completion; it does not promise rollback.
- Import replacement has receipt/retry semantics only; do not advertise Undo because TASK-16309 exposes no recovery-content API.
- The route is local Library Notes only. File Notes and server Notes fail closed.
- `#library-notes-sync-open`, its handler, legacy panel, timer, and behavior remain intact.

## TASK-19003 — Ship the Import once Notes workflow

**Files:**

- Create: `tldw_chatbook/Library/library_note_import_state.py`
- Create: `tldw_chatbook/UI/Library_Modules/library_note_import_controller.py`
- Create: `tldw_chatbook/Widgets/Library/library_note_import_canvas.py`
- Modify: `tldw_chatbook/Widgets/Library/__init__.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Notes/note_import_receipts.py`
- Modify: `Docs/User_Guide/library/notes.md`
- Create: `Tests/Library/test_library_note_import_state.py`
- Create: `Tests/UI/Library_Modules/test_library_note_import_controller.py`
- Create: `Tests/Widgets/Library/test_library_note_import_canvas.py`
- Create: `Tests/UI/test_library_note_import_flow.py`
- Modify: `Tests/UI/test_library_shell.py`
- Modify: `Tests/UI/test_library_canvas_scoped_sync.py`
- Modify: `Tests/UI/test_library_modal_dismissal.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`
- Regression: `Tests/Notes/test_note_import_execution_models.py`
- Regression: `Tests/Notes/test_note_import_receipts.py`
- Regression: `Tests/Notes/test_note_import_executor.py`
- Regression: `Tests/Notes/test_note_import_planner.py`
- Regression: `Tests/Notes/test_note_import_windows_fs.py`

- [ ] Start TASK-19003 and add its implementation plan.

  ```bash
  backlog task edit 19003 -a @codex -s "In Progress"
  backlog task edit 19003 --plan "Follow Docs/superpowers/plans/2026-08-20-notes-import-once-ui.md task-by-task; preserve legacy Sync and ADR-059/073."
  backlog task 19003 --plain
  ```

- [ ] Write failing pure workflow-state tests.

  In `Tests/Library/test_library_note_import_state.py`, define the required phases:

  ```text
  select -> destination -> checking -> review -> importing -> receipt
  ```

  Test frozen/redacted snapshots, bounded paging, selected-path accumulation, file-versus-folder exclusivity, required file destination segments, collision gates, uncertain confirmation, per-item override authority, stale approval invalidation, progress reduction, cancellation, partial receipt, retry gates, and same-session receipt revisit.

  Run:

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Library/test_library_note_import_state.py
  ```

  Expected: FAIL because the module does not exist.

- [ ] Implement the minimal pure state projection.

  Create frozen dataclasses and pure reducer/helpers only. Import the public types from `note_import_plan_models.py` and `note_import_execution_models.py`; do not duplicate classification, action, collision, receipt, or approval enums. Page preview items rather than mounting up to the planner ceiling. Ensure `repr` and diagnostics exclude paths, note contents, keywords, and exception text.

- [ ] Write a failing no-mutation prior-observation test.

  Add a test to `Tests/Notes/test_note_import_receipts.py` that calls a new read-only observation method when the database path is missing and when a SQLite file exists without the receipt schema. Assert the file is not created, schema/user version is unchanged, and the result is empty or a bounded typed unavailable result.

  Expected: FAIL because `prior_observations_for_plan()` currently initializes schema inside a transaction.

- [ ] Add the narrow read-only receipt lookup.

  Add `prior_observations_for_plan_read_only(plan)` to `NoteImportReceiptRepository`. Open an existing database in SQLite URI `mode=ro`; never call `_initialize_schema()` or the write transaction helper. Treat missing file/table as no prior observations and propagate corrupt/newer-schema conditions as bounded planning failures. Keep `prior_observations_for_plan()` unchanged for executor compatibility.

  Run:

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_receipts.py Tests/Notes/test_note_import_planner.py
  ```

- [ ] Write failing canvas rendering and physical-message tests.

  In `Tests/Widgets/Library/test_library_note_import_canvas.py`, cover accumulated file selection with `Add another file`, one-folder selection, destination input, Checking, grouped Review, collision controls, uncertain confirmation, independent replace/membership choices, Importing progress/cancel, Receipt/retry, bracket-safe filenames, disabled reasons, plain-text/glyph states, scrollability, and a 60-column compositor capture. The shared `Import once` versus `Keep a folder synced` chooser belongs only to TASK-19010/TASK-19011.

- [ ] Implement the render/message-only child canvas.

  Follow `library_export_canvas.py` and `library_ingest_canvas.py`. The canvas receives one immutable snapshot and posts typed messages; it does not call the planner, SQLite, Notes services, or filesystem. Reuse Button/Static/Input/VerticalScroll patterns; do not introduce `Select`, a generic workflow framework, or new keybindings.

- [ ] Write failing retained-shell route tests.

  In `Tests/UI/test_library_note_import_flow.py` and existing shell tests, drive:

  1. `#library-notes-import` -> existing `FileOpen(offer_select_folder=True)`.
  2. In the callback, distinguish the returned `Path` with `is_dir()`; a folder ends selection, while a file may be followed by `Add another file`.
  3. Supply destination segments for file selections only.
  4. Check -> review -> approve -> importing -> receipt.

  Assert zero note/folder/receipt/config/private-schema mutation through review cancellation. Use a file-backed `CharactersRAGDB`; do not use `:memory:` across `execute_async()` threads. Assert the exact approved object reaches `NoteImportExecutor` and any change after approval discards that authority.

- [ ] Implement the focused controller with named late-bound dependencies.

  `LibraryNoteImportController` receives callables/objects for picker callbacks, `CharactersRAGDB`, `LocalNoteFolderRepository`, receipt path/repository, planner transforms, executor construction, UI snapshot publication, and post-settlement refresh. It owns the workflow state and task handles; it exposes typed methods such as `begin_selection`, `accept_selected_path`, `check`, `approve_and_execute`, `cancel`, `retry_failed`, and `snapshot`. Tests inject fakes without mounting Textual. Do not add a generic controller base class.

- [ ] Replace the legacy immediate-import handler with controller delegation.

  In `library_screen.py`, replace `_import_library_note_from_path()` and `_fail_library_note_import()` routing with dedicated import workflow state. Use:

  - `discover_import_sources()`;
  - `parse_import_sources()`;
  - `classify_import_batch()`;
  - `prior_observations_for_plan_read_only()`;
  - a second `classify_import_batch()` with observations;
  - paged `LocalNoteFolderRepository.list_children()` for collision names;
  - `analyze_root_collision()` / explicit resolution;
  - `approve_note_import_plan()` only after final review;
  - `NoteImportExecutor.execute_async()` for first execution;
  - `asyncio.to_thread()` plus loop-marshalled progress for `retry_failed()`.

  Repeated file selection uses the existing `FileOpen(offer_select_folder=True)` one selection at a time; do not build a new multi-file picker. Its callback returns either a file or the viewed folder, and the controller branches on `Path.is_dir()`. A folder selection terminates accumulation. Destination segments remain in workflow state and are not persisted before approval.

- [ ] Preserve retained canvas, focus, footer, and hidden completion.

  Add import values to `_library_notes_view` and update `_library_notes_canvas_kwargs()`, `_library_notes_focus_region()`, `_library_notes_semantic_role()`, `_library_notes_scroll_owner()`, `_library_notes_role_target()`, `_library_notes_fallback_focus_target()`, `_library_notes_footer_shortcuts()`, `action_library_notes_escape()`, and `_select_library_rail_row_after_source_admission()`. These methods delegate behavior to `LibraryNoteImportController`; do not put planner/executor branches back into `library_screen.py`.

  Completion updates screen-owned progress/receipt even when the canvas is hidden. Only DOM synchronization is route-gated. Navigation during admitted mutation follows the existing operation fence; navigation after settlement can revisit the latest receipt.

- [ ] Prove refresh, cancellation, retry, and legacy Sync coexistence.

  After settlement, call `_refresh_local_source_snapshot()` and `_request_library_notes_tree_refresh(refresh_root=True)`. Add tests for cancellation before approval, cooperative cancellation during execution, partial receipt counts, retry of failed items, stale plan/collision failure, route changes, and missing local targets. Extend `Tests/UI/test_library_canvas_scoped_sync.py` to prove Review -> Importing -> Receipt retains the same rail/canvas and then prove legacy Sync still opens, runs, and exits normally.

- [ ] Run focused and established backend gates.

  ```bash
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Library/test_library_note_import_state.py Tests/UI/Library_Modules/test_library_note_import_controller.py Tests/Widgets/Library/test_library_note_import_canvas.py Tests/UI/test_library_note_import_flow.py Tests/UI/test_library_canvas_scoped_sync.py Tests/UI/test_library_modal_dismissal.py Tests/Widgets/Library/test_library_notes_canvas.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/Notes/test_note_import_execution_models.py Tests/Notes/test_note_import_receipts.py Tests/Notes/test_note_import_executor.py Tests/Notes/test_note_import_planner.py Tests/Notes/test_note_import_windows_fs.py Tests/Notes/test_note_folder_models.py Tests/Notes/test_note_folder_repository.py Tests/Notes/test_notes_scope_service.py Tests/Notes/test_notes_scope_service_folders.py Tests/DB/test_private_sqlite.py Tests/DB/test_private_sqlite_inventory.py
  ../../.venv/bin/python -B -m pytest -q -p no:cacheprovider -o addopts="" Tests/UI/test_library_shell.py
  git diff --check
  ```

- [ ] Update the guide, commit, and close TASK-19003 only after all gates pass.

  Document `Import once`, one-or-more-file accumulation, one-folder selection, destination semantics, review classifications, partial cancellation, retry, same-session receipt revisit, and the continuing separate legacy Sync entry.

  Commit: `feat(notes): ship reviewed import-once workflow`

  Then check every AC, add exact Implementation Notes/evidence, confirm ADR-059/073, and set TASK-19003 to Done.
