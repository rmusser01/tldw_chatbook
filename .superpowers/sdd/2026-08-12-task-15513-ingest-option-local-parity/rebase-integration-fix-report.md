# TASK-15513 rebase integration fix report

## Scope

Fixed all findings from `rebase-integration-fix-brief.md` without changing task status or acceptance criteria.

## Changes

- Backend selection now uses `_sync_library_canvas(self, "ingest")`, so only the ingest canvas recomposes; the Library shell, rail, and footer retain identity.
- Added `capabilities_for_backend()` as the shared backend-filtering seam. Both canvas composition and in-place collapsed-title receipts use it, so retained Server-only `keep_original_file` cannot appear after switching to Local and editing a textarea.
- Updated stale generic-option snapshot assertions to include the TASK-15513 declared defaults.
- Updated all evidence-supported execution checkboxes in the TASK-15513 plan, with explicit annotations for broad-suite network-guard limits, compositor verification substitution, and mutation coverage.

## TDD evidence

- RED: `test_ingest_backend_switch_recomposes_only_the_ingest_canvas` failed with a recorded `LibraryScreen.refresh(recompose=True)` call.
- RED: `test_local_prompt_receipt_hides_retained_server_only_keep_original_file` failed because the Local collapsed receipt included `Keep original file`.
- GREEN: both tests pass after the shared canvas-sync/capability-filter changes.
- Mutation checks: changing the switch back to full-screen refresh red the identity test; changing the receipt back to the raw capability schema red the Local receipt test. Both mutations were restored.

## Verification

- Passed: two new RED/GREEN regression tests using worktree-local `--basetemp`.
- Passed: `Tests/Library/test_server_ingest_request.py` (40 tests).
- Passed: `pytest -m allow_network Tests/UI/test_library_canvas_scoped_sync.py Tests/UI/test_library_ingest_canvas.py` (10 selected tests; 126 deselected).
- Passed: focused snapshot/backend switch set before broader test selection.
- Passed: `git diff --check`.
- Blocked/annotated: combined UI/App/field-contract commands encounter the repository's Windows Proactor `socket.socketpair()` interaction with the no-network guard after an unmarked async test. This is unrelated to this wave; targeted `allow_network` UI pilots are green.
- Ruff was run through the installed `C:\Python312\Scripts\ruff.exe`; it reports seven existing E721 violations in untouched portions of `library_screen.py`. The project venv has no `ruff` module.

## Review

No new architectural decision is required: this is a narrow correction that reuses the established canvas-scoped sync architecture and existing capability visibility model.
