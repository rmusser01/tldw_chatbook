---
id: TASK-564
title: Fix three pre-existing personas/library test failures
status: In Progress
assignee: []
created_date: '2026-07-24 23:34'
updated_date: '2026-07-25 09:30'
labels:
  - tests
  - library
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three test failures surfaced by the image-gen P3 verification sweeps (2026-07-24) and attributed as PRE-EXISTING via a throwaway worktree at the P3 plan base (`133330366` — they fail identically without any P3 changes): a library-scale notify-signature mismatch, a workbench hidden-directory export assertion, and a library footer-hint text drift. They are unrelated to image generation; filing so the baseline stops accreting known-red tests that every branch must re-attribute.

First step per the P3 final review: re-confirm each still reproduces on current `origin/dev` before fixing, so the attribution survives review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each failure is re-confirmed on current `origin/dev` (or closed as already-fixed) with the failing test name + one-line root cause recorded in the Implementation Notes.
- [ ] #2 The genuine defects (product code or stale test expectation — determine which per case) are fixed; all three tests pass.
- [ ] #3 The relevant suites run green with no new failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-run Tests/UI/test_personas_library_scale.py + test_personas_workbench.py on current origin/dev to get the real current failure set (not the stale 133330366 baseline description).
2. git log -p --follow each failing test + its product call site to find which side drifted and in which commit.
3. Fix genuine product regressions at their root; update tests only where the product change was deliberate and documented.
4. Re-run the two files plus any suite touching the changed product code (Utils/path_validation.py callers) to confirm green + no collateral breakage.
5. For any fix that would require editing tldw_chatbook/UI/Screens/personas_screen.py or personas_character_editor_widget.py (owned by open PR #865, claude/image-gen-followups-personas), do not edit those files — record the root cause and the exact minimal fix for a later PR instead.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-ran Tests/UI/test_personas_library_scale.py + test_personas_workbench.py on current origin/dev (99d90f0a5): exactly 3 failures, independently corroborated by PR #865's own regression-sweep notes (task-563 doc) naming these same 3 tests as task-564's scope.

1. test_import_offpage_name_conflict_message (test_personas_library_scale.py) — TEST drifted stale. Root cause: task-445 (commit 5eb878cfda, "roleplay polish sundries") deliberately added a keyword-only `timeout=6.0` to the two import-notify `_notify()` calls in personas_screen.py (so the toast lingers past the simultaneous card-view swap), but did not update this test's `monkeypatch.setattr(screen, "_notify", lambda msg, sev="warning": ...)` stub, which lacked a `timeout` parameter and raised TypeError when the real call site passed it. Fixed the TEST: the fake now accepts `timeout=None` too, matching the real `_notify(message, severity="warning", *, timeout=None)` signature. No product change.

2. test_export_json_rejects_hidden_directory_destination (test_personas_workbench.py) — genuine PRODUCT bug, present since inception (PR #509 review-feedback commit ac995f72b, 2026-06-11), never actually enforced. Root cause: `personas_screen.py._write_text_file` (and `_export_character_png_sync`) call `validate_path(target, base_directory=target.parent)` to allow an arbitrary user-chosen destination directory while still using validate_path for traversal/symlink checks. Because `validate_path`'s hidden-file check only inspects path parts *relative to* base_directory, and base_directory here IS the (possibly hidden) destination directory itself, a hidden immediate-parent directory (e.g. `.sneaky/out.json`) was silently folded into the excluded base and never flagged — writes into dot-directories always succeeded. Fixed the PRODUCT at its root in `tldw_chatbook/Utils/path_validation.py` (`validate_path`): added a check that also rejects when `base_directory`'s own final path component is dotted, independent of the existing relative-parts check. This does not affect the documented `~/.local/share/...`-ancestor exception (only base_directory's own name is checked, not its ancestors). Verified no other current `validate_path` caller (Tools/file_operation_tools.py sandbox root, OCR_Backends.py, Chat_Dictionary_Lib.py, Character_Chat_Lib.py avatar/chat-log/export paths, Prompts_Interop.py, rag_service.py, Home/active_work_adapter.py — which has the identical `base_directory = path.parent` pattern and is fixed as a side effect) uses a dotted base_directory basename; ran their test suites, all green.
   Fixed in Utils/path_validation.py rather than personas_screen.py specifically to avoid the file-ownership conflict noted below (this is also the more correct fix location: the same `base_directory=target.parent` bug pattern exists in Home/active_work_adapter.py too, so fixing the shared validator closes it everywhere at once).

3. test_character_book_errors_render_in_editor_footer (test_personas_workbench.py) — genuine PRODUCT regression, BLOCKED, not fixed. Root cause: commit 414183488 ("Fix character card import: implement stub loader, loosen V2 validation", merged to dev the same day, after the P3 baseline) changed `validate_character_book()` in Character_Chat_Lib.py to only return `ok=False` when `book_data` itself is not a dict — entries-type problems now return `ok=True` with warnings in the second tuple element (a deliberate leniency change scoped entirely to Character_Chat_Lib.py + its own test file; no other files touched). `personas_screen.py._validate_character()` (line ~6365-6369) still gates on `if not ok:` before extending its `errors` list with `book_errors`, so it silently stopped surfacing/blocking on malformed `character_book.entries` — a downstream contract-drift regression the loosening commit's author was not exercising (its own test run didn't touch Tests/UI/). The minimal, faithful fix is a one-line change in personas_screen.py: gate on `if book_errors:` instead of `if not ok:` (restores exactly the pre-regression, still-desired behavior: the editor footer surfaces the "entries should be a list" message and blocks Save, per `_validate_character`'s own docstring "failures block Save").
   BLOCKED: this fix requires editing tldw_chatbook/UI/Screens/personas_screen.py, which is under active, substantial, concurrent modification by open PR #865 (claude/image-gen-followups-personas, task-563, 181/8 lines changed in that file). Per this task's explicit repo-trap instructions, edits to that file were out of scope here; PR #865's own regression-sweep notes confirm it deliberately left this test's failure "not touched here, task-564's scope." Confirmed via `gh pr diff 865` that PR #865's hunks in personas_screen.py (old-line ranges 4694-4903, 5151-5236, 6496-6501) do not overlap `_validate_character`/`_handle_save_requested` (~6358-6403), so the 1-line fix above is low-conflict-risk once file ownership frees up (PR #865 merges, or an explicit exception is granted).

Verification: Tests/UI/test_personas_library_scale.py (8 passed). Tests/UI/test_personas_workbench.py (202 passed, 1 failed — the blocked #3, no other regressions; was 208 passed/3 failed before). Tests/ -k path_validation (25 passed, 1 skipped). Tests/Tools/test_file_tool_sandbox.py + Tests/integration/test_file_operations_with_validation.py + Tests/Home/test_active_work_adapter.py + Tests/Character_Chat/{test_character_file_operations,test_character_image_upload,test_character_export_no_image,test_character_dictionaries_portability}.py (94 passed). ruff check clean on touched files. `python -c "import tldw_chatbook.app"` clean.

Modified files: Tests/UI/test_personas_library_scale.py (test-side notify-stub fix), tldw_chatbook/Utils/path_validation.py (product-side hidden-base-directory fix).

Follow-up needed: a small, low-risk PR (once personas_screen.py is free of PR #865's conflict) applying the one-line `_validate_character` fix in item 3 above, to close out this task fully.
<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:PLAN:END -->
