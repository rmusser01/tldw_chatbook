---
id: TASK-38700
title: "File Notes: push 'needs attention' lives only in the workspace, so a rehydrate or teardown can rewrite it"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: 2026-11-04 01:35
labels:
- follow-up
- library
priority: low
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_git_push.py
dependencies:
- TASK-36527
- TASK-38648
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36527's review, PR #3377, 2026-11-03). Checked by reading the code on dev 9bbd5c2515; no test reproduces it yet.

Two push failures set `_push_phase = "needs_attention"` in the workspace and leave no record in the session owner:
- `_publish_push_review_admission_failure` (library_file_notes_workspace.py:4414-4433). It also clears `_push_operation` when the view is the list.
- `_publish_push_observer_failure` (:4160-4189).

A later Git rehydrate re-derives the phase from the owner snapshot in `_rehydrate_push_state` (:4691-4848). TASK-38648 showed that the 10 s poll tick runs one whenever an owed `refresh_files` is pending.
- After the admission failure, the branch for no retained operation sets the phase to `idle`, because `snapshot.push_recovery` is None (:4733-4736). It keeps the "Push review unavailable" result view, so the Session Git label and the panel disagree.
- After an observer failure on an unsettled operation, the "same operation" branch resets the phase and view to `checking` or `pushing` (:4774-4783). That replaces the "needs attention" copy.

**Lifecycle.** The observer task is never cancelled, either in `on_unmount` (:2662) or in `shutdown()`. That may be by design ("one process-lifetime observer", :4083). But its `except (asyncio.CancelledError, Exception)` (:4141) turns a loop-teardown cancellation into the same "needs attention" publication, and swallows the cancellation.

Depends on TASK-38648 (in PR #3367).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a push review admission failure and after a push observer failure, a Git rehydrate leaves the Session Git label and the push panel showing the same state the failure published, or both cleared together; a test drives each through a real rehydrate and fails on the base build
- [x] #2 When the push observer task is cancelled (for example at loop teardown), it publishes no 'needs attention' result and the cancellation propagates; pinned by a test
<!-- AC:END -->

## Implementation Plan

1. Tests first, in `Tests/UI/test_library_file_notes_git_push.py`, through the poll's real path: record an owner change, then `await workspace.refresh_files()`, which reaches `_rehydrate_push_state`.
   - A review admission failure from the list.
   - An observer that raises while its local-proof operation is still unsettled.
   - Each asserts the Session Git label and the push panel still show the published attention.
   - A third test cancels a live observer and expects `CancelledError` with nothing published.
   - Show all red.
2. Mark the attention projection that only the workspace knows: the review-admission and observer-failure publishers record it. While that exact projection is the shown result, rehydrate keeps `needs_attention` in the no-operation branch and does not reset the same unsettled operation to checking/pushing. Back, a new operation, or a binding/service change replaces the projection, which ends the mark.
3. The observer re-raises `CancelledError` and publishes only for `Exception`.
4. Run the new tests green, plus the push/git/workspace files, comparing failures against base.

## Implementation Notes

**Approach.** The two workspace-only attention publishers, `_publish_push_review_admission_failure` and `_publish_push_observer_failure`, now go through `_set_push_attention`. It records the exact projection it shows. `_push_attention_shown()` is true while that same projection is the panel's result.
- `_rehydrate_push_state` keeps `needs_attention` in its no-operation branch while it is shown. It also skips the "same unsettled operation" reset to checking/pushing, which nothing would observe again.
- Back, a new operation, or a binding/service change replaces the projection, and that ends the mark. So the label and the panel move together: both still show the published attention, or both clear.
- I did not add an owner record. That would put workspace-only copy into `FileNotesSessionOwner` for no gain over the identity mark.

**Observer cancellation.** The observer now re-raises `CancelledError` when its own task is being cancelled (`asyncio.current_task().cancelling()`), and publishes nothing.
- A `CancelledError` that comes from a cancelled settlement (the shielded `operation.wait()`) is still published as an observation failure, as before.
- A blanket re-raise broke `test_workspace_accepted_push_cancel_ignores_late_local_proof[cancelled]` (the awaited observer ended cancelled), which is why the check exists. The existing `cancelling()` entries in `lessons-testing-evidence.md` already cover this distinction, so I added no new lesson.
- I left the observer uncancelled in `on_unmount`/`shutdown()`, as the description allowed.

**Tests** (in `Tests/UI/test_library_file_notes_git_push.py`, right after the observer-exception test). Each rehydrate goes through the poll's path: an owner change, then `await workspace.refresh_files()`, with a spy asserting `_rehydrate_push_state` ran.
- `test_workspace_rehydrate_keeps_review_admission_attention`. Red on base: after the rehydrate the label read "Review session changes (3)" (idle) under the "Push review unavailable" panel.
- `test_workspace_rehydrate_keeps_observer_failure_attention`. The local-proof operation stays unsettled while its observer raises. Red on base even before the explicit poll: the git-mode refresh's own rehydrate had already reset the label to "· Checking push" and the view to `checking_candidate` under the failure projection.
- `test_workspace_push_observer_cancellation_publishes_nothing`. Red on base: `DID NOT RAISE CancelledError`, because the observer swallowed it.
- Green: all 3 pass, and so does the cancelled-settlement test.
- Covering run of 12 File Notes / Library Notes files (`-n 4`, load average ~19): 56 failed, 581 passed. Every failure is in the set that fails on base with the same command.

**Docs.** None. `Docs/User_Guide/library/file-notes.md` already says the header reports **Push needs attention** "as appropriate", and this makes the code match it.

**Files.** `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, `Tests/UI/test_library_file_notes_git_push.py`, `Tests/Architecture/test_module_size_ratchet.py` (9615 -> 9640, dated).
