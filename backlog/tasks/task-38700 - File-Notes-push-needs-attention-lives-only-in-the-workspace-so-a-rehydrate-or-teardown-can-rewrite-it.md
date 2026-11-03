---
id: TASK-38700
title: "File Notes: push 'needs attention' lives only in the workspace, so a rehydrate or teardown can rewrite it"
status: To Do
created_date: 2026-11-03 11:40
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a push review admission failure and after a push observer failure, a Git rehydrate leaves the Session Git label and the push panel showing the same state the failure published, or both cleared together; a test drives each through a real rehydrate and fails on the base build
- [ ] #2 Cancelling the push observer (workspace shutdown or loop teardown) publishes no 'needs attention' result, and the cancellation propagates; pinned by a test
<!-- AC:END -->
