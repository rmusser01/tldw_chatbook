---
id: TASK-41139
title: "Console command popup and visit dispatch reds left by TASK-40323: Enter with the popup closed never sends, and a first visit syncs task-resume state 5 times"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: '2026-11-15 19:30'
labels:
- follow-up
- console
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_command_popup.py
- Tests/UI/test_console_visit_dispatch_dedupe.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40323 laned the green nodes of these files and left these reds, which may be product defects: a send that never happens, and redundant work on every Console visit.

**Evidence (TASK-40323's notes, base 26b2e677dd, files under `bootstrap_profile`).**
- `test_console_command_popup.py::test_enter_with_popup_closed_sends_normally`: "Expected mock to have been awaited"; the Send press never reaches `submit_draft`. Red with the stand-in factory too.
- `test_console_visit_dispatch_dedupe.py::test_console_first_visit_refreshes_skills_and_task_resume_once`: "Console synced task-resume state 5x for one visit".
- `test_console_visit_dispatch_dedupe.py::test_console_later_resume_still_refreshes`: "Returning to Console must re-sync task-resume state".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node is classified as test drift or a product defect, with the commit that moved it named for drift
- [x] #2 A product defect is fixed with the node red before the fix; drift is re-pinned
- [x] #3 The nodes run on a PR lane once green
<!-- AC:END -->

## Implementation Plan

1. Run each node at the base through run.sh and read the failure.
2. Bisect / read history to classify each as test drift (name the commit) or a product defect.
3. Fix defects with the node red first; re-pin drift.
4. Put the green nodes on a PR lane (UI census tail or an existing lane).

## Implementation Notes

All three nodes are test drift; none is a product defect. Each is re-pinned and runs on the UI lane.

**Base cc6006fcf5, through run.sh (`-n 3`, load 82):** the popup node failed "Expected mock to have been awaited"; both visit nodes failed with `KeyError` on the pushed screen's skill counter. (TASK-40323 saw the visit nodes fail their count assertions at 26b2e677dd.)

- `test_console_command_popup.py::test_enter_with_popup_closed_sends_normally`: **drift, 3e618d7119** (2026-10-15). A stock Send is now received into runtime custody (`receive_console_visible_intent` -> `ConsoleRuntime.accept_received_intent`), and `controller.submit_draft` runs on the runtime's custody task, not a Textual worker, so `workers.wait_for_complete()` plus one pause no longer covers it. A probe saw the spy awaited 5.4 s after the press (load 106). Re-pinned: the test polls (up to 30 s) for the call.
- `test_console_visit_dispatch_dedupe.py::test_console_first_visit_refreshes_skills_and_task_resume_once`: **drift.** A probe of one first visit counted 5 `sync_task_resume_state` calls: one bare per-visit sync (the 0.05 s timer) and four state pushes through `set_task_resume_state`, from the pending question (`_set_console_pending_question`, routed there by 524de00360, 2026-09-04) and pending decision being re-projected on attach and visibility (`on_console_view_visibility_changed(True)` on resume since 11150b849d, 2026-09-08, `finish_view_reconciliation`, the approval remount). The per-visit dispatch is still once; the pushes are equality-guarded card syncs carrying new state. Separately, on a loaded host the ChatScreen the test pushed landed under the app's own initial Console (stack `Screen`, the test's ChatScreen, the app's ChatScreen), so the measured screen was never visited: that is the `KeyError`. It depends on load, not on a commit.
- `test_console_visit_dispatch_dedupe.py::test_console_later_resume_still_refreshes`: **drift, the same pushes.** A resume also re-projects the pending decision and question, so the count rose by 3, not 1.

**Re-pin.** The fixture also patches `set_task_resume_state` and does not count a sync made inside it. Both tests now observe the app's own first Console visit (the class methods are patched before the app is built): they wait until it is reconciled and has dispatched both refreshes (up to 60 s), then 0.5 s for repeats. The resume test waits the same way for the resume's refreshes. Negative control: with `on_screen_resume`'s visit token disabled (`mount_already_refreshed = False`, reverted), the first-visit test fails "Console dispatched the skill-candidate refresh 2x for one visit".

**Green:** 3 passed (`-n 3`, load 94). **Lane (AC#3):** UI census tail, the popup node and the visit file whole; `MINIMUM_FILES` 2147, no re-deal. Serial with the lane's flags (`-p no:xdist --timeout=180 --durations=0`, load 104-118, setup + call + teardown): popup node 42.3 s, first visit 43.3 s, later resume 42.9 s.

Files: `Tests/UI/test_console_command_popup.py`, `Tests/UI/test_console_visit_dispatch_dedupe.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
