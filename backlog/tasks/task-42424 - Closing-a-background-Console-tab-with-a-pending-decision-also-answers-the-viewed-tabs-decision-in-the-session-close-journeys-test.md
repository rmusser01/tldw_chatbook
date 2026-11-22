---
id: TASK-42424
title: "Closing a background Console tab with a pending decision also answers the viewed tab's decision in the session-close journeys test"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_session_tab_close.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** The required "UI Fast Lane (26)" fails on `Tests/UI/test_console_session_tab_close.py::test_session_close_pending_journeys` with `closing the background tab answered the viewed tab`. In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), it failed 25 jobs, from 2026-11-20 09:06Z to 2026-11-22 20:01Z, every one with this message (all 25 logs read), for example runs 44948612804, 44952102973, 44953045328, 44974735998, 44974790480, 44978286545, and 45016281720 attempts 2 and 3. It also fails locally at the base: 2 of 3 runs at load 6-8 (one in a `-n 4` run, one of two serial runs), with the same message at line 1129.

**The failure.** The scenario (`test_console_session_tab_close.py:1074-1131`) arms a pending decision in the viewed chat (`sibling`, session `keeper`) and another in a background chat (`doomed`), then closes the background tab. The close must answer only the doomed session's decision. At line 1129, `assert not sibling.done()` fails: the viewed chat's decision was answered too. If that is not a test artefact, closing one chat resolves another chat's approval or question, which a user would see as an answer they never gave.

TASK-35587 (In Progress since 2026-10-19) is about the same test failing at app shutdown with a `CancelledError`, a different signature. TASK-42342 (round 21) changed how a session close drains the controller's preparation reads. This stack carries that change and the test still fails here, but re-check on the dev that merges it before deciding the cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Why the viewed chat's pending decision is answered when the background tab closes is recorded: a cross-session product defect, or a test artefact, with the evidence
- [ ] #2 A defect is fixed so that closing a chat answers only that chat's pending decisions, with a test shown red first; an artefact is fixed in the test
- [ ] #3 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->
