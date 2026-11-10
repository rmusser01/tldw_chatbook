---
id: TASK-39163
title: "test_console_automatic_library_preparation.py still has 14 unbounded waits on pre-provider boundaries, so a turn that ends early hangs the file until the 300 s timeout"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_automatic_library_preparation.py
dependencies:
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39045 on round-12 branch fu/r12-ctests; its notes propose this follow-up). Checked on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are the stack's, since TASK-39045 changes this file.

**The helper.** TASK-39045 added `_until_provider_starts(started, turn, timeout=30)` and moved every provider-start wait onto it. If the turn ends first, the helper fails naming the turn's result.

**The 14 waits still bare.** These waits on boundaries before the provider are still a bare `await ….wait()`:
- `service.entered` (`:1239`);
- `held.entered` (`:2444`, `:2486`, `:2549`, `:2711`, `:2754`, `:2879`, `:2962`, `:3170`, `:3260`, `:3406`, `:3493`);
- `boundary.entered` (`:2777`);
- `gateway.both_entered` (`:2820`).

**What happened.** TASK-39045's control was a temporary edit that made `_apply_skill_substitution` refuse every turn. Under it, the whole-file run hung on these waits until it was stopped.

**Lanes.** The file has 90 tests, 23.6 s at `-n 4` and load 24-34 (TASK-39045). It runs on no PR lane.

**Fixed elsewhere.** The vision-seam red that TASK-39045 also noted (`Tests/Chat/test_console_chat_controller.py::test_submit_draft_sends_image_parts_when_vision_capable`) is fixed by TASK-39138 in round 12. It is red on dev and passed on the stack (run.sh, load ~23).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every boundary wait in the file is bounded and, when the turn ends first, fails naming the turn's result
- [ ] #2 Under the refuse-every-turn control, the whole file finishes with each such test failing on its bounded wait, well inside the 300 s timeout
- [ ] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
