---
id: TASK-39168
title: "Two goal-dispatch tests bound the provider's arrival by 3 s of wall clock, so they fail on a loaded host"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
- flaky
priority: low
references:
- Tests/Chat/test_console_goal_dispatch.py
dependencies:
- TASK-39143
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (lanes and cprod streams). Checked by reading on dev b590a15afd, 2026-11-08. Line numbers are dev's.

**The two tests.** `test_repeated_start_cannot_reprovision_an_accepted_attempt` (`:373`) and `test_goal_authorization_cannot_be_forged_or_moved_to_another_session` (`:408`) each hold a threading provider.
- The provider sets `entered` and blocks on `assert release.wait(3)` (`:383`, `:420`).
- The test asserts `await asyncio.to_thread(entered.wait, 3)` (`:391`, `:437`).

**What failed.** The lanes stream saw both red on dev 8b762b52fa under load, where `entered.wait(3)` returned False. They passed 3 of 3 at load 37-43, and passed here at load ~21.

**The precedent.** TASK-39143 (round 12, fu/r12-ctlr) replaced the same shape in the controller file with `until_provider_starts`. That helper waits on the provider's start, and fails naming the turn's result if the turn ends first.

**Lanes.** No PR lane runs the file, on dev or on the stack. Round 12 (fu/r12-cprod, TASK-39119) adds nodes to it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests wait on the provider's start or the turn's end, not a 3 s wall-clock bound, and fail naming the turn's result when the turn ends first; the provider's own `release.wait(3)` cannot fail a run that otherwise passes
- [ ] #2 Both pass 20 of 20 runs at `-n 4`, with load recorded, at least ten of them at a one-minute load of 40 or more
- [ ] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
