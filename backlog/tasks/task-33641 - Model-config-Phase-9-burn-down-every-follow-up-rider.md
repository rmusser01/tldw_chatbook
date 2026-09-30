---
id: TASK-33641
title: 'Model config Phase 9: burn down every follow-up rider'
status: To Do
assignee: []
created_date: '2026-09-29 22:40'
labels:
  - model-config-redesign
  - phase-9
  - follow-ups
dependencies:
  - TASK-33008
references:
  - backlog/docs/spec-2026-09-26-model-config-redesign.md
  - qa/model-config-ux-review-2026-09-26/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This is the final phase of the model-configuration redesign, added by owner decision on 2026-09-29. Phases 1–8 and the two hotfixes filed small follow-up riders as review, live-capture and Qodo findings surfaced. Some are pre-existing defects, some are test-hygiene gaps, and some are copy or edge cases that fell outside their phase. This phase closes all of them, so the program ends with no open tail.

Riders open on dev when this task was filed:
- **TASK-33001 series:** .8, .9, .10, .11, .12, .14, .15
- **TASK-33002 series:** .7, .8, .9, .10, .11, .13, .14, .15, .16, .17, .19

Riders that Phases 3–8 file are in scope too. Riders already assigned to a specific phase are done in that phase, not here (for example TASK-33003.8, 33004.8, 33006.6 and 33007.9). The work ships as one PR, or as a few grouped PRs if the riders split cleanly by area, through the program's usual SDD plus Qodo loop.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every rider listed in the description is Done with its ACs ticked, or closed with a written rationale the owner can review. None is silently dropped.
- [ ] #2 Every rider filed during Phases 3–8 that no later phase owns is likewise Done or closed with a rationale.
- [ ] #3 A final sweep of backlog/tasks for open TASK-33001.x through TASK-33008.x riders comes back empty, or lists only the closures named in AC#1 and AC#2.
- [ ] #4 Each fix carries a test that fails before it; config and provider fixes use real-implementation tests on a scratch profile.
- [ ] #5 ./scripts/preflight.sh passes, and each PR is rebased on dev, Qodo-clean and merged.
<!-- AC:END -->
