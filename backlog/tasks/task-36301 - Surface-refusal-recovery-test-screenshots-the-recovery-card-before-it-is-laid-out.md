---
id: TASK-36301
title: >-
  Surface-refusal recovery test screenshots the recovery card before it is laid
  out
status: To Do
assignee: []
created_date: '2026-10-22 19:07'
labels:
  - console
  - tests
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_console_surface_refusal_recovery.py::test_surface_refusal_recovers_the_existing_accepted_user waits only for the TraceCallRecoveryCallout's display flag, then exports a screenshot and asserts 'Trace capture blocked' is in it. Under host load the card can be display=True while it has not been laid out yet (region 0x0), so the frame lacks the card and a random parametrized case fails. Seen on dev 7700c63a68 itself (detached worktree, 2026-10-22): 2 of 7 file runs failed (retry+cancel in one, send-without in another), both while other pytest runs loaded the host; the same race failed one case in 4 of 9 file runs of the TASK-33621.20 branch under the same load. A probe that dumped the frame on failure showed card title 'Trace capture blocked', display=True, region=Region(0,0,0,0); passing frames show region 78x7 at y=6. Found while diagnosing PR #3066's merge-queue eviction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test waits until the recovery card is laid out and painted (non-empty region) before it exports the screenshot
- [ ] #2 Ten consecutive file runs on a loaded host pass, against a recorded failure rate on the pre-fix test
- [ ] #3 The wait stays within the test's existing bound, so a card that never paints still fails
<!-- AC:END -->
