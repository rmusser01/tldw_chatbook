---
id: TASK-14828
title: Start sits below the fold because the ingest form is long
status: Done
assignee:
  - '@codex'
created_date: '2026-08-10 22:40'
updated_date: '2026-08-12 21:12'
labels:
  - library
  - ingest
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split out of task-14822 after its AC#2 was re-measured in the shipped screen (the original tick came from the canvas mounted alone, without the Library shell chrome — see that task's notes for the full measurement and the lesson).

With the tooling warnings folded, the warning wall is no longer what pushes the form down — the summary block is now a fixed height regardless of warning count, and the type breakdown is in view. But at 235x52 with four staged groups, **Start is still 17 rows below the fold**: the canvas viewport is 43 rows (shell chrome takes 9 of 52) and Start sits at virtual y=59. It first clears at a 60-row canvas viewport, i.e. terminal height 69.

The remaining distance is the form's own length: four collapsed type panels plus three metadata Inputs at four rows each. So the commit point — the moment the user acts — is off-screen for a realistic multi-type import on a standard terminal, and the forecast/consent lines that sit beside it go with it.

Worth considering together rather than trimming blindly: whether the metadata trio (Title/Author/Keywords) needs full-height Inputs at rest, whether collapsed panels can be denser, or whether the commit row should be pinned the way the Settings screen pins its contract row (the repo already has that pattern, task-1716).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a realistic multi-type selection and warnings present, Start and its forecast/consent lines are reachable without scrolling at a supported terminal size, measured in a harness carrying the real shell chrome
- [x] #2 `Tests/UI/test_library_ingest_structural.py::test_start_still_needs_scrolling_at_52_rows` is replaced by its positive counterpart (that test fails once this lands — it is the signal)
- [x] #3 task-14822's AC#2 is re-ticked on the new evidence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This resolves a bounded layout defect within the existing Ingest canvas and Library shell.

1. Pin the forecast, consent copy, and Start decision as one viewport-owned region.
2. Reduce metadata height and verify the real shipped-shell geometry.
3. Replace the negative regression with a positive visibility contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed under TASK-15702. The forecast, gate/consent copy, and Start now live in a docked review bar and the metadata trio is a compact, persistently labeled row. The former negative 235x52 shipped-shell test is now test_start_and_forecast_are_visible_without_scrolling_at_52_rows, and the same behavior is visible in the 80x24 mixed-preflight capture. The review bar hides after submit so it does not compete with queue activity.

ADR required: no. This is a bounded layout correction within the existing Ingest canvas and Library shell.
<!-- SECTION:NOTES:END -->
