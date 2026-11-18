---
id: TASK-41798
title: Batch Home Notes preview and count
status: In Progress
created_date: 2026-11-18 17:55
assignee:
- '@codex'
labels:
- performance
- home
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 18:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce time to usable Home content by reusing one guarded Notes connection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The installed file-backed Home route returns the same recent Notes and exact count using one native connection preparation.
- [x] #2 Partial failures, custom stores, and hidden Home keep their existing behavior and retire native reads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the Home demand tests with the actual Notes owner, connection identity, and preview failure.
2. Reuse the Library combined Notes reader through the finite local worker on the existing Home seam. A failed combined preview still tries the exact count.
3. Verify Home demand and existing Notes controls and measure native preparations.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: Reuse the existing finite admitted owner and data contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The stock file-backed Home route reuses the guarded Library Notes snapshot reader. The count survives a failed preview and the current worker retires the native read on cancellation. The paired snapshot median improved from 80.9 to 42.4 ms with two preparations reduced to one. Three cold Home Terminal pairs improved the populated-frame median from 478.5 to 344.3 ms. The Home seam and demand regressions and ordinary Quit passed. ADR-126 governs the existing ownership. Local acceptance is verified. Hosted qualification is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
