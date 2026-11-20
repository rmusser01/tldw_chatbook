---
id: TASK-40971
title: Keep the saved Library rail mounted when source counts arrive
status: Done
created_date: 2026-11-18 00:55
assignee:
- '@codex'
labels:
- library
- performance
updated_date: 2026-11-18 01:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A returning Library profile rebuilds its already-mounted rail when its first source counts land, briefly replacing keyboard targets and rerunning layout. Keep the existing controls for a stable full rail and measure the visible effect through the private macOS PTY.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The saved full rail shows fresh source counts while keeping the same focused control and retaining the native rebuild for changed structure or pending focus work.
- [x] #2 Mounted regression coverage exercises the normal snapshot and confirms onboarding, route and diagnostics behavior still holds.
- [x] #3 Paired runs of the same sealed private PTY navigation show whether the populated Library frame benefits, with every full observation window, result and ordinary SQLite retirement retained.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the production navigation timeline, separating transaction setup from the page SQL and mounted rail work.
2. Hold the existing snapshot in a mounted saved-profile test, focus the original search input, and release the real source reads. Verify the current generation updates the count on the same control tree.
3. Extend the existing rail update path only for a stable graduated landing. Keep the native rebuild for changes to the diagnostics, widget shape, lifecycle, route, and any pending or active recompose or focus callback.
4. Run alternating private PTY pairs with the full 13-second cold and warm windows. Retain all outcomes and keep the change only with a visible or interaction benefit backed by the evidence. Check adjacent mounted onboarding, focus, counts, diagnostics and architecture budgets.
ADR required: no new ADR
ADR path: backlog/decisions/076-library-lifecycle-progressive-disclosure.md
Reason: The existing full rail, source authority, recompose mechanism, and lifecycle command are unchanged. This uses the established in-place count update for the saved landing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The graduated Library landing updates the mounted rail when the layout, diagnostics, and controls remain stable. The mounted baseline focused Search before releasing the source worker and caught focus moving to the shell navigation overflow button. The final test preserves the original focused input and conversation row while the true count appears. It also checks that a queued focus completion reaches the replacement input after a native rebuild. ADR-076 covers the lifecycle and rail contract.

Across three alternating private macOS PTY pairs, the cold populated median moved from 1,520 ms to 1,330 ms and the warm median from 61 ms to 56 ms. Cold results improved twice and regressed once. Host load varied. Time from snapshot completion to the populated frame fell in all three pairs, and every candidate removed the one cold rail rebuild. The worst candidate cold loop gap was 604 ms. Both 13-second windows were kept, and all eight diagnostic and paired PTY processes exited normally with no private SQLite descriptors left open and no profile guard violations.

Two original docked-cue tests failed on both versions because they expected Browse to lead directly to Create after Artifacts was added. The test setup now focuses the last Artifacts row and proves Create is below the cue before Tab. All original painted focus, scroll, and toggle checks remain. Both themes passed with the corrected fixture on the frozen baseline and the candidate. The final 57 mounted checks, 10 architecture and import checks, Ruff gates, and changed-file formatting checks passed. The two import notices report budget headroom. The architecture and evidence lesson document the behavior. See the [paired results and integration evidence](../../Docs/QA/console-library-responsiveness/README.md#library-pty-results).

During integration, dev already included an in-place sync for both full rail lifecycles. The mounted test on dev confirmed that Search remained focused but a queued focus callback never ran. The PR keeps the broader in-place sync and lets a pending or active recompose, or its queued callback, finish through the native replacement path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Library keeps Search focused when counts arrive on the stable graduated landing. The three sealed PTY pairs each skipped the rail rebuild. Median populated cold opening was 190.61 ms sooner, with one slower pair retained.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
