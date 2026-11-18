---
id: TASK-41605
title: Keep the Notes Sync runtime off cold Library construction
status: In Progress
created_date: 2026-11-18 02:37
assignee:
- '@codex'
labels:
- library
- performance
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
updated_date: 2026-11-18 03:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Entering Library on a fresh app imports the Notes Sync engine and executor on the UI thread to show the initial status. Keep that import and construction off the initial Library path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh Library constructor projects the existing preparation state without importing the Notes Sync engine or executor.
- [x] #2 Mounted Notes Sync retains the same app-owned runtime, status validation, and reviewed-action behavior.
- [x] #3 The sealed macOS PTY records first Library paint and import work under the full observation windows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Freeze the current PR head and pin fresh Library construction in an isolated interpreter.
2. Share the existing snapshot models without loading the runtime or executor. Keep the existing finite worker and exact type checks for live operations.
3. Verify mounted Notes Sync, architecture guards, and paired cold/warm navigation in the sealed macOS PTY.
ADR required: no
ADR path: backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md and backlog/decisions/097-boot-budget-ratchets.md
Reason: The existing projection and runtime ownership contracts stay intact.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing preparation, root, and control projections now live in the shared Notes Sync models. The runtime keeps the same exports and the Library's worker still attaches the app-owned runtime. Live operations keep their exact result-type checks. The fresh constructor regression caught both eager imports on the baseline. The chooser test keeps the disabled and ready assertions while checking the existing preparation copy.

Across three sealed PTY pairs, Notes Sync constructor import CPU fell from 67.85 ms to 8.76 ms at the median. Cold populated paint moved from 1,401 ms to 1,227 ms at the median. One cold pair improved. Gaps above 50 ms fell from 26 to 23 across the full observation windows, and the worst gap rose from 560 ms to 745 ms. The 477 focused Notes and Library tests passed. Frozen baseline controls reproduce the first-Send warm-up and legacy boot-suite failures. Follow-up CI and qualification remain open.

See [the paired evidence](../../Docs/QA/console-library-responsiveness/README.md#cold-notes-sync-and-ordinary-quit), [ADR-073](../decisions/073-notes-sync-round-trip-and-interoperability-constraints.md), and [ADR-097](../decisions/097-boot-budget-ratchets.md).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
