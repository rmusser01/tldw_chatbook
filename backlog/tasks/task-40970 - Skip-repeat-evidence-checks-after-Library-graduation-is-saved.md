---
id: TASK-40970
title: Skip repeat evidence checks after Library graduation is saved
status: Done
created_date: 2026-11-18 00:19
assignee:
- '@codex'
labels:
- library
- performance
- onboarding
updated_date: 2026-11-18 00:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persistently graduated Library visits still start seven content evidence checks that cannot change the permanent rail decision. Those reads compete with the source snapshot that owns the populated first frame. Eliminate the repeated work while keeping current Library contents, recovery attention, and first-time evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saved graduated profiles refresh the current source snapshot and landing attention without re-running onboarding evidence on entry or return.
- [x] #2 First-time and unsaved graduation retain evidence, generation fencing, and the existing persistence retry.
- [x] #3 Paired private macOS PTY visits retain every result and demonstrate a faster populated first frame without losing ordinary quit or SQLite retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the current source snapshot and native calls on both saved and unsettled Library profiles.
2. Compare the baseline and disposable candidate through the same private PTY. Preserve the landing refresh for hidden-time attention changes and retain all paired results.
3. Pin saved entry and return, visible ingest failure attention, and the first failed graduation save followed by a successful retry. Use the existing private bootstrap-profile mode for mounted controls.
4. Apply the corrected refresh in the existing Library screen support module, retaining the screen hook, owners, generation fence and current source snapshot. Verify the unchanged screen size budget and replay a final end-to-end pair.
ADR required: yes, covered by existing ADR-076
ADR path: backlog/decisions/076-library-lifecycle-progressive-disclosure.md
Reason: Saved graduation is already a permanent rail decision. The current source snapshot and landing recovery continue to resolve live owners, and the existing evidence and persistence path still handles unsaved outcomes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The Library screen delegates onboarding refresh to the existing Library support module. Saved graduation settles onboarding and preserves the landing attention refresh without starting the seven repeat evidence checks. ADR-076 defines the saved full-rail decision. First-run and unsaved graduation keep the existing evidence generation and persistence retry. The normal source snapshot remains live.

The exact-source private macOS PTY pair reached populated Library in 1,214 ms versus 1,359 ms at baseline. Warm return was 61 ms versus 59 ms. Two preceding pairs with the corrected behavior saved 237 and 287 ms on cold entry. Host load varied and all timings are retained. Both 13-second windows were preserved. All 16 processes exited normally with zero owned private SQLite descriptors and zero profile guard violations. The worst final cold loop gap remained above 50 ms. The first prototype was discarded after a mounted test showed that it hid attention for an import failure that arrived while Library was covered. See the [paired results](../../Docs/QA/console-library-responsiveness/README.md#library-pty-results).

All 44 mounted Library cases and 46 focused evidence, architecture, and import checks passed. The two import warnings report budget headroom. The final three focused cases passed again after formatting. Ruff gates and touched-file lint and format comparisons passed against the frozen baseline. The Library architecture note and testing lesson record the return path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved graduation no longer launches seven repeat evidence checks. The final paired PTY run painted populated Library 145 ms sooner while preserving landing recovery, first-run evidence, and ordinary resource retirement.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
