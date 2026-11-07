---
id: TASK-35588
title: Defer Library notes-sync owner construction until after mount
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 03:52'
updated_date: '2026-10-19 04:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep first Library navigation responsive while preserving the app-owned notes-sync lifecycle and original checked configuration reads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library construction does not build the lazy notes-sync owner.
- [x] #2 The mounted screen binds the original app owner off-thread and retains honest loading, failure, and existing ready-owner behavior.
- [x] #3 Targeted lifecycle tests and native Windows profiling verify the change without weakening storage checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; preserve ADR-126 and TASK-21249 app ownership. Reason: routine scheduling correction using the existing finite Library worker boundary. 1. Pin deferred construction and late publication behavior. 2. Prepare the original safe runtime reader after mount and bind its result on the UI loop. 3. Verify lifecycle and sync regressions, then repeat the same Windows tour.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Library construction now reuses a ready owner or preparing projection. The original safe reader is prepared through the existing finite service worker after mount; Notes attention/location/hint paths use the same boundary, and synchronous listeners never trigger construction. Safe-reader failure check/publication is serialized to prevent conflicting concurrent outcomes. Tests cover eager construction, off-loop publication, unmount, failed construction, preserved attention and concurrent readers; all new controls observed red before repair. 51 targeted cases pass, plus a final 22-case batch; the one excluded focus case and seven authoring fixture failures also fail on original HEAD. Single native Windows tour: first Library arrival 1074ms to 534ms, first refresh 2519ms to 2230ms; warm visits remain variable. No main-thread construction samples remain. Independent review approved; no new per-file lint diagnostics and changed-region formatting passes. Architecture docs updated. ADR required: no; existing ADR-126 and TASK-21249 lifecycle retained. Full evidence/limits: Docs/Development/evidence/windows-performance-2026-10-18/FIX_RESULTS.md.
<!-- SECTION:NOTES:END -->
