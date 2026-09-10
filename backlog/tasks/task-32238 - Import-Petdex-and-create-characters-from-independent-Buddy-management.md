---
id: TASK-32238
title: Import Petdex and create characters from independent Buddy management
status: In Progress
created_date: 2026-09-10 15:16
assignee:
- '@codex'
labels:
- buddy
- petdex
updated_date: 2026-09-10 16:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the approved Petdex import and independent character conversion available from Console Buddy management without creating or selecting a Persona.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can review and install a Petdex URL or local package as an independent Buddy from Buddy management without a Persona.
- [x] #2 A user can create an independent editable character from the selected installed Buddy, retaining artwork credits and expressions after source changes.
- [x] #3 Cancel, stale selection, changed source, profile changes and publication failure do not retarget or overwrite existing content or settings.
- [ ] #4 The live prepared Petdex archive is saved through the real app journey, works offline after reload, and preserves exact creator/source/terms in a disposable profile.
- [x] #5 Targeted current-dev regression tests cover management, guarded publication, conversion and Dynamic/Static expression behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: amendment to existing ADR139, with ADR074, ADR144 and ADR145 retaining conversion, playback and Petdex trust contracts. Reason: extend the existing independent-owner management entry points; no new runtime, schema or Persona ownership. 1. Amend ADR139 and the current Buddy programme spec to describe management-owned review/publication and source snapshot guards. 2. Reuse restored Petdex source review and native export as the existing BuddyLibrary.review_archive/publish_review input. 3. Expose selected-Buddy character conversion using a revision-checked independent snapshot and the existing character review/publication service. 4. Cover cancellation/staleness and real app import/save/reload/conversion in a disposable profile, with targeted regression checks. 5. Update user and installer guidance and review the combined branch before PR/merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented independent management Petdex review/staging and selected-Buddy character creation using the existing native snapshot and publication services. Source, selection, profile and service guards remain active through preparation/publication; explicit character creation is separate from management Apply. Existing Persona routes retain their defaults. Targeted management/snapshot checks: 69 passed, 1 optional collection probe skipped. Adjacent Petdex review, character conversion and playback checks: 56 passed. Eight changed Python files pass Ruff check/format; all ten CSS outputs reproduce. ADR139 amended with existing ADR074/144/145. Real downloaded-archive qualification and independent review are still in progress; task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
