---
id: TASK-32238
title: Import Petdex and create characters from independent Buddy management
status: Done
created_date: 2026-09-10 15:16
assignee:
- '@codex'
labels:
- buddy
- petdex
updated_date: 2026-09-10 18:04
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
- [x] #4 The live prepared Petdex archive is saved through the real app journey, works offline after reload, and preserves exact creator/source/terms in a disposable profile.
- [x] #5 Targeted current-dev regression tests cover management, guarded publication, conversion and Dynamic/Static expression behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/146-independent-buddy-petdex-publication-and-character-creation.md. Reason: ADR-146 supplements Accepted ADR-139 without rewriting it and partially supersedes only ADR-145 saved-Persona-only destination restriction; saved Persona authoring and every source, transport, review and attribution decision remain in force. 1. Reuse the existing Petdex review and BuddyLibrary publication boundaries with capability-safe local file access. 2. Expose independent Buddy publication and selected-Buddy character creation through the existing management form with source/profile/selection guards. 3. Keep cancellation before Apply or Save side-effect free; when publication succeeds before settings persistence fails, retain the installed Buddy and previous settings, then reuse it on retry or direct the user to reopen and verify it before importing again. 4. Preserve publication and character attribution, source snapshots, Dynamic/Static playback and path-free actionable errors. 5. Run only the targeted Petdex, management, conversion, formatting, generated-asset and diagnostic preflight checks; record fresh results and leave the task In Progress for root closeout.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented independent management Petdex staging/publication and selected-Buddy character creation through existing guarded native services. Final review fixes add capability-probed local folder, pet.json and ZIP reads with bounded lstat/open/recheck fallback; shared path-free native/staged publication recovery; accurate publication-before-settings partial-Apply guidance and retry reuse; the reviewed diagnostic inventory pin; and only the named Buddy formatting hunks. Accepted ADR-139 is restored exactly. Supplemental Accepted ADR-146 extends its management entry points and partially supersedes only ADR-145 saved-Persona-only destination restriction; saved Persona authoring and every ADR-145 source, transport, review and attribution decision remain in force. Fresh targeted tests: Petdex sources/conversion/publication 39 passed; management Petdex/import/modal 57 passed; coordinator 17 passed. Scoped Ruff checks pass, CSS generated assets reproduce, diagnostic inventory has no drift, and git diff --check passes. Earlier live-source qualification passed on af85f8a854 and remains recorded in Docs/superpowers/reviews/2026-09-10-independent-buddy-journey-verification.md. Root owns final current-dev integration, combined-head downloaded-source app rerun and closeout; task remains In Progress.
Root closeout: current dev 5a9371c4f3 integrated as 8aaadb56f1; 113 targeted Buddy/Petdex checks and 30 shell/CSS checks passed. Actual downloaded archive production-app journey passed in 37.04 seconds, retaining 14 independent expressions, Dynamic/Static behavior, credits and offline restart. All seven required preflight checks passed. Scoped final re-review addressed all six original findings; root fixed the remaining Windows reparse-point gap using shared filesystem identity checks, with eight discriminating cases and 47 Petdex source/conversion/publication tests passing. Scoped Ruff and diff checks pass. ADR146 and the committed journey verification record final contracts, evidence, rulings and actual-Windows/native/physical limits. Implementation and self/independent review complete; root continues authorized PR publication and merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Independent Buddy management now supports reviewed Petdex installation and editable character copies without a Persona, retaining credits and Dynamic/Static expressions. Current-dev targeted and real-archive headless application checks pass; native terminal and physical voice acceptance remain separate tracked tasks.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
