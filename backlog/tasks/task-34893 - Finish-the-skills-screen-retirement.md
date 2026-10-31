---
id: TASK-34893
title: Finish the skills screen retirement
status: Done
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
assignee:
- '@zcode'
updated_date: 2026-10-12 19:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI/Screens/skills_screen.py (1351 LOC) is unreachable behind the skills-to-library alias in screen_registry.py and survives only because Library imports its two trust modals. The recorded retirement decision should be finished.

Evidence (verified on dev df4985e): screen_registry.py _SCREEN_ALIASES documents nine completed retirements; skills was kept-not-deleted deliberately (comment in registry) for tests and modal reuse. Importers of the modals: UI/Library_Modules/library_skills_controller.py:410 and UI/Screens/library_screen.py:557. Net deletion ~1k LOC after extraction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SkillTrustBootstrapModal and SkillTrustPassphraseModal extracted to a Library-owned module,SkillsScreen class and its scaffolding deleted,Screen registry alias for skills unchanged,Skills tests retargeted to Library surfaces
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract the two trust modals to a Library-owned module 2. Update the two production importers and the modal/census test registries 3. Delete SkillsScreen and its scaffolding 4. Retarget or trim skills screen tests 5. Targeted test run vs baseline 6. Notes + AC tick
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Extracted THREE trust modals (not two as the task estimated - SkillRecoveryReviewModal was also live) to UI/Library_Modules/skills_trust_modals.py; deleted UI/Screens/skills_screen.py (1351 LOC). Importers repointed: library_screen.py library_skills_controller.py plus five test files and three registries (modal_wide_tier width table css parse-cache probe recovery taxonomy screen list). Deleted the skills destination-shell suite (20 mount/flow tests three fakes two wait helpers map entry PHASE4 constant and the skills-only unwired-action test) and the parity file skills entries (workbench contract pane titles loading contract; the Library seam-placeholder assertions were kept - they pin live LibraryScreen behavior). Registry alias comment updated. Kept: skills->library alias unchanged per AC. Verification: 764 tests collect and run; no new failures vs baseline (the ~20 test_skills_library_flow failures reproduce with changes stashed AND in the untouched main checkout - pre-existing private-path guard environment issue); 22 baseline failures eliminated with the removed skills cases.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
