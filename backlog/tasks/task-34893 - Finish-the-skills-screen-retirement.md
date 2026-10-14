---
id: TASK-34893
title: Finish the skills screen retirement
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
- dead-code
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI/Screens/skills_screen.py (1351 LOC) is unreachable behind the skills-to-library alias in screen_registry.py and survives only because Library imports its two trust modals. The recorded retirement decision should be finished.

Evidence (verified on dev df4985e): screen_registry.py _SCREEN_ALIASES documents nine completed retirements; skills was kept-not-deleted deliberately (comment in registry) for tests and modal reuse. Importers of the modals: UI/Library_Modules/library_skills_controller.py:410 and UI/Screens/library_screen.py:557. Net deletion ~1k LOC after extraction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SkillTrustBootstrapModal and SkillTrustPassphraseModal extracted to a Library-owned module,SkillsScreen class and its scaffolding deleted,Screen registry alias for skills unchanged,Skills tests retargeted to Library surfaces
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
