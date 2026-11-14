---
id: TASK-34909
title: Decide form_components adopt-or-delete
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
- decision
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Widgets/form_components.py is advertised in AGENTS.md as the standardized form builder but has exactly one production consumer while 181 UI and Widgets files construct Input directly. Either it becomes the real standard or it is deleted and the signpost fixed; the current state misleads every new contributor.

Evidence (verified on dev df4985e): Single consumer: UI/Workflows_Modules/editor.py:20. Rejection documented at UI/STTS_Window.py:109 (generator and widget incompatibility). Heavy hand-rollers: settings_screen.py (76 Input constructions) LLM_Management_Window.py (29) Outputs_Panel.py (22) backup_restore_screen.py (18). No ADR references form_components. This is a cheap decision task that unblocks a later large collapse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded with rationale,Adoption path: highest-density surfaces (settings LLM management backup restore sharing) migrate to the builder, or Delete path: module removed and AGENTS.md guidance updated,If adopting the STTS-documented generator incompatibility is investigated and resolved
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
