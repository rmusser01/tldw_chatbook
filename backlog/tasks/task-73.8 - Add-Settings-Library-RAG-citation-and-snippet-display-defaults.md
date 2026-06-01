---
id: TASK-73.8
title: Add Settings Library RAG citation and snippet display defaults
status: To Do
labels:
- settings
- library
- rag
- citations
- snippets
priority: medium
parent_task_id: TASK-73
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add real Settings controls for Library/RAG citation and snippet display defaults after the source-of-truth config contract is defined, without changing retrieval execution ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Library/RAG citation display defaults have a concrete persisted source of truth.
- [ ] #2 Library/RAG snippet display defaults have a concrete persisted source of truth.
- [ ] #3 Settings can load, validate, save, and revert the display defaults without mutating retrieval execution state.
- [ ] #4 Console and Library RAG displays use the same effective citation/snippet defaults.
- [ ] #5 Actual CDP/Textual-web screenshot QA verifies the controls before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Settings controls are backed by a concrete persisted config source
- [ ] #3 Console and Library/RAG consumers use the same effective defaults
- [ ] #4 Automated tests cover load, save, revert, and display behavior
- [ ] #5 Actual CDP/Textual-web screenshot QA is completed and approved
- [ ] #6 Documentation and task notes updated
<!-- DOD:END -->
