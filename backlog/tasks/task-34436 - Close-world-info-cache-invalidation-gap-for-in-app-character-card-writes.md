---
id: TASK-34436
title: Close world-info cache invalidation gap for in-app character-card writes
status: To Do
created_date: 2026-10-08 00:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34415 review: in-app card writes (ccp_character_handler update_character, local_character_persona_service:821) bypass WorldBookManager so a changed embedded character_book serves stale lore with no bounded window. Bump on book-bearing card updates or fold a card version into the ADR-221 cache key; also add the counter-atomicity note to ADR-221.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Card write that changes embedded book invalidates the processor cache,ADR-221 updated (residual + atomicity),Tests cover the ccp and persona-service write paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 review minors + task-3-report.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
