---
id: TASK-26024
title: Auxiliary model routing for side tasks
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - providers
  - performance
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Titles, compaction and other side work run on the user's main chat model. Verified on origin/dev: Chat/Chat_Functions.py:186-217 defines SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS - the auxiliary set is identified and audited - but those calls dispatch through the same API_CALL_HANDLERS table at the same model, so a user on an expensive reasoning model pays that rate to generate a conversation title. Hermes routes side tasks to a cheaper tier with a documented resolution order.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A configurable auxiliary model handles side tasks (at minimum: conversation titling and compaction) instead of the main chat model
- [ ] #2 With no auxiliary model configured, behavior is exactly as today
- [ ] #3 Auxiliary selection falls back to the main model when the configured auxiliary is unavailable or unconfigured, rather than failing the side task
- [ ] #4 Auxiliary usage and cost are attributed separately in accounting so the saving is measurable
- [ ] #5 The auxiliary model never handles user-visible chat turns - asserted by a test over the audited endpoint set
- [ ] #6 Sensitive auxiliary endpoints continue to honor the existing audit constraints when routed to a different provider
<!-- AC:END -->
