---
id: TASK-22515
title: Make Console provider Apply update and persist conversation settings
status: To Do
assignee: []
created_date: '2026-08-28 05:52'
updated_date: '2026-08-28 06:23'
labels: []
dependencies: []
references:
  - ADR-095
documentation:
  - >-
    Docs/superpowers/specs/2026-08-27-console-provider-apply-persistence-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the Console Provider/Model popover and full Console Settings apply provider-generation choices and compaction to the exact conversation immediately, preserve them across restart through their existing owners, and give mouse and keyboard users the same reliable Apply behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mouse and keyboard Apply both close the popover and update the exact originating conversation.
- [ ] #2 Provider, model, temperature, streaming, and compaction apply immediately to all work whose execution context is resolved after Apply while already-captured work remains unchanged.
- [ ] #3 Quick Provider and full Console Settings use one exact-origin Apply orchestration and the same durable conversation-generation contract while compaction keeps its existing context-policy owner.
- [ ] #4 Reopening a persisted conversation after restart restores its applied generation settings and compaction without storing credentials or endpoints.
- [ ] #5 Changing providers cannot retain the previous provider endpoint or incompatible provider-specific settings.
- [ ] #6 Unsaved conversations stage both durable components until first persistence, while temporary conversations remain non-durable unless promoted.
- [ ] #7 Invalid input and dismissed deferred callbacks cannot create a false-success close or teardown error.
- [ ] #8 After the modal closes, generation-settings, context-policy, and dual save failures remain visibly identified per component with Retry; a quick-surface context-policy failure may be labeled compaction.
- [ ] #9 The Provider/Model popover retains compaction mode and applies it through the existing context-policy owner without changing compaction storage or schema.
- [ ] #10 Targeted tests cover routed mouse clicks, keyboard Apply, exact-origin execution, persistence and resume, partial failure and retry, metadata conflicts, staged first persistence and promotion, and stale-endpoint prevention.
<!-- AC:END -->
