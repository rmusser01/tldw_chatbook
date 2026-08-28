---
id: TASK-22515
title: Make Console provider Apply update and persist conversation settings
status: To Do
assignee: []
created_date: '2026-08-28 05:52'
updated_date: '2026-08-28 06:07'
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
Ensure the Console Provider/Model popover and full Console Settings apply provider-generation choices to the exact conversation immediately, preserve them across restart, and give mouse and keyboard users the same reliable Apply behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mouse and keyboard Apply both close the popover and update the exact originating conversation.
- [ ] #2 Provider, model, temperature, and streaming apply immediately to all work created after Apply while already-running work remains unchanged.
- [ ] #3 Quick Provider and full Console Settings use one conversation-owned durable generation-settings contract.
- [ ] #4 Reopening a persisted conversation after restart restores its applied generation settings without storing credentials or endpoints.
- [ ] #5 Changing providers cannot retain the previous provider endpoint or incompatible provider-specific settings.
- [ ] #6 Unsaved conversations stage settings until first persistence, while temporary conversations remain non-durable unless promoted.
- [ ] #7 Invalid input and dismissed deferred callbacks cannot create a false-success close or teardown error.
- [ ] #8 Targeted tests cover routed mouse clicks, keyboard Apply, persistence and resume, metadata conflicts, and stale-endpoint prevention.
- [ ] #9 The Provider/Model popover contains and returns no compaction controls or values; compaction remains independently available in full Console Settings.
<!-- AC:END -->
