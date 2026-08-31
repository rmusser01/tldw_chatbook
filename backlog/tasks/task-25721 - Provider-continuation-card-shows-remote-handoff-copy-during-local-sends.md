---
id: TASK-25721
title: Provider continuation card shows remote-handoff copy during local sends
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A plain local send surfaced a card reading that response delivery status is unknown on the source device and warning that retrying may send a duplicate request. There is no source device in a local single-machine send. The card also omits the Owner, Problem and Impact structure the sibling interrupt card uses, so the product presents two different grammars for the same class of blocking decision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Continuation-recovery copy appears only for sends that genuinely crossed a device boundary
- [ ] #2 All blocking interrupt cards use one consistent Owner, Problem, Impact and action structure
- [ ] #3 Card copy names a concrete cause rather than an internal subsystem
<!-- AC:END -->
