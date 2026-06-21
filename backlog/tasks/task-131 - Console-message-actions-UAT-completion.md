---
id: TASK-131
title: Console message actions UAT completion
status: To Do
assignee: []
created_date: '2026-06-21 00:36'
labels:
  - console
  - messages
  - uat
dependencies:
  - TASK-128
  - TASK-129
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden message selection and per-message actions so users can operate on transcript messages with both mouse and keyboard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can select a transcript message with mouse and keyboard and see the selected-message action row.
- [ ] #2 Enter activates the selected message or focused action consistently, and Tab moves between major Console screen areas.
- [ ] #3 Copy works and provides visible status feedback.
- [ ] #4 Edit opens a dedicated edit modal, saves changes to the selected message, and does not confuse editing with composing a new message.
- [ ] #5 Save as opens the conversion modal and clearly labels unavailable destinations as unavailable or WIP.
- [ ] #6 Regenerate creates variants and exposes previous/next controls for choosing which variant to continue from.
- [ ] #7 Continue extends the selected message/thread context according to its documented meaning.
- [ ] #8 Thumbs up/down and delete provide clear feedback and safe recovery or confirmation where destructive.
- [ ] #9 Focused regression coverage verifies mouse and keyboard paths for the action row and the implemented actions.
- [ ] #10 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Parallel Ownership

Owns transcript message selection and per-message action behavior. Avoid provider readiness, workspace state, and tab/composer lifecycle except where required to create fixture messages for action testing.

ADR required: no, unless implementation changes message storage schema, variant ownership, or destructive-action policy.
ADR path: N/A unless implementation planning identifies a contract change.
Reason: Expected work is behavior hardening over existing message/action UI seams.
