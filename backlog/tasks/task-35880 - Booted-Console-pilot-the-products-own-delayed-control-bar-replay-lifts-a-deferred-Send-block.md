---
id: TASK-35880
title: >-
  Booted-Console pilot: the product's own delayed control-bar replay lifts a
  deferred Send block
status: To Do
assignee: []
created_date: '2026-10-21 00:34'
labels:
  - console
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shared test helper _sync_console_control_bar_now (Tests/UI/test_console_native_chat_flow.py) renders the Console control bar itself, so none of its roughly 70 callers checks that the product's own coalesced replay (_request_console_control_bar_sync(delayed=True), the 0.2 s retry a deferred sync hands off to under TASK-34415) lifts a deferred Send block in a booted app. The first version of TASK-35660's stale-proof replay covered this by accident, because Send was enabled by that replay during its 1.1 s stall; the tightened replay checks only the helper's own syncs and no longer does. The gap predates TASK-35660; it was raised by that task's independent review on 2026-10-20.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A pilot that boots the real Console makes the provider selection's control-bar sync defer, renders nothing from the test, and sees Send become pressable through the product's own delayed replay within a stated bound
- [ ] #2 With the delayed replay removed or made a no-op (scratch mutation, recorded in the task notes), the pilot fails naming Send's blocked state
- [ ] #3 The pilot passes under the plain pytest command and carries the bootstrap_profile marker
<!-- AC:END -->
