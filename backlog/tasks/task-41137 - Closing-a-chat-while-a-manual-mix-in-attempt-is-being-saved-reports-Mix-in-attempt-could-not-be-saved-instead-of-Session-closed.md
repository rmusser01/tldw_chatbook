---
id: TASK-41137
title: "Closing a chat while a manual mix-in attempt is being saved reports 'Mix-in attempt could not be saved' instead of 'Session closed.'"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- console
- copy
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_prompt_mixin_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-22690 gave every close path one outcome, `SESSION_CLOSED_COPY = "Session closed."`, and TASK-40312 fixed the native-commit path to match. A close that lands while a manual mix-in attempt is being accepted still reads as a save failure.

**Evidence.** `console_chat_controller.py:24614-24629`: any `MixinSnapshotUnavailableError` from `accept_attempt` blocks with "Mix-in attempt could not be saved; response was not sent.", including the owner-changed refusal a close produces (`console_prompt_mixin_service.py`, `mixin_attempt_owner_changed`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A close during the manual mix-in accept reports 'Session closed.' in the result and the run state
- [ ] #2 A real save failure still reports the save copy
- [ ] #3 Tests cover both, the first red before the change
<!-- AC:END -->
