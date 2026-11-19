---
id: TASK-41643
title: "Console received-route wiring still checks collaborators by code identity"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
- architecture
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/wiring.py
- Tests/Architecture/test_code_identity_ratchet.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** This is a second ADR-356 rule 4 follow-up from TASK-41319. `UI/Console_Modules/wiring.py` still decides the received route by identity:
- `_CONSOLE_RECEIVED_SCREEN_SELECTORS`;
- the `__func__` comparison against `_CONSOLE_TURN_CONTEXT_BUILDER`;
- the scratch-provider and RAG-reader `__code__` checks;
- `_CONSOLE_RECEIVED_QUEUE_METHOD`;
- `composer_run_controls`' `_CONSOLE_PROMPT_QUEUE_DISPATCH`/`__code__` and `_CONSOLE_PREPARING_SESSION_READERS`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The received route decides from registrations or runtime-owned state, not code identity
- [ ] #2 Wrapped collaborators behave as stock; foreign ones are refused
- [ ] #3 The code-identity ratchet is lowered by the removed reads
<!-- AC:END -->
