---
id: TASK-41643
title: "Console received-route wiring still checks collaborators by code identity"
status: In Progress
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR-356 rule 4 already decides this; the change applies it to the received route the way TASK-41319 did for Send preparation (registration + per-instance override refusal).

Per site (`W` = `UI/Console_Modules/wiring.py::receive_console_visible_intent`, `C` = `UI/Console_Modules/composer_run_controls.py`):

| Site | Decision |
|---|---|
| W `_CONSOLE_RECEIVED_SCREEN_SELECTORS` (`__func__`/`__code__`) | REPLACE: the constant becomes method names; the screen must resolve each to its own bound method with no per-instance override (new `own_methods` beside `stock_native_methods` in `Chat/console_received_dispatch.py`) |
| W `__func__` vs `_CONSOLE_TURN_CONTEXT_BUILDER` | REPLACE: `own_methods(owner, ...)` (the owner's type is already exact) |
| W scratch provider `__code__` | REPLACE: the slot is already in `_console_received_live_adapters` (what the wiring registered); keep partial/args/keywords, drop func/code |
| W RAG readers `__code__` (`_CONSOLE_RECEIVED_RAG_READERS`) | REPLACE: delete; the received route and the stock accessor both call the current chat_screen readers, so a wrapped reader gives both the same answer. The accessor slots stay registered and must be the stock accessors |
| W `_CONSOLE_RECEIVED_QUEUE_METHOD` | REPLACE: `own_methods(controller, ("queue_prompt",))` |
| C `_CONSOLE_PROMPT_QUEUE_DISPATCH`/`__func__`/`__code__` | REPLACE: exact queue type + `own_methods(queue, ("dispatch",))`; the adapters stay registered |
| C `_CONSOLE_PREPARING_SESSION_READERS` (+ the context-provider `__func__` and the RAG loop) | REPLACE: names + `own_methods`; the context provider must `==` the session's bound builder |

Steps: 1) a new UI test drives the real received route with each collaborator wrapped by `functools.wraps` (red on base where it was declined) and with a per-instance/foreign callback (refused); 2) apply the table, delete what becomes unused (`_CONSOLE_RECEIVED_RAG_READERS`), keep definitions other routes still read (`_STOCK_CONSOLE_SCRATCH_SNAPSHOT`, `_CONSOLE_TURN_CONTEXT_BUILDER`, `_CONSOLE_RECEIVED_QUEUE_METHOD` for turn_admission/prompt_queue/received_dispatch); 3) retarget tests that pin a code-identity decline (`test_console_preparing_current_sources.py` session rows, `test_screen_fallback_preserves_replaced_class_dispatch_abi`) to per-instance state changes; 4) lower the ratchet by the measured removed reads; covering files base vs head, `Tests/Architecture` serially, preflight.
<!-- SECTION:PLAN:END -->
