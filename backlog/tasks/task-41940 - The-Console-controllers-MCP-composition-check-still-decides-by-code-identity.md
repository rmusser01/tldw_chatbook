---
id: TASK-41940
title: "The Console controller's MCP composition check still decides by code identity"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- console
- adr-356
priority: medium
references:
- tldw_chatbook/Chat/console_chat_controller.py
- Tests/Architecture/test_code_identity_ratchet.py
dependencies:
- TASK-41644
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** ADR-356 rule 4: no authority from code identity. TASK-41644 removed the identity reads from `Chat/console_configuration_preparation.py`'s MCP composition check, but the controller keeps its own copy: the deferred branch in `console_chat_controller.py` still reads the checker's `__code__`/`__globals__`, and `_stock_console_tool_composition_current` reads `__code__`, `__globals__`, `__defaults__`, `__kwdefaults__`, `__closure__` and `__func__`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The controller's composition check decides from the registered checker and runtime state, not code identity
- [ ] #2 A wrapped checker behaves as stock and a foreign or per-instance one is refused (tests)
- [ ] #3 The code-identity ratchet pins drop by the removed reads
<!-- AC:END -->
