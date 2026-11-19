---
id: TASK-41644
title: "Console configuration preparation still compares MCP capture and bound methods by identity"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- architecture
priority: low
references:
- tldw_chatbook/Chat/console_configuration_preparation.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** These are the last two TASK-41319 follow-ups (ADR-356 rule 4), in `Chat/console_configuration_preparation.py`:
- `_composition_mcp_capture_current` (`:1498`) checks the checker's `__code__`/`__globals__`;
- the bound-method comparisons at `:2473` and `:2670` test `__self__`/`__func__` identity, where `==` on the bound method expresses the same thing without reading code identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Neither site reads `__code__`, `__globals__` or `__func__` identity
- [ ] #2 Existing preparation tests pass, and the code-identity ratchet is lowered accordingly
<!-- AC:END -->
