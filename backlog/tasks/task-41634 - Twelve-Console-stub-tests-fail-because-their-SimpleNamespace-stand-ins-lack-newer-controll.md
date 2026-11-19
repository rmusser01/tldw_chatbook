---
id: TASK-41634
title: "Twelve Console stub tests fail because their SimpleNamespace stand-ins lack newer controller attributes"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_turn_attention.py
- Tests/UI/test_console_transcript_context_projection.py
dependencies:
- TASK-41313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41313 found 12 tests red on dev: 3 in `Tests/UI/test_console_turn_attention.py` and 9 in `Tests/UI/test_console_transcript_context_projection.py`. They build the screen or controller from a `SimpleNamespace`, which lacks attributes the product now reads (`_current_console_conversation_id`, `sync_unproven_cleanup`). This is test drift, not a product bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The 12 tests pass, built from the real collaborator or a stub that carries what the code under test reads
- [ ] #2 No product code changes only to satisfy a stub
<!-- AC:END -->
