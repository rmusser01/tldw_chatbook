---
id: TASK-41951
title: "The Conversation Inspector leaves out a spawned child's trace calls"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- trace
- console
priority: low
references:
- tldw_chatbook/Chat/console_trace_repository.py
dependencies:
- TASK-41650
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41650 shows a child's calls in the Trace view, but the Conversation Inspector walks only the parent's lineage (`iter_message_call_lineage`), so child calls are missing there. ADR-337's note records the gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Inspector lists a turn's child calls with the parent's, labelled as a sub-agent's
- [ ] #2 A test spawns a child with capture on and asserts the Inspector shows its calls
<!-- AC:END -->
