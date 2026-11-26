---
id: TASK-42555
title: "Console Retry of a failed agent turn records no trace call under Capture On"
status: To Do
assignee: []
created_date: 2026-11-26 04:50
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_trace_service.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-42525 round-3 re-review (2026-11-25), outside that change's scope. With Capture On, the reviewer's probe of Console's Retry on a failed agent turn recorded no trace call at all, with or without a provider switch. The trace route registry lists RETRY as `CONVERSATION_TRACE`, so a retried agent turn is expected to be traced like any other captured call.

Either Retry reaches the provider without a trace call (a capture gap: a captured chat would send a request the trace never recorded), or Retry of a failed agent turn does not send at all (a broken action). Which one is not yet known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes say what Retry of a failed agent turn does under Capture On today (whether the provider is called, what the trace records, what the user sees), shown by a test through the real controller with only the provider handlers faked
- [ ] #2 With Capture On, Retry of a failed agent turn either sends and is traced as a call of that turn, or is refused before any provider call with copy that says why; it never reaches the provider untraced
- [ ] #3 The test is red before the change and green after, and runs on a PR lane
<!-- AC:END -->
