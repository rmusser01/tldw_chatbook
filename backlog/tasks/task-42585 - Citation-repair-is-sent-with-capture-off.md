---
id: TASK-42585
title: "Citation repair is sent with capture off"
status: To Do
assignee: []
created_date: 2026-11-26 18:48
labels:
- follow-up
- console
- trace
priority: low
references:
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_trace_provenance.py
- Tests/Chat/test_console_route_census_capture.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42560
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42560 (round 27, 2026-11-26).

**Why.** With Capture On, a citation repair ("Citations repaired") sends its repair call with capture off, so the trace has no call for the request that produced the repaired reply. The route census says so (TASK-42552), and `Tests/Chat/test_console_route_census_capture.py` drives it. TASK-42560 traced Edit & resend, Regenerate and Continue, and left citation repair out of scope. The repair replaces the saved reply after the captured call, so it is expected to need the same kind of rule as a Regenerate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On, a citation repair's call is traced as a call of its turn, and the message after it is sent and traced, shown by a test through the real controller, red before and green after; or an owner decision to keep it capture off is recorded in the task
- [ ] #2 The route census row for citation repair, its capture test and the User Guide agree with what the repair does
- [ ] #3 Any new admission rule is recorded in ADR-337 before the code
<!-- AC:END -->
