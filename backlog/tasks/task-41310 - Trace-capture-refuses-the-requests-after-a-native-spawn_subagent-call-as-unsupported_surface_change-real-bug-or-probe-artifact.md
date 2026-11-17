---
id: TASK-41310
title: "Trace capture refuses the requests after a native spawn_subagent call as unsupported_surface_change: real bug or probe artifact?"
status: To Do
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
- tracing
priority: medium
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_errors.py
- Tests/Chat/test_console_default_send_capture.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715's real-app probe found it (PR #3548 lists it under "Follow-ups to be filed"). Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`. It was not re-run.

**Why.** In a headless real-app probe (the private-child harness in `Tests/Chat/test_console_default_send_capture.py`), a native `spawn_subagent` call was followed by two model requests: the child's, and the parent's follow-up. Trace capture refused both with `TraceSurfaceChangeRefused('unsupported_surface_change')`. If this is real, every sub-agent turn with trace capture on stops at "Trace capture blocked". The probe's provider response was fake, so the refusal may instead come from a reply shape no real provider sends. Nobody has yet determined which.

**Evidence.**
- `Chat/console_trace_errors.py:39-43`: `TraceSurfaceChangeRefused` is raised by a surface fence before any call is reserved.
- `Chat/console_trace_service.py:2766-2795`: the surface check admits only appended messages before an unchanged continuation, or a bounded rewritten tail. Anything else is refused as `unsupported_surface_change`, recorded as `count_or_span` or `sequence_gap`.
- Related, not the same: TASK-37602 covers the same refusal after Edit & resend on providers that record continuation groups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reproduction with a real provider adapter, or a provider double whose native tool-call response matches a real provider's, records whether the child's request and the parent's follow-up after a native `spawn_subagent` call are refused, and which refusal reason (`count_or_span` or `sequence_gap`) fires
- [ ] #2 If they are refused, both requests reach the provider with trace capture intact, pinned by a regression test through the real controller, store and trace runtime that is red on dev
- [ ] #3 If they are not refused, the task notes name the probe's response shape that caused the refusal, and the probe harness is corrected
<!-- AC:END -->
