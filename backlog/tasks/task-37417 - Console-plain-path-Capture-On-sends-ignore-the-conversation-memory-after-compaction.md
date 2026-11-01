---
id: TASK-37417
title: >-
  Console: plain-path Capture-On sends ignore the conversation memory after
  compaction
status: To Do
assignee: []
created_date: '2026-10-28 01:59'
labels:
  - console
  - trace
  - compaction
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-33621.29 (2026-10-27). On the plain-provider path (character chats, and any chat with `[console] agent_runtime = false`), a Capture-On send dispatches the trace request frozen at durable acceptance (`_run_direct_provider_reply` uses `trace_request` as the preparation input). That request is built before the conversation-memory preflight runs, so the memory projection is dropped: after a successful Compact now, the provider still receives every older turn and no memory. With Capture Off the same send carries the memory and only the retained turns.

Measured with the real controller, store, gateway preparation and compaction service (only the provider adapter recorded), dev f9baac8706: Capture On sent q0..q3 plus the new turn and `system_message=None`; Capture Off sent q3, its reply and the new turn with the memory as `system_message`. The agent path (the Console default) is not affected: its trace request factory binds the projected rows. Re-confirmed on dev 3efe8476ce (2026-10-29): after Compact now the plain path's Capture-On send still carried q0..q3, while the agent path carried only q3, its reply and the new turn.

Impact: compaction does nothing for these chats when capture is on (the summary call is still billed), so they keep growing toward the context limit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a successful Compact now, a Capture-On send on the plain-provider path carries the conversation memory and only the retained turns, as a Capture-Off send does
- [ ] #2 That send keeps trace capture: its trace call records exactly the request the provider received
- [ ] #3 A regression test drives Compact now and a Capture-On plain-path send through the real controller, store, gateway preparation and trace runtime, and fails on dev f9baac8706
<!-- AC:END -->
