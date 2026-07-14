---
id: TASK-227
title: >-
  Stop task-cancel can race the agent worker finally-reset and finish done
  instead of cancelled
status: To Do
assignee: []
created_date: '2026-07-14 03:33'
updated_date: '2026-07-14 04:06'
labels:
  - console
  - agents
  - reliability
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while testing the stop-before-first-token fix: stop_active_run()'s task cancellation can race _run_agent_reply's finally block resetting _stop_requested, letting a still-running background bridge thread complete its run as 'done' although the user stopped it. Documented in .superpowers/sdd/task-8-report.md (gate fix wave). Narrow window; the store-level no-op fix covers the common case.

Broadened by the Plan-B final whole-branch review (.superpowers/sdd/plan-b-final-review.md) to also cover two related narrow races in the same stop/cancel family:
- LOW-1: ConsoleChatStore.reset_stream_content unconditionally sets a message's status to "streaming", so a disobedient model's post-stop tool-call turn can resurrect an already-stopped message back to "streaming" and leave it stuck there (append_stream_chunk was hardened to no-op on "stopped"; reset_stream_content was not).
- LOW-2: ConsoleChatController._finalize_agent_reply's mark_message_complete/finalize_variant_stream/mark_message_failed calls all raise ValueError via _validate_can_mark_terminal when a Stop lands in the narrow window after asyncio.to_thread returns RUN_DONE but before finalize (the message is already "stopped"); benign in effect (run_state still correctly settles STOPPED) but surfaces a logged exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stop that lands during an in-flight agent bridge thread always persists the run as cancelled,A regression test covers the cancel-vs-finally race deterministically
- [ ] #2 reset_stream_content no-ops (mirroring append_stream_chunk) when the target message is already stopped, with a regression test (LOW-1)
- [ ] #3 The post-completion Stop race in _finalize_agent_reply no longer raises or logs an unhandled ValueError, with a regression test (LOW-2)
<!-- AC:END -->
