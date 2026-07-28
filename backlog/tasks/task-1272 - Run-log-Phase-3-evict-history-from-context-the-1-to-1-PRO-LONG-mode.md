---
id: TASK-1272
title: Run log Phase 3 — evict history from context (the 1:1 PRO-LONG mode)
status: To Do
assignee: []
created_date: '2026-07-28 00:00'
labels: [agents, run-log, llm]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**This is the phase that actually delivers the original goals.** Phase 1 (PR #1066) is
deliberately additive: it writes a lossless log and makes truncated content recoverable, but
`run_agent_loop`'s `messages` list is untouched, so context usage is unchanged. Long-horizon
runs and small-context local models — goals 1 and 2 of the design — land here.

The mechanism: keep recent rounds in context verbatim, replace older ones with a pointer to
the log, and let the agent retrieve on demand via `search_run_log`. The log is already the
authoritative record, and the record format already carries `call=<tool_call_id>` for exactly
this purpose.

**Three traps, all discovered during Phase 1 and recorded so they are not rediscovered:**

1. **Whole-group eviction is mandatory.** The native tool-call protocol pairs an assistant
   `tool_calls` echo with its `role="tool"` replies by `tool_call_id`. Evicting either half
   alone produces a request that strict providers reject. Any policy must operate on entire
   call/result groups.

2. **Reuse `bound_messages_to_window`, do not reimplement it.** TASK-322 shipped
   `Chat/console_history_budget.py` with window lookup, safety margin, reply reservation and
   system-prefix preservation already solved. It also already bounds the history an agent run
   *starts* from, because `console_chat_controller.py` does `agent_messages =
   list(provider_messages)`. This task is the in-run bound; the two are layered, not
   alternatives.

3. **`_group_turns` is wrong for fence-protocol runs.** It splits on `role == "user"`, and its
   docstring notes it never splits a tool_call/tool_result pair *"were tool rows ever present
   in the payload"* — they never are on the Console send path it was built for. In an agent
   run they are, and the two protocols differ: native appends `{"role": "tool", ...}` (grouped
   correctly), but **fence appends `{"role": "user", "content": "Tool result for ..."}`**,
   which reads as a new turn boundary and splits an assistant turn from the result answering
   it. Fence is the protocol local models use — precisely the case this phase targets — so
   reuse is correct for native runs and broken for fence runs until grouping learns the
   convention, or grouping is done on the log's own record structure where `call=` pairs them
   unambiguously.

Also note a Phase 1 failure mode this phase resolves (design spec §10.2): because nothing is
evicted today, a `search_run_log` result enters history like any other tool result, so heavy
log searching currently *increases* context pressure rather than reducing it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An agent run bounded by a small context window completes work that currently fails, with evicted history replaced by a pointer the agent can act on
- [ ] #2 Eviction operates on whole call/result groups; no request is ever emitted with an orphaned assistant `tool_calls` echo or an orphaned `role="tool"` reply
- [ ] #3 Correct behaviour is proven for BOTH the native and the fence tool-call protocols, with a test that would fail if fence tool-results were treated as turn boundaries
- [ ] #4 The trimming primitive from `console_history_budget.py` is reused rather than reimplemented, or a recorded reason explains why it could not be
- [ ] #5 The mode is configurable and off by default, so existing runs are unchanged until opted in
- [ ] #6 A live run against a local small-context model demonstrates a task completing that does not complete with eviction disabled
<!-- AC:END -->
