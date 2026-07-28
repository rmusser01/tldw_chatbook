---
id: TASK-1272
title: 'Run log Phase 3 — evict history from context (the 1:1 PRO-LONG mode)'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 00:00'
updated_date: '2026-07-28 22:33'
labels:
  - agents
  - run-log
  - llm
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
- [x] #1 An agent run bounded by a small context window completes work that currently fails, with evicted history replaced by a pointer the agent can act on
- [x] #2 Eviction operates on whole call/result groups; no request is ever emitted with an orphaned assistant `tool_calls` echo or an orphaned `role="tool"` reply
- [x] #3 Correct behaviour is proven for BOTH the native and the fence tool-call protocols, with a test that would fail if fence tool-results were treated as turn boundaries
- [x] #4 The trimming primitive from `console_history_budget.py` is reused rather than reimplemented, or a recorded reason explains why it could not be
- [x] #5 The mode is configurable and off by default, so existing runs are unchanged until opted in
- [ ] #6 A live run against a local small-context model demonstrates a task completing that does not complete with eviction disabled
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Phase 1/2 code (agent_service._make_call_model, agent_runtime._append_tool_result) and the design spec's Phase 3 section (10, 10.1, 10.2, 14.1).
2. Extend console_history_budget.py: add an optional is_turn_boundary predicate to _group_turns/bound_messages_to_window (default unchanged, so Console callers are byte-identical), and a dropped_turns count on BoundResult.
3. Promote the fence tool-result prefix ("Tool result for ") to a shared constant in agent_models.py, used by both agent_runtime._append_tool_result and the new eviction module, so the two can never drift.
4. Add Agents/run_log_eviction.py: a round-boundary predicate (every assistant message starts a new round; a native role="tool" reply or a fence role="user" tool-result row is a continuation) and bound_history_for_send(), which calls bound_messages_to_window and splices in a synthetic note when something was dropped.
5. Wire bound_history_for_send into agent_service._make_call_model's call_model closure, gated on log_active (reused verbatim) AND a new off-by-default [agents] run_log_evict_enabled config flag (via run_log._setting, so it gets the same env-var/TOML/default tiering as the other run-log keys).
6. Tests: unit tests proving the fence/native round-boundary fix against the raw primitive (including the explicit "force naive grouping" experiment), plus integration tests through AgentService.run_turn proving the flag-off/log-unavailable hard gates and the end-to-end payload shape.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented eviction at the SEND seam (agent_service._make_call_model's call_model
closure), never touching run_agent_loop's own messages list. Reused
bound_messages_to_window (console_history_budget.py) rather than reimplementing it,
extending it with an optional is_turn_boundary predicate (default unchanged, so
every existing Console call site stays byte-identical) and a new dropped_turns
count on BoundResult.

Key design call beyond the literal "fix _group_turns for fence" framing: the
reused primitive's "turn" is Console's own -- anchored on the last human message
-- which inside a single agent run (one such message, at the start) would
collapse the run's ENTIRE growth into one undroppable "current turn" and evict
nothing while a run is in progress, defeating goals 1/2 (long single run, small
local model). Agents/run_log_eviction.py therefore uses a finer ROUND boundary:
every assistant-authored message starts a new round; a native role="tool" reply
or a fence role="user" tool-result row is a continuation, never a boundary.
Verified via an explicit experiment (recorded in the task's test file) that the
UNMODIFIED primitive (no is_turn_boundary) orphans a fence tool-result and never
trims a native run's own growth at all -- both failure modes the round boundary
fixes, with no orphaned call/result pair for either protocol at any drop size.

FENCE_TOOL_RESULT_PREFIX ("Tool result for ") promoted to a shared constant in
agent_models.py so agent_runtime._append_tool_result and the eviction module's
protocol check can never drift apart.

Gated on log_active (reused verbatim from _run_one -- the same condition gating
the tool and the prompt section) AND a new [agents] run_log_evict_enabled flag,
off by default, resolved via run_log._setting (same env-var/TOML/default tiering
as the other run-log keys). When something drops, a role="user" synthetic note
(not role="system" -- some local chat templates reject a system row that isn't
first) is spliced in naming a round count and search_run_log; never a specific
record number, since the loop doesn't track which record backs which round.
Eviction never raises: any failure inside bound_history_for_send is caught,
logged at warning, and degrades to sending the full history for that turn.

Files: tldw_chatbook/Agents/run_log_eviction.py (new),
tldw_chatbook/Agents/agent_service.py, tldw_chatbook/Agents/agent_models.py,
tldw_chatbook/Agents/agent_runtime.py, tldw_chatbook/Chat/console_history_budget.py,
Tests/Agents/test_run_log_eviction.py (new, 13 tests), design spec doc updated
(§8, §10).

AC #6 (live run against a local small-context model) is NOT verified here --
no local model available in this environment; left open for live verification
per the repo's live-verification rule.
<!-- SECTION:NOTES:END -->
