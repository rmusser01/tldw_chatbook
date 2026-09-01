---
id: TASK-25903
title: 'Console: mid-run steering for the primary agent'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:08'
updated_date: '2026-09-01 05:31'
labels:
  - console
  - agents
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user who wants to correct a running agent must stop it, losing every completed tool result, then retype. Verified on origin/dev: the steering machinery already exists and is wired for fleet children only - Agents/agent_runtime.py:1196-1230 drains a mailbox before each model call and never splits a tool_calls/tool pair, format_steering_message and the MAX_STEERING_CHARS cap are in place, and a user-facing steering bar exists (UI/Console_Modules/agent.py:1445) - but drain_mailbox is None for a primary run by explicit design (Agents/agent_service.py:3486), so typed text goes to the prompt queue for the next turn instead (Chat/console_prompt_queue.py:60). This is the core TUI interaction and the smallest of the top-ranked gaps because the protocol-coherent drain point is already proven in production for children.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Text submitted while the primary agent is running can be delivered to the current run instead of queued, at the user's choice
- [x] #2 Steered text is drained at the same protocol-coherent point children use - before a model call, after budget and cancel checks - and never splits a tool_calls/tool message pair
- [x] #3 The steered message is visible in the transcript as user-authored, distinct from the original prompt
- [x] #4 The existing queue path remains available and unchanged for users who prefer it; the default is stated explicitly in the task notes
- [x] #5 Steering a run that has already finished or been cancelled is refused honestly rather than silently dropped
- [x] #6 The same character cap and sanitization applied to child steering applies here - verified by tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Wires an existing, production-proven seam (the fleet drain point) to a new producer; no protocol change.

1. Reuse the child steering machinery wholesale: the drain point, format_steering_message, STEERING_SOURCE_USER, MAX_STEERING_CHARS. The task is a producer, not a protocol.
2. Service-side mailbox keyed by run_id, registered in _run_one for primary runs only; unregistration in the finally makes stale steer callables refuse honestly (AC#5 is structural, not policed).
3. Run ids are minted inside run_turn, so the Console cannot key by one: on_primary_steer_ready hands the controller a bound steer(text) at registration time, threaded through run_reply's on_steer_ready kwarg -- the same optional-callback pattern the bridge already uses.
4. The user's choice (AC#1) is per message: plain submission still queues (AC#4 default unchanged); /steer is the explicit opt-in.
5. Update the two tests pinning "primary stays unwired" -- that WAS the design, because no producer existed; steer_primary is now the producer, and the pins now assert the child/primary drains are distinct objects and inline children stay unwired.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The primary run can now be steered mid-flight. Every piece of machinery existed -- the protocol-coherent drain point, `format_steering_message`, the 4000-char cap, `STEERING_SOURCE_USER` -- but was wired for fleet children only; `drain_mailbox` was None for a primary BY DESIGN, the design reason being that no producer existed. `AgentService.steer_primary` is now that producer.

**Service half.** A mailbox per live primary run, registered in `_run_one` before the first model call and unregistered in a `finally` around `run_agent_loop` -- so a steer against a finished, cancelled or crashed run refuses honestly by construction rather than by policing (AC#5). Validation mirrors the fleet `send_to_agent` path exactly (AC#6): stripped, non-empty, capped at `MAX_STEERING_CHARS` with the cap named in the refusal. Delivery rides the EXISTING drain seam untouched, so steered text is consumed before a model call, after the in-flight batch's results are fully appended -- it structurally cannot split a native `tool_calls`/`role:"tool"` pair (AC#2), and it lands as a `format_steering_message`-labelled user-role message (AC#3). An end-to-end loop test steers between turns and asserts the text arrives after the batch's tool result.

**Run ids are minted inside `run_turn`,** so the Console cannot key by one. `AgentService(on_primary_steer_ready=...)` fires at mailbox registration with a `steer(text) -> refusal | None` bound to that run; `run_reply` threads it through as the additive `on_steer_ready` kwarg (the bridge's established optional-callback pattern), and the controller stores it per session beside `_active_cancel_events` -- a stale entry is harmless because the service refuses once the mailbox is gone.

**The user's choice (AC#1/#4).** Plain submission while a run is active still queues for the next turn -- the default is unchanged. `/steer <text>` is the explicit per-message opt-in, registered in the console grammar with a suggestion-popup description, dispatched through the existing name->handler map on the chat screen, and surfacing every refusal as a visible notify. `Docs/User_Guide/console.md` documents the command and the queue-by-default contract.

**Two superseded pins updated, not routed around.** `test_only_the_threaded_fleet_child_is_wired_for_drain` and `test_inline_children_and_their_primary_stay_unwired` pinned "primary drain is None". Each now pins what still matters: the child and primary drains are DISTINCT objects (different producers), and inline turn-scoped children remain unwired. The updated tests catch the wiring's removal -- verified by mutation: skipping mailbox registration fails both.

**Verification.** 7 service/loop tests (refusals for unknown/finished/empty/over-cap, user-sourced drain, ready-callback lifecycle, end-to-end transcript placement); grammar/suggestion pins updated for the new command; `Tests/Agents/` holds at the stable 7 baseline (2320 passing); the wider sweep unchanged at the 2 known MCP baselines.

**Scope note:** the `/steer` handler chain is verified link-by-link (grammar parse, handler-map entry, controller seam, service refusals) but was not driven through a live TUI session in this pass -- a runtime check with the `verify` skill on a real long-running turn is the remaining confidence step, worth doing alongside 26000's UI work.

**Files:** `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_command_grammar.py`, `tldw_chatbook/Chat/console_command_suggestions.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Docs/User_Guide/console.md`, `Tests/Agents/test_primary_steering.py` (new), `Tests/Agents/test_fleet_steering_mailbox.py` (2 pins updated), `Tests/Chat/test_console_command_grammar.py`, `Tests/Chat/test_console_command_suggestions.py`.
<!-- SECTION:NOTES:END -->
