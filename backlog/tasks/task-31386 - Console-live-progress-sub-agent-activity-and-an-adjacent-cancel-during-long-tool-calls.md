---
id: TASK-31386
title: >-
  Console live progress: sub-agent activity and an adjacent cancel during long
  tool calls
status: Done
assignee:
  - '@claude'
created_date: '2026-09-04 19:29'
updated_date: '2026-09-05 01:41'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project E of the design spec (2026-08-19-console-user-interaction-design.md section 4) asked for 'what the agent is doing now, elapsed time, visible cancel during long tool calls'. Part of it has since shipped: the unfinished Assistant row shows a live activity line during a turn (tool name and elapsed seconds, then Thinking and Generating states; see Docs/User_Guide/console/agent-runs-and-tools.md 'In the reply row itself'), and the composer's Stop button cancels the run. What remains from E: a sub-agent's work never appears in that line, so a fleet turn reads as idle while children run; and the Stop button is the run-wide stop, with no per-tool-call cancel next to the activity line for a single long tool call the user wants abandoned without ending the turn. This task is the residual, re-scoped against what exists rather than the spec's original ground truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 During a fleet turn the activity line reflects the running child agents (count and the longest-running tool) rather than reading idle
- [x] #2 A long-running tool call exposes a cancel adjacent to the activity line that abandons that call and lets the turn continue, using the existing per-call abandon path
- [x] #3 Single-agent turns are unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Activity line: console_turn_activity_text gains children= (the running sub-agents' own live snapshots via bridge.live_run_snapshot); when the primary is between tools it renders the child count and the longest-running child tool instead of Thinking….
2. Per-call abandon: a module-level in-flight registry in Agents/agent_service.py; the dispatch marks the call before _call_with_timeout and ORs tool_call_abandon_requested(run_id) into that wrapper's cancel probe only, so the loop continues with the existing 'tool call cancelled' result.
3. Affordance: after CONSOLE_TURN_ACTIVITY_ABANDON_AFTER_SECONDS the tick passes a click action (Style meta, not markup) that the row header renders as '✕ abandon call'; ChatScreen.action_abandon_console_tool_call -> ConsoleChatController.abandon_active_tool_call -> request_tool_call_abandon(latest unanchored primary run).
4. Tests: pure fleet/abandon-action cases, mounted row affordance, service registry + wrapper cancel; user-guide paragraph.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fleet activity (AC #1).** `console_turn_activity_text(snapshot, *, now, children=())` in `UI/Console_Modules/agent.py`: when the primary's latest step is not a tool call and `children` (the running sub-agents' own live snapshots, read through `ConsoleAgentBridge.live_run_snapshot`, the one live source of a working child's steps) is non-empty, the line renders `N sub-agent(s) · ⚙ <tool> · <elapsed>` for the child tool that has run longest, or `N sub-agent(s) working` when no child is inside a tool call. A primary tool call still wins, and a turn with no children renders exactly as before (AC #3).

**Per-call abandon (AC #2).** `Agents/agent_service.py` keeps a module-level in-flight registry (`_mark_tool_call_inflight` / `_clear_tool_call_inflight` around the existing `_call_with_timeout` call) and `request_tool_call_abandon(run_id) -> tool name | None`. The request is ORed into that wrapper's cancel probe only, so the wrapper returns the existing "tool call cancelled" result on its next poll slice and abandons the worker thread exactly as a run-wide Stop does, while the run's own `should_cancel` stays False and the loop hands the failed result to the model. A request with nothing in flight is refused, never queued; the flag is cleared with the mark when the call ends. Calls dispatched without a timeout (`max_tool_call_seconds = 0`) bypass the wrapper and cannot be abandoned, as before.

**Affordance.** Once the primary's tool call has run `CONSOLE_TURN_ACTIVITY_ABANDON_AFTER_SECONDS` (5 s), the 0.2 s transcript tick passes `action=CONSOLE_TURN_ACTIVITY_ABANDON_ACTION` to `ConsoleTranscript.apply_turn_activity`; the row header appends `✕ abandon call` carrying the action in a `Style.from_meta({"@click": ...})` (the same mechanism Textual's Markdown links use), so the line's text still renders literally. `ChatScreen.action_abandon_console_tool_call` -> `ConsoleChatController.abandon_active_tool_call` (viewed session's durable conversation -> newest unanchored primary run, the stop path's lookup) -> `request_tool_call_abandon`, with a notice either way. The plain (non-markdown) renderer shows the line without the link.

**Files.** `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/UI/Console_Modules/agent.py`, `tldw_chatbook/Widgets/Console/console_transcript.py`, `tldw_chatbook/Chat/console_chat_models.py` (`live_activity_action`), `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Docs/User_Guide/console/agent-runs-and-tools.md`; tests in `Tests/UI/test_console_turn_activity_line.py` (fleet line, primary-wins, action threshold, mounted affordance) and `Tests/Agents/test_tool_call_abandon.py` (refused when idle, wrapper cancels only the call and clears).
<!-- SECTION:NOTES:END -->
