---
id: TASK-1234
title: 'Fleet UX copy and ergonomics batch'
status: Done
assignee: ['@claude']
created_date: '2026-07-28 09:30'
labels: [console, ux, polish, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F5/F6/F7 batch: (a) cap-refusal number agreement ("1 agents" -> "1 agent") and consider naming the refusal actionable; (b) tab auto-titles ellipsize mid-string ("What is t…ate an.") -> end-truncate; (c) Stop button tooltip "Stop this tab's run" (scoping uncommunicated under parallel runs); (d) Settings Scope Inspector doesn't auto-scroll to the Focused-field guide on focus (consequences copy below the fold); (e) "Tools: 0 ready" chip before first run despite enabled tools (lazy catalog) reads as no-tools; (f) single-row approval cards could offer one-click Approve-once/Deny without Submit; (g) "(high risk)" on reads lacks a why affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed item fixed or explicitly ruled out with a reason in the notes.
- [x] #2 No behavioral change to approval safety or run scoping.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate each item's source (grep-driven): cap-refusal copy in `ConsoleChatController.send_refusal_copy`; tab-title truncation in `ConsoleSessionSurface._display_title`; Stop button build (both compose-time and the live `sync_action_state` override) in `console_composer_bar.py`; the guided-Settings `DescendantFocus` handler in `settings_screen.py`; the "Tools" chip in `ConsoleControlState.from_values`; the approval card in `chat_approval_card.py`.
2. Fix (a) singular/plural agreement; update the pinned plural-form test coherently.
3. Fix (b): switch `_display_title` to end-truncation (helper scoped to tab titles only, not shared with session auto-titling); document the disambiguation trade-off task-375 originally solved; update its two dependent tests.
4. Fix (c): tooltip copy at BOTH the compose-time button build and the live `sync_action_state` override that had been silently clobbering it every refresh.
5. Fix (d): shared `_scroll_impact_pane_to_field_guide` helper wired into every guided category's branch of `handle_descendant_focus`, using the `call_after_refresh` + `force=True` pattern proven in `library_screen._preserve_library_rail_scroll`.
6. Fix (e): neutral "Tools: not loaded" placeholder at a zero effective tool count in `ConsoleControlState.from_values`, scoped to that one chip's copy only (not the count/active semantics, not `ConsoleInspectorState`'s separate "review tool call" gate).
7. Fix (f): single-row fast "Approve once"/"Deny" buttons in `ChatApprovalCard.set_batch`, wired through the existing `ApprovalDecided`/`round_id` seam.
8. Fix (g): why-affordance tooltip on the row header for `risk_floored` rows.
9. TDD per item; run the required gates; mutation-check downstream consumers of changed copy (caught two additional pre-existing-adjacent regressions this way -- see notes).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All seven items fixed; none ruled out. No change to approval safety (the fast-approve button maps to `approve_once` ONLY, through the unmodified `ApprovalDecided`/`round_id` resolution path) or run scoping (cap/refusal math, `_live_busy_session_ids`, `max_parallel_runs` all untouched).

**(a) Cap-refusal grammar** -- `ConsoleChatController.send_refusal_copy` (`Chat/console_chat_controller.py`) now picks "agent"/"agents" by count. The "actionable refusal" half of the finding (jump-to-tab) was explicitly out of scope per the batch's own disposition guidance (toasts aren't interactive in Textual; the fleet line + ◆ markers already provide navigation) -- not implemented, not re-litigated here.

**(b) Tab-title truncation** -- `ConsoleSessionSurface._display_title` switched from TASK-375's middle-truncation to end-truncation, matching `derive_console_session_title`'s existing convention. `_display_title` is scoped to this one file (confirmed via grep -- no other caller), so no shared-helper blast radius. Trade-off documented in the docstring and in a test: two titles sharing a long common prefix can render an identical tab label again (TASK-375 added AC#2 specifically to prevent this); accepted because the mid-word garble was judged the worse defect by the live UAT review, and the full title remains one hover away in the tab tooltip.

**(c) Stop button tooltip** -- two call sites needed the fix, not one: the compose-time `_bounded_button(..., tooltip=...)` AND `ConsoleComposerBar.sync_action_state`'s live override, which unconditionally re-set the tooltip on every action-state refresh and would have silently clobbered a compose-time-only fix. Caught by writing a test that actually activates a run (mirroring `test_console_stop_button_hidden_unless_streaming`'s setup) rather than checking the idle/just-mounted state.

**(d) Settings Scope Inspector auto-scroll** -- new shared `SettingsScreen._scroll_impact_pane_to_field_guide`, called from every guided category's branch of `handle_descendant_focus` (Appearance, Storage, Library/RAG, Console Behavior, Providers & Models). Uses `call_after_refresh` + `scroll_to_widget(..., force=True)`, the same pattern proven in `library_screen._preserve_library_rail_scroll` for a just-recomposed container whose scroll bounds haven't settled yet. Revert-checked: temporarily disabling the call reproduces the exact pre-fix failure.

**(e) "Tools: 0 ready" chip** -- `ConsoleControlState.from_values` now reads "Tools: not loaded" at a zero effective count instead of "Tools: 0 ready". Investigated the eager-honest-count alternative (counting `Agents/tool_catalog.py`'s `ALWAYS_ON_BUILTIN_NAMES` + config-enabled `_GATEABLE_BUILTINS` -- genuinely cheap, no catalog building) but rejected it: it would also feed `ConsoleInspectorState`'s separate "Review tool call" gate (a DIFFERENT concept -- "were any tool calls actually made this run" -- not "how many tools are configured") and falsely mark review actionable before any call ever happened, breaking that gate's own tests. Scoped the fix to the chip's copy alone; `tools_active` (dim/emphasis) is unchanged. Caught (mutation-check, not the named gate) that the compact mode-bar summary (`ChatScreen._console_mode_summary`) parsed `tools_label`'s first word as a literal count -- "Tools: not loaded" rendered as the nonsensical "Tools not"; fixed `readiness_count` to fall back to "—" for any non-numeric token.

**(f) Single-row approval fast path** -- `ChatApprovalCard.set_batch` mounts two additional compact buttons ("Approve once" / "Deny", `variant="success"`/`"error"`) only when the batch collapses to exactly one row; both decisions are legal for every row this card ever renders (verified: MCP rows always get the full four-option set, and the one narrowed case in production -- built-in tools -- keeps both `approve_once` and `deny`), so no `legal_values` gating was needed. Both post the same `ApprovalDecided(decisions, round_id=...)` message through `_submit_fast_decision`, mirroring `_submit_batch_decisions` exactly -- no new resolution seam. Multi-row batches and "Approve for session"/"Always allow" still require Select+Submit. Total diff well under the 150-line rule-out threshold.

**(g) "(high risk)" why-affordance** -- the row header `Static` now carries a `.tooltip` for `risk_floored` rows ("Reads can exfiltrate file contents; built-in file tools always ask before running."); `config_changed` rows get no tooltip (their badge is already self-explanatory).

**Verification discipline**: for every non-trivial disposition call (b, e) and for two mutation-check findings outside the named gate (the mode-bar "Tools not" text, the Stop button's live tooltip override), used a stash-based before/after comparison to prove pre-existing failures were genuinely pre-existing (identical failure with the fix reverted) rather than assuming from context.

Gates (each one blocking foreground `pytest`, interpreter `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, worktree `/private/tmp/tldw-fxr`):
- `Tests/Chat/test_console_run_state_per_session.py` + `Tests/UI/test_console_mcp_approval.py` + `Tests/UI/test_settings_configuration_hub.py` + `Tests/UI/test_console_session_tab_strip.py`: 305 passed, 24 failed -- all 24 confirmed pre-existing via stash-diff (provider-model-resolution `TypeError`, a `save_setting_to_cli_config`/`save_settings_to_cli_config` naming-drift bug, an unrelated `PrivatePathError`, and one MCP-approval-card geometry test), none touching this task's changed files.
- `Tests/UI/test_console_parallel_runs.py`: 28 passed, 0 failed.
- Non-gate spot-check (not required, run because these files exercise the changed widgets/copy directly): `Tests/UI/test_console_internals_decomposition.py` + `Tests/UI/test_console_native_chat_flow.py` + `Tests/Chat/test_console_display_state.py`: 348+ passed; remaining failures all confirmed pre-existing/flaky via stash-diff except the two real regressions this pass caught and fixed (mode-bar "Tools not" text; Stop button's live-override tooltip).

Files modified: `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/console_display_state.py`, `tldw_chatbook/Widgets/Console/console_session_surface.py`, `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated `tldw_chatbook/css/tldw_cli_modular.tcss`); test updates across `Tests/Chat/test_console_run_state_per_session.py`, `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_internals_decomposition.py`, `Tests/UI/test_settings_configuration_hub.py`, `Tests/UI/test_console_mcp_approval.py`.
<!-- SECTION:NOTES:END -->
