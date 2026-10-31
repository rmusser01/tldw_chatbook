---
id: TASK-18924
title: 'Console conversation browser: date dividers'
status: Done
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-10-11 00:00'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's sidebar date dividers (2026-08-19 hermes-release review). The Console left-rail Conversations browser groups rows only as Starred / Workspaces / Chats. Within the "Chats" group (composition decided in implementation), bucket conversation rows under date headers — Today / Yesterday / This week / Older — computed from last-activity time, so a long list reads chronologically at a glance. Pure polish over the existing browser state (Workspaces/conversation_browser_state.py).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation rows render under date-bucket headers computed from last-activity time, with deterministic bucket boundaries (today/yesterday/this-week/older) stable in tests — console_conversation_date_bucket (UTC calendar dates: Today = same UTC date, Yesterday = 1 back, This week = 2-6 back, Older = 7+; missing/unparseable falls to Older, future/clock-skew to Today) + console_conversation_date_dividers ((index, label) pairs before the first row of each bucket run, correct under recency, starred-first, and filtered orders); carried on ConsoleConversationBrowserSection.date_dividers (chats section only) and rendered by the tray. Pinned by test_date_bucket_boundaries_are_deterministic, test_chats_section_carries_date_dividers_by_recency_order, test_conversation_browser_renders_date_dividers
- [x] #2 Search filtering remains authoritative: the pinned behavior documents whether dividers stay or flatten while filtering, and tests pin that choice — DECISION (documented in the helper docstring, the state tests, and the user guide): dividers STAY while filtering — the display order stays recency-based under a filter so buckets stay truthful, and the list keeps its structure instead of reflowing on the first keystroke. Pinned by test_dividers_stay_while_search_filtering_is_active
- [x] #3 Empty buckets render nothing; the existing group collapse behavior and row secondary lines are unchanged — dividers are emitted only at bucket runs (a bucket with no rows emits nothing); collapsed sections carry date_dividers == (); rows themselves, their secondary lines, and all row-typed consumers (keyboard navigation, selection, collapse toggles) are untouched — dividers are recorded through the tray fixed-node signature so the recompose guard still catches structural changes
- [x] #4 UI tests cover bucket boundaries, search interplay, and no regression to Starred/Workspaces/Chats grouping — 9 state-layer tests in Tests/Workspaces/test_console_conversation_browser_state.py (boundaries, recency/starred order, single-bucket leading divider, cap-addressing, collapsed/empty, filtering) plus 3 UI tests in Tests/UI/test_console_workspace_context_rail.py (render + classes, list-height accounting, styles-declared per ADR-161). Neighboring suites A/B at dev 9f8533686c: Tests/Workspaces + switcher state = 8 pre-existing reds both sides, +9 passes; rail file = the pre-existing flapping private-profile family (varying membership at base), all new tests green in isolation and full-file runs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: display-only grouping over existing browser state; no schema or boundary change.

1. Compute date buckets in the browser state layer from conversation last-activity
2. Render dividers in the tray's Chats group (decide and pin search interplay)
3. UI tests + sessions-tabs-workspaces.md note
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Premise note: the task described the browser as "Starred / Workspaces / Chats" groups; the shipped design (post-TASK-2154.3) is a flat Default/unassigned browser with ONE "Chats" section built by `build_console_conversation_browser_state` (named-workspace rows live in the native Workspaces Tree). Dividers were implemented for that Chats section — the "composition decided in implementation" clause.
- Approach: pure state-layer computation (`date_dividers: tuple[(row_index, label)]` on the section, indices addressing the VISIBLE post-cap row tuple) so every existing row-typed consumer (navigation, selection, toggles, the Ctrl+K switcher path) is untouched; the tray renders a one-line dim Static before the addressed row and the height precompute adds exactly one row per divider (labels never wrap at any rail width).
- Design details: dividers are recorded through `_record_composed_node` so a row aging across a date boundary (state value change with no row-content change) still repaints; the CSS rule `.console-conversation-browser-date-divider` composes `$ds-text-muted` with a `$ds-space-2` left indent in `css/features/_console.tcss` (rebuilt via `build_css.py`); Starred-section rows keep no dividers (cross-workspace pins) and Workspaces groups own their rows.
- Modified files: `tldw_chatbook/Workspaces/conversation_browser_state.py`, `tldw_chatbook/Widgets/Console/console_workspace_context.py`, `tldw_chatbook/css/features/_console.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt), `Tests/Workspaces/test_console_conversation_browser_state.py`, `Tests/UI/test_console_workspace_context_rail.py`, `Docs/User_Guide/console/sessions-tabs-workspaces.md`, this task file.
