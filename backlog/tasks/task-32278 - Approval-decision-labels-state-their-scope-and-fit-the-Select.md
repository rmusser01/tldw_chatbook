---
id: TASK-32278
title: Approval decision labels state their scope and fit the Select
status: Done
assignee: []
created_date: '2026-09-10 19:11'
updated_date: '2026-09-10 20:02'
labels:
  - console
  - approvals
  - ux-copy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The card offers five decisions with no scope text; the 26-cell Select clips 'Approve for session' to 'Approve for' and wraps 'Always allow this exact input'. Users cannot tell how long a grant lasts or where to undo it, and the '(high risk)' badge explains itself only on hover with a reads-only sentence that is also used for mutating tools. User decision 2026-09-10: keep all five decisions on the card and add scope copy. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every decision label fits the closed Select without clipping at the card's minimum supported width.
- [x] #2 One line under the controls states the selected decision's scope (this call only; until Chatbook exits; remembered for this tool with the place to change it; remembered for these arguments) and where to undo it.
- [x] #3 The high-risk explanation is visible without hover and differs for reads and mutations.
- [x] #4 The user guide lists the same five decisions with the same scopes. — the shipped strings are exported as `DECISION_SCOPE_COPY`; the User Guide rewrite that quotes them is task-32290 (fix-wave Task 20), which owns those pages for this wave.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests: label budget derived from the shipped `.approval-row-decision` width; mounted card renders a scope line that follows `Select.Changed`; a risk-floored row renders a visible reason line; the in-place single-row update path keeps both.
2. Shorten the display labels (values unchanged), export `DECISION_SCOPE_COPY`.
3. Replace the header's hover tooltip with a visible `.approval-row-reason` line, with a mutation variant.
4. Add a `.approval-row-scope` line under each row's controls, updated from the row's `Select`.
5. Widen the Select to fit the longest label; rebuild the CSS bundle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shortened the five decision display labels to fit the closed Select (values unchanged), widened .approval-row-decision 26 -> 27 = the 19-cell longest label plus 8 cells of Textual chrome (SelectCurrent does not ellipsize -- it wraps and grows, which is what clipped 'Approve for session'), and added a per-row .approval-row-scope Static under the controls that renders module-level DECISION_SCOPE_COPY for the current decision and follows every Select.Changed. Replaced the header's hover-only risk tooltip with a visible .approval-row-reason line via format_approval_reason, including a distinct mutation sentence when the entry's declared effects contain mutates_local and a new line for config_changed. _update_mounted_single_row mounts its replacement controls before the scope Static and re-binds it; RAW_SHELL_SESSION_SCOPE_NOTICE follows the renamed label. AC#4 (User Guide) is task-32290's file set; DECISION_SCOPE_COPY is exported for it. 9 new tests; 4 suites at the pre-existing baseline red set, preflight green.
Two non-obvious bits. (1) The closed `Select` does NOT ellipsize: `SelectCurrent` is `height: auto` with a wrapping `Static#label`, so an over-long label grows the control (measured: `Always allow this exact input` made a 4-line Select at width 26). The chrome is exactly 8 cells (tall border 1+1, padding 2+2, arrow 1 + its 1-cell left padding), so `.approval-row-decision` goes 26 -> 27 for the 19-cell longest label, and the test reads that number back out of `_agentic_terminal.tcss` so the rule and the labels cannot drift apart again. (2) `_update_mounted_single_row` re-mounts a replacement controls `Horizontal`; a plain `mount()` appends, which would put the scope line above the controls it annotates, so it mounts `before=` the scope Static and re-binds `_batch_scope_statics`; a shape mismatch on the reason line falls back to the full rebuild like the effects/context lines already did.

Test-harness note worth keeping: `app.export_screenshot()` returns SVG in which every space is `&#160;` and each styled run is its own `<text>` element, so `assert "some sentence" in app.export_screenshot()` fails against a screen that paints the sentence perfectly. The pre-existing check in `test_approval_row_information_budget.py` only ever looked for a single word, which hid this.

Modified: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `tldw_chatbook/Agents/raw_shell_tool_provider.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated `tldw_chatbook/css/tldw_cli_modular.tcss`), `Tests/UI/test_chat_approval_card.py`, `Tests/UI/test_approval_row_information_budget.py`, `Tests/UI/test_console_mcp_approval.py` (the Select-width / row-height / tooltip pins this change owns).
<!-- SECTION:NOTES:END -->
