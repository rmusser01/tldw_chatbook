---
id: TASK-39132
title: "Crossing 34/35 rows restyles the Console's whole 528-node shell for a class whose rules style six nodes"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/css/features/_console.tcss
- Tests/UI/test_console_projection_restyles.py
dependencies:
- TASK-38950
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. Line numbers are dev's. TASK-38950 (round-11 branch fu/r11-console) scopes the header breakpoint's restyle; this one is separate.

**The flip.**
- `ChatScreen._adapt_console_shell_to_height` (`chat_screen.py:21656-21657`) runs `shell.set_class(height < 35, "-console-compact")` (`CONSOLE_COMPACT_HEIGHT_ROWS = 35`, `:745`).
- In Textual 8 a class change restyles the node's whole subtree.
- The only rules keyed on the class (`css/features/_console.tcss:2323-2349`) style six nodes.

**Measured.** A probe in dev's `test_console_projection_restyles.py` harness at 160x40:

| Height change | Crosses 34/35? | Nodes restyled |
|---|---|---|
| 40 → 38 | no | 0 |
| 38 → 34 | yes | 528 |
| 34 → 30 | no | 0 |
| 30 → 36 | yes | 528 |

528 is every node in `#console-shell`, of 555 on the screen. That is the same shape as TASK-38950's breakpoint flip, paid on every resize across 34 rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A crossing in either direction restyles only the nodes the `-console-compact` rules style, plus any subtree their inherited properties need, counted in a test that is red on dev (528 of 528)
- [ ] #2 Styles after each flip equal those of a full restyle
- [ ] #3 Height changes that do not cross the line still restyle nothing, and readers of the class (for example `ChatApprovalCard`) still see it
<!-- AC:END -->
