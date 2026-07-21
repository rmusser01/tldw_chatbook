---
id: TASK-450
title: 'Console rail: flush-left width-aware two-line conversation names'
status: Done
assignee: []
created_date: '2026-07-21 07:07'
updated_date: '2026-07-21 21:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Left-rail conversation names are hard-truncated at 20 chars and indented by a marker prefix. Implement the approved design: flush-left names wrapping to up to 2 width-aware lines, metadata line kept, guarded relabel on width change. Spec: Docs/superpowers/specs/2026-07-20-console-rail-conversation-row-layout-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Conversation names render flush left with no marker prefix
- [x] #2 Names wrap to at most 2 lines at the rail's measured width and ellipsize only when 2 lines are insufficient
- [x] #3 Metadata line renders as the row's final line and is cell-truncated to the row budget
- [x] #4 Precomputed list heights match rendered row heights for mixed wrapped/badge rows
- [x] #5 No recompose oscillation when the rail scrollbar toggles or the terminal resizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-20-console-rail-row-wrap.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reworked ConsoleWorkspaceContextTray so left-rail conversation rows render flush-left and wrap to up to two width-aware lines instead of hard-truncating at 20 chars.

Approach: two pure cell-aware helpers (wrap_console_conversation_title, truncate_console_row_cells, measured via rich.cells.cell_len) feed BOTH label building and the precomputed container heights, so they cannot diverge (AC#4). The marker prefix and 20-char truncation were removed; selection is shown by the existing selected-row CSS class alone (AC#1). Row/star/list heights derive from the same wrap result at the same budget (AC#4). The wrap budget is measured from the tray's content_region.width inside _fit_height_to_content (_maybe_relabel_for_width), equality-guarded so a recompose fires only on a genuine width change (AC#2, AC#5). Because that recompose discards an out-of-band-mounted alias button, the tray posts a Relabeled message and chat_screen re-runs its idempotent alias worker. Two CSS guards: .console-conversation-browser-row-line { height: auto } (fixes latent Horizontal 1fr equal-division) and #console-left-rail-body { scrollbar-gutter: stable } (decouples wrap width from scroll state, killing the flap loop) (AC#5). Metadata line cell-truncated to the same budget (AC#3).

Verification: 12 wrap unit tests + real-Textual-app mount tests at two terminal widths; full UI+Chat sweep zero novel regressions vs origin/dev; opus whole-branch review Ready-to-merge (0 Critical/Important); live tmux verification of layout, alias-button survival, metadata truncation, and no-oscillation resize (235<->110 cols). Live 2-line NAME wrap deferred to real-app tests (native browser only lists console-send-path conversations).

Files: tldw_chatbook/Widgets/Console/console_workspace_context.py (helpers, row rework, relabel), tldw_chatbook/UI/Screens/chat_screen.py (Relabeled handler), tldw_chatbook/Workspaces/display_state.py (constant comment), tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_console_conversation_row_wrap.py (new) + 3 updated test files.
<!-- SECTION:NOTES:END -->
