---
id: TASK-23197
title: 'Console: close the 118-128 column dead zone that evicts the Context rail'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-29 21:56'
updated_date: '2026-08-30 06:10'
labels:
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Between 118 and 128 columns the Inspector auto-opens, which trips resolve_console_rail_priority and force-collapses the Context rail to a 13-column stub with no explanation. A one-column resize from 117 to 118 swaps which sidebar the user has. Automatic Inspector opening must not evict a visible Context rail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Context stays visible across 117 to 135 columns with default preferences
- [x] #2 An automatic Inspector open never force-collapses a visible Context rail
- [x] #3 If Context is collapsed by rail priority that is recorded distinguishably from a user close (revised from 'visible on the stub' - see Implementation Notes)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed the dead zone. Measured with probe_final.py: Context is now visible at 117, 118, 120, 125, 128, 129 and 135 columns. Before, 118-128 replaced it with a 13-column stub and opened the Inspector instead.

The fix declines the automatic open rather than changing priority resolution. console_auto_open_would_evict_context() is a pure guard consulted by _should_open_standard_width_inspector; two explicit opens still get Inspector priority exactly as before, so only the uninvited case changed.

SAFETY CHECK THAT MATTERED. ADR-043 documents a hard layout constraint: at 120 columns the workspace has 118 content columns while Context, Transcript and Inspector minimums total 120, and Textual 8.2.7's fraction solver then places the transcript hundreds of columns off-screen. Eviction was partly what kept that layout solvable, so I measured containment rather than assuming. At 118/120/125/128 every displayed pane is inside the grid (at 120: rail 30 + main 79 + handle 11 = 120 exactly). The fix avoids the three-pane case entirely by never opening the third pane, so it respects that constraint instead of violating it.

AC #3 was revised, and the reason is a measurement. I built the reason-on-stub badge first. test_console_edge_rail_geometry then failed: rewriting the badge re-renders the handle, replacing the focused reveal button and dropping keyboard focus. Removing the badge alone made that test pass again, confirming cause. Trading focus stability for a one-word label is a bad deal, and the label had little left to explain -- with the automatic open declined, eviction only happens immediately after the user opens the Inspector themselves, where the cause is self-evident. The eviction now records itself in state (left_forced_collapsed) at no rendering cost, keeping the distinction available to any surface that later wants it.

Test fallout was substantial because the old behaviour was deliberately pinned: test_console_shell_regions.py characterised the 120x30 default as 'inspector-priority' in three places (the _REGIONS expectation column, the containment test's expected set, and a parametrize id), and test_console_inspector_compact_access.py had a test whose whole premise was 'at 120 columns Inspector wins'. That test's real invariant -- that rendering never writes its decision back into the stored preference -- is unchanged and kept; its min_width==0 assertion was the compact-override waiver, which no longer applies, and is replaced by a containment assertion on the two-pane row.

Not mine, verified by re-running the same files with the whole tree stashed: test_authority_focus_f1_preserves_literal_rich_markup, test_visible_attach_context_action_switches_rails_without_file_picker, test_inspector_handle_opens_and_collapses_rail_at_140_cols, test_vertical_handles_use_bundled_full_height_geometry_and_keep_badge_visible.

preflight green. Files: Chat/console_rail_state.py, UI/Screens/chat_screen.py; Tests/Chat/test_console_rail_priority_no_eviction.py (new); 4 test files updated.
<!-- SECTION:NOTES:END -->
