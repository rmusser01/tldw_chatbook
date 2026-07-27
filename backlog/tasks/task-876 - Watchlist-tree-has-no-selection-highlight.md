---
id: TASK-876
title: The watchlist tree does not show which node the screen is scoped to
status: Done
assignee: []
created_date: '2026-07-26 23:20'
updated_date: '2026-07-27 15:08'
labels:
  - watchlists
  - ui
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`WatchlistTree` renders its nodes as buttons and never reads `tree_scope`, so nothing in the left rail indicates which node the centre is currently scoped to. The Feeds region's heading (`Feeds in Morning AI Brief (1)`) is the only feedback the user gets.

That was tolerable while the scope only drove a breadcrumb. Phase C made it drive what the Feeds region renders and what "Stage in Console" sends, so the tree is now the primary navigation control for the screen — and it gives no sign of its own state.

Two related gaps found in the same review, worth folding in:

- The panes have no selected-row styling either. `SourcesPane`, `RunsPane` and `NotificationsPane` rely entirely on Textual's stock DataTable focus cursor, which is a focus affordance rather than a selection indicator — it always sits somewhere, including on rows the screen does not consider selected. Phase C deliberately declined to move it, because relocating it to row 0 on a tree move would assert a different wrong selection rather than fix one.
- `_load_tree_data` logs at debug where every sibling loader on the screen calls `notify(severity="error")`. A real database failure therefore renders identically to "you have zero watchlists" — two empty roots and no message. The tree is its own only error surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The tree node matching the current `tree_scope` is visually distinguished from its siblings
- [x] #2 The highlight follows a breadcrumb promotion as well as a direct tree click, since both move the scope through `_apply_tree_scope`
- [x] #3 The highlight survives a section switch and a rail toggle, both of which rebuild the tree from screen-held state
- [x] #4 Selected rows in the Sources, Runs and Notifications panes are distinguishable from merely focused ones
- [x] #5 A failure inside `_load_tree_data` is surfaced to the user distinguishably from an empty result
- [x] #6 Tests pin each highlight against the production stylesheet, not a bare `App`
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a screen-owned WatchlistTree.active_scope reactive (recompose=True, seeded via set_reactive like expanded/active_tag) that compose() reads to mark the node matching tree_scope with is-active. _build_tree_pane seeds it from self.tree_scope (covers a rebuild: section switch, rail toggle -- AC #3). watch_tree_scope -- already the single reconciliation point for both a real tree click and a breadcrumb promotion -- also pushes the new scope into the already-mounted tree instance (neither path rebuilds it on its own), covering AC #2 without merging tree_scope/selected_scope, which stay deliberately separate per the existing screen comment.

CSS: added a scoped .watchlist-tree .watchlist-tree-{root,watchlist,source}.is-active block to _watchlists.tcss (background/color/bold, plus a :focus guard matching the Watchlists tab-strip/MCP/Lab idiom), modelled on that tab-strip fix per the task brief. Unlike LabModeStrip/the tab strip, these buttons are compact=True, which already forces border:none !important, so there is no border-clipping risk here -- confirmed by temporarily reverting the CSS and rebuilding the bundle: the dedicated production-stylesheet test fails on the background/colour assertion (not a border/height one), then passes again once restored.

Sources/Runs/Notifications: selected-row highlighting uses Rich's own terminal-agnostic "reverse bold" Text style baked into the row's cells (same idiom as snippet_editor.py's _WHITESPACE_MARKER_STYLE and library_media_viewer.py's match highlighting), since a DataTable cell cannot reference Textual CSS variables. NotificationsPane's selected_notification is already recompose=True, so the style is applied entirely in compose(). SourcesPane/RunsPane's selected_source/selected_run are deliberately NOT recompose=True (a selection must not rebuild the table -- would lose scroll/cursor position), so each gained a small _update_selection_highlight that reverts the old row and re-styles the new one via DataTable.update_cell, tracking the last-highlighted row key as a plain (non-reactive) attribute.

_load_tree_data now captures app_instance.notify the same way every sibling loader on the screen does and calls notify(..., severity="error") on any exception, in addition to the existing debug log -- so a real DB failure no longer renders identically to "zero watchlists."

AC #6: the tree highlight is CSS-class-based, so it needed (and got) a dedicated production-stylesheet test (test_watchlists_tree_selection_is_visually_distinct_against_the_bundle in test_destination_visual_parity_correction.py), which fails without the CSS fix (verified) and asserts resolved .styles.background/.styles.color differ, plus a render_strips() check that the label itself paints, not just render_line(). The three DataTable highlights are Rich-Text-embedded and CSS-independent by construction, so a bare-App test is not the same false-negative risk it is for a CSS class rule; even so, each pane also got one production-stylesheet integration test (test_{sources,runs,notifications}_pane_selected_row_renders_reverse_video_under_the_bundle) confirming the highlight actually paints as reverse video once real CSS/theme are in the loop, alongside bare-App unit tests in Tests/Watchlists/test_watchlists_*_pane.py covering reselect-moves-the-highlight and clear-removes-it.

Found but explicitly OUT OF SCOPE, not fixed: SourcesPane's #sources-toolbar claims nearly all of the pane's vertical budget inside the real Watchlists destination shell (measured 33 of 34 rows at 160x60), leaving its DataTable only 1 visible row regardless of terminal size. This blocked an original attempt at a two-row focus-vs-selection compositor test and is unrelated to this task's CSS-vs-bare-App concern; worth a follow-up task.

Files changed: tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py, tldw_chatbook/UI/Screens/watchlists_collections_screen.py, tldw_chatbook/UI/Watchlists_Modules/{sources_pane,runs_pane,notifications_pane}.py, tldw_chatbook/css/features/_watchlists.tcss (+ regenerated tldw_cli_modular.tcss). Tests: Tests/Watchlists/test_watchlist_tree.py, test_watchlists_collections_screen.py, test_watchlists_sources_pane.py, test_watchlists_runs_pane.py, new test_watchlists_notifications_pane.py, and Tests/UI/test_destination_visual_parity_correction.py.
<!-- SECTION:NOTES:END -->
