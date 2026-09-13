# Media UX fix wave 4 — PR B (footer honesty + Escape/F6/Back) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Library ▸ Media footer tell the truth at its four lying seams, and make Escape, F6 and Back never strand keyboard focus in a text input.

**Architecture:** Footer chips are computed from the real state at the moment they render (focus, set position, bar presence) instead of static tuples; select-mode entry moves focus onto a row; the Reader's Escape ladder ends on rows, never on Inputs; the F6 content stop gets a visible border treatment; in the side-by-side layout the Reader stays live after Escape and the "‹ Back" control is not composed. Tests pin every chip and focus target on the production-CSS harness; painted-text where paint-over is the risk.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio; `LibraryProductionCSSHarness` (`Tests/UI/test_library_shell.py`), `ControlledDetailMediaService` / `_flow_app` (`Tests/UI/test_library_media_reader_flow.py`), `_painted` (`Tests/UI/test_library_media_render_fixes.py`).

**Spec:** `.impeccable/critique/2026-09-04T13-50-05Z__tldw-chatbook-ui-screens-library-screen-py.md` priority issues 3 and 4; tasks `backlog/tasks/task-31271 - …md` and `task-31272 - …md` (their Acceptance Criteria are binding).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave4-b`, branch `fix/media-wave4-b`, stacked on `fix/media-wave4-a` (PR #2378). Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`. Absolute paths everywhere; the shell cwd can be reset between calls.
- Run UI test files in separate processes. Compare any failure against the clean base (`git stash` / a detached worktree at the branch base) before calling it yours: `test_library_shell.py` carries known failures listed in `backlog/tasks/task-31249 - …md`.
- No new `logger.*` calls (each one forces `python scripts/check_persistent_diagnostic_inventory.py --write` + committing `Docs/security/production-diagnostic-inventory.json`).
- After editing any `BUNDLED_CSS` or `tldw_chatbook/css/components/*.tcss`: `python -m tldw_chatbook.css.build_css`, then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0), commit every regenerated file under `tldw_chatbook/css/`.
- The footer is the single source for F1 too (`_library_footer_shortcuts_for_current_state`, task-2858): change chips there, never in a second place.
- Widget-tier CSS (`BUNDLED_CSS`) loses to app-tier rules regardless of specificity; the app-global `*:focus` outline PAINTS OVER content (task-31221) — any new focus treatment must be proven with a painted-text assertion that content still paints.
- Textual key names in `pilot.press`: `]` = `right_square_bracket`, `[` = `left_square_bracket`, Escape = `escape`, F6 = `f6`, Space = `space`.
- Commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_011LebG4HPfSniVohbuXkU4n`. Backlog task files are flipped by the controller, not the implementer.

---

### Task 1: Footer honesty at four seams (task-31271)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `_library_media_escape_label` (~:39775, reads `#library-media-content-search-controls` from the DOM), `_close_library_media_find` (~:41117) and its callers in `action_library_media_viewer_back` (~:38536), `_toggle_library_media_select_mode` (~:24215; calls `_sync_library_canvas(self, "media")` which accepts `then=`), `LIBRARY_MEDIA_SELECT_SHORTCUTS` (:1161-1167), `check_action` for `library_media_toggle_row_selection` (~:27812), `_review_footer_entries` (:3667-3690, static, takes only `progress`), the Reader branch of `_library_footer_shortcuts_for_current_state` (~:3691), `_register_footer_shortcuts` (:9086), BINDINGS `l`/`c`/`t` (:1016-1018, `show=False`).
- Test: `Tests/UI/test_library_media_render_fixes.py` (append), `Tests/UI/test_review_set_walker.py` (append unit tests for `_review_footer_entries`).

**Interfaces:**
- Produces: `_review_footer_entries(progress, *, at_last: bool = False)` — when `at_last` and the set is not complete, the first entry is `("]", "finish review")` instead of `("]", "next in set")`; the Reader footer set gains `("l", "read later")`, `("c", "use in Console")`, `("t", "trash")` after the `]`/`[` entries (short words — the footer compacts at 100 cols, see task-31272's width note); Space's `check_action` returns `True` whenever select mode is active (the action itself stays a no-op off-row, so Space never falls through to the pane grip).
- Consumes: `_focus_library_media_items_pane()` (~:39745; prefers the selected row, else row 0, else the filter) as the `then=` hook of the select-mode sync.

- [ ] **Step 1: Write the failing tests** (one per seam; production-CSS harness, size (235, 52); reuse `_analysis_flow_host`, `_load_row_0`, `_walk_next` from `test_library_media_render_fixes.py`):
  - `test_footer_drops_close_find_after_escape_closes_the_bar`: open a row, press Find, wait for `#library-media-content-search`, press `escape`, pause twice; assert `"close find"` not in the label of the footer's esc chip — read it via `screen._library_footer_shortcuts_for_current_state()` (the F1 source) AND via the painted footer row (`_painted(host, footer.region)` where `footer = screen.query_one("#library-footer")` or the widget that renders the shortcuts — grep `esc` rendering in `_register_footer_shortcuts`).
  - `test_pressing_s_focuses_a_media_row_so_space_toggles_immediately`: open the media list, press `s`, pause twice; assert `screen.focused.has_class("library-media-row")`; press `space`; assert `screen._library_media_row_selection.count == 1`.
  - `test_space_in_select_mode_never_reaches_the_pane_grip`: enter select mode, focus the shell grip (`screen.query_one("#library-media-reader-shell").library_grip` — grep `LibraryMediaReaderShell` for the grip attribute/ids), press `space`; assert the Library pane is still open (`shell.effective_layout.library_open is True`) and no row toggled.
  - `test_review_footer_names_the_completion_gesture_on_the_last_item` (unit, walker file): `LibraryScreen._review_footer_entries("6 of 6 · 5 reviewed", at_last=True)[0] == ("]", "finish review")`; with `at_last=False` the first entry is `("]", "next in set")`; a complete progress (`"All 6 reviewed"`) is unchanged.
  - `test_reader_footer_advertises_l_c_t`: open a row; the Reader footer entries (via `_library_footer_shortcuts_for_current_state()`) contain keys `"l"`, `"c"`, `"t"`.
- [ ] **Step 2: Run them; each must fail for its seam's reason** (stale label present; focus not on a row / count 0; library pane collapsed or an AttributeError on the grip lookup you must fix in the test; `next in set` on the last item; `l`/`c`/`t` absent).
- [ ] **Step 3: Implement, seam by seam.** (a) In `action_library_media_viewer_back`'s find-close branch (and anywhere `_close_library_media_find` runs from a key/Escape), schedule `self.call_after_refresh(self._register_footer_shortcuts)` so the label recomputes after the recompose removes the bar. (b) In `_toggle_library_media_select_mode`, pass `then=self._focus_library_media_items_pane` to `_sync_library_canvas(self, "media", …)` on entry (not on exit). (c) `check_action("library_media_toggle_row_selection")` returns `True` whenever `_library_media_select_mode` (keep the stale/confirm/in-flight guards inside the action). (d) `_review_footer_entries(progress, *, at_last=False)`; every caller computes `at_last` from the live review progress (cursor is the last live index; use `review_progress`'s fields — grep `format_review_progress` callers at ~:3691 and the review banner code ~:38930). (e) Add the `l`/`c`/`t` entries to the Reader footer set with the short labels above.
- [ ] **Step 4: Run the new tests, then `Tests/UI/test_library_media_render_fixes.py`, `Tests/UI/test_review_set_walker.py`, `Tests/UI/test_library_media_reader_flow.py`, `Tests/UI/test_library_multiselect_media.py`, and `Tests/UI/test_library_shell.py -k "footer or select or shortcut"` (compare against the base for that subset).**
- [ ] **Step 5: Live in tmux 235x52** (seed 3 items via `MediaDatabase.add_media_with_keywords`; launch `PYTHONPATH=<worktree> <python> -m tldw_chatbook.app` under `tmux -L w4b`, sleep 14 in the SAME call; palette → `library` → Down → Enter; click labels with `python3 /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/b8e66fcb-3445-4682-b238-8dc5e07235e2/scratchpad/click.py w4b "<label>"`; NEVER end a turn with tmux running — sleeps stay inside the call; kill the server and soft-delete the seeds at the end): Find → Escape → footer has no `esc close find`; `s` → Space toggles row 0 at once; Review these → `]` to the last item → footer shows `] finish review`; Reader footer shows `l`/`c`/`t`.
- [ ] **Step 6: Commit** `fix(library): footer tells the truth at its four seams (task-31271)`.

---

### Task 2: Escape, F6 and Back never strand focus in a text input (task-31272)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `action_library_media_viewer_back` (~:38536; the Items-pane branch calls `self._focus_library_rail_action("#library-search-input")` — that is the trap), `_library_media_escape_label` (~:39775; eight labels today), `_exit_library_media_viewer` (~:38493), the `more_open` branch and `check_action("library_media_viewer_back")` (grep), `_MEDIA_WORKBENCH_FOCUS_TARGETS` (content-text is the first viewer stop; its focus is invisible since task-31221 suppressed the outline).
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` (~:225 `yield Button("‹ Back", id="library-media-back", compact=True)`) — compose it only when the shell layout is NOT side-by-side (the viewer needs a `back_visible: bool` constructor flag threaded from the screen; add it to `_sync_library_media_viewer_state`'s compare/assign, the #2351 trap).
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (library-media section, near `#library-media-viewer-content { … border: solid $ds-grid-line …}` ~:3599) — add `#library-media-viewer-content:focus-within, #library-media-viewer-content-text:focus { border: solid $accent; }` (tint the EXISTING border; no outline, no new geometry) — and the same in `LibraryMediaViewer.BUNDLED_CSS` if the viewer has one (grep), else in the screen's bundled sheet.
- Test: `Tests/UI/test_library_media_render_fixes.py` (append), `Tests/UI/test_library_media_reader_flow.py` (append; `test_media_global_f6_reaches_content_scroller` is the precedent for F6).

**Interfaces:**
- Produces: Escape ladder in the side-by-side layout: Reader region → the loaded Items ROW (`_focus_library_media_items_pane`); Items region → the ACTIVE RAIL ROW (`#library-row-browse-media`, via `_focus_library_rail_action`), never `#library-search-input`; the Reader stays live (`_library_media_view` remains `"viewer"`, `]`/`[` still bound). In the compact/stacked layout the existing "back to list" exit is unchanged. Escape label vocabulary reduced to four: `close` (find bar / More menu / an armed delete or edit cancel), `focus Items`, `focus Library`, `back` (compact exit) — record the mapping table in the task notes.
- Consumes: Task 1's `_register_footer_shortcuts` refresh after Escape.

- [ ] **Step 1: Write the failing tests:**
  - `test_escape_from_the_reader_lands_on_the_loaded_row_then_the_rail_row`: open row 0 (size 235x52, side-by-side); press `escape` → `screen.focused.has_class("library-media-row")` and `screen._library_media_view == "viewer"` and `screen.check_action("library_media_next_item", ())` still gated only by neighbour existence (not by view); press `escape` again → `screen.focused.id == "library-row-browse-media"` (never an `Input`).
  - `test_escape_closes_the_more_menu_from_any_reader_focus`: open a row, press `#library-media-reader-more`, move focus to `#library-media-viewer-content`, press `escape` → `screen._library_media_reader_session.more_open is False`.
  - `test_f6_content_stop_is_visible_and_content_still_paints` (painted-text): open a row, press `f6` until `screen.focused.id in {"library-media-viewer-content-text", "library-media-viewer-content"}` (cap 6 presses); assert the box's border color is the accent (`screen.query_one("#library-media-viewer-content").styles.border_top[1]` compared to the unfocused color captured before), and `_painted(host, box.region)` still contains the first content line.
  - `test_back_button_is_not_composed_in_the_side_by_side_layout`: at 235x52 `not screen.query("#library-media-back")`; at (100, 30) (compact) `screen.query("#library-media-back")` exists.
  - `test_escape_labels_are_one_of_four`: drive the states (plain Reader, Find open, More open, armed delete via `t`, Items focus, rail focus) and assert each `_library_media_escape_label()` ∈ {"close", "focus Items", "focus Library", "back"}.
- [ ] **Step 2: Run them; confirm each fails for the documented reason.**
- [ ] **Step 3: Implement** per the Interfaces block. Keep `_exit_library_media_viewer` for the compact layout and for the Back button where it still renders. Do not change the walker or review-set code.
- [ ] **Step 4: Run** the new tests, `test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_media_side_by_side.py`, `test_library_review_round_t21116.py` and `test_library_per_click_recompose_t21116.py` (both carry known base failures — compare against base), `test_library_shell.py -k "escape or back or f6 or focus"` (compare against base).
- [ ] **Step 5: Live in tmux 235x52 and 100x30**: Escape, Escape from the Reader ends on the Media rail row (footer never `typing in field`); More → Escape closes it; F6 to content shows the tinted border; 100x30 still shows a Back control and Escape returns to the list.
- [ ] **Step 6: Commit** `fix(library): Escape, F6 and Back never strand focus in an input (task-31272)`.
