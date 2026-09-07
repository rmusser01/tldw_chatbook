---
id: TASK-1232
title: 'Fleet capability discoverability: Help, coach-mark, footer'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 09:30'
updated_date: '2026-07-28 19:17'
labels:
  - console
  - ux
  - docs
  - uat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F2: nothing at rest communicates that each Console tab runs its own agent in parallel under a cap. F1 Help covers panes/transcript/composer only (zero mentions of agents, approvals, workspaces, parallel runs); the footer omits Alt+W and Alt+1..9; the capability teaches itself only after accidental use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F1 Help gains an Agents section (tabs=agents, cap + where to change it, approval flow, marker legend, Alt+W / Alt+1..9).
- [x] #2 A dismissible coach-mark on first second-tab creation states the parallel model and the cap; shown until acknowledged, never again after dismissal (persisted). RULING (fleet-UX review round 1): dismiss-persisted stands over shown-once-ever -- an unacknowledged banner reappearing on a genuine trigger repeat (the 2-count transition only, no nag loop) is better UX than a one-shot the user could miss forever.
- [x] #3 Footer (or Help) lists the workspace/tab-jump hotkeys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the F1 help content source (CONSOLE_WORKBENCH_SHORTCUT_GROUPS / WorkbenchHelpState in chat_screen.py + UI/Workbench/help.py) and extend WorkbenchHelpState with a generic notes/notes_heading block; add an Agents section (tabs=agents, live cap via ConsoleChatController.max_parallel_runs, approval flow, a clearly-marked marker-legend constant for task-1233 to reuse, Alt+W/Alt+1..9/Ctrl+T/Ctrl+K hotkeys, screen-scope caveat).
2. Add a one-time dismissible coach-mark banner to ConsoleSessionSurface (hidden Horizontal under the tab strip, always composed, toggled via show/hide methods -- not a mount-time write). Hook the show condition into _sync_console_native_session_tabs via a session-count transition check (fires once when count first goes from <2 to 2, covering every tab-creation path uniformly). Persist the seen-flag on dismiss via the established console.onboarding config seam (mirrors _console_first_send_completed/_record_console_first_send).
3. Judge footer crowding: the AppFooterStatus footer is a single-line, non-wrapping Static already ~120 chars for its 7 hints at the narrowest width the suite tests Console against (80 cols) -- adding Alt+W there increases an already-overflowing line. Alt+W/Alt+1..9 are Real BINDINGS with show=True already, confirming the footer's omission is NOT a show=False bug (this app never mounts Textual's native Footer widget for Console; the footer is a bespoke AppFooterStatus driven by a flat shortcuts tuple). Satisfy AC#3 via Help only.
4. Write TDD coverage: pure-function content tests, a real end-to-end F1-help test proving the live cap is read, and coach-mark tests (first second-tab creation shows it, dismiss persists + hides, third tab doesn't reshow, and a real-config-seam test proving the flag survives a simulated restart).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
F1 Help now has an "Agents" section: WorkbenchHelpState gained a generic notes/notes_heading pair (render_text() slots it between Actions and Shortcuts), reusable by any Workbench route. chat_screen.py's action_show_workbench_help reads the LIVE ConsoleChatController.max_parallel_runs and passes _console_workbench_agents_notes(cap) -- tabs=agents, the cap + "Settings > Console Behavior", the approval one-liner, and CONSOLE_FLEET_MARKER_LEGEND (a standalone constant so task-1233's glyph-legend work can reuse it verbatim), plus the screen-scope caveat. A new "Agents & fleet" shortcut group covers Alt+W/Alt+1..9/Ctrl+T/Ctrl+K.

Coach-mark (AC#2): ConsoleSessionSurface always composes a hidden `#console-fleet-coachmark` Horizontal (Static + compact "✕" dismiss Button) under the tab strip -- mirrors the composer's #console-clear-attachment "always mount display:none, toggle later" pattern so visibility is state-driven, not a mount-time write. `_maybe_show_fleet_coachmark` hooks into `_sync_console_native_session_tabs` (the common choke point every tab-creation path -- Ctrl+T, the "+New tab" button, workspace auto-tabs, Personas Start Chat -- already runs through) and fires on the session-count TRANSITION to exactly 2, seeded from whatever count the screen first observes so a restore that starts at 2+ sessions is never mistaken for a "creation". The seen-flag persists on DISMISS (not on show, so an unacknowledged banner can reappear) via the same console.onboarding config seam _record_console_first_send already established (manual nested-dict app_config read/write + a save_setting_to_cli_config("console.onboarding", "fleet_coachmark_seen", True) background worker) -- get_cli_setting was deliberately avoided since it does not resolve a dotted "console.onboarding" section (documented prior-program trap).

Footer (AC#3): CORRECTION (round 1 review, Minor a) -- the original notes here overclaimed "alt+w/alt+1..9 already carry show=True"; only alt+w does (BINDINGS has it `show=True`), alt+1..9 are all `show=False`. The conclusion is unchanged despite the correction: Console never mounts Textual's native Footer widget at all, so Binding.show is not the mechanism in play either way -- the visible footer is the bespoke AppFooterStatus driven by the flat CONSOLE_WORKBENCH_SHORTCUTS tuple, which ignores Binding.show entirely regardless of any binding's show value. That tuple already renders ~120 chars for 7 hints as a single non-wrapping Static with no truncation, at the narrowest width (80 cols) this suite tests Console against -- so an 8th hint would just extend an already-overflowing line. Per the task's own permitted fallback ("Alt+W minimum in Help"), AC#3 is satisfied via the new Help shortcut group only; the footer tuple is unchanged.

Tests: new Tests/UI/test_console_fleet_discoverability.py (7 tests) -- pure-function Agents-notes content, the grouped-shortcuts hotkey list, a full ChatScreen F1-help end-to-end test proving the live cap (monkeypatched to a non-default 7) reaches the rendered panel body, coach-mark show-on-first-second-tab, dismiss-hides-and-persists, no-reshow-on-a-third-tab, and a real-config-seam test (tmp TLDW_CONFIG_PATH + save_setting_to_cli_config, not a mock) proving the flag survives a simulated app restart. Ran together with Tests/UI/test_workbench_focus_help.py, Tests/UI/test_screen_footer_hints.py, Tests/UI/test_console_parallel_runs.py, and Tests/UI/test_console_session_tab_strip.py in one foreground call: 57 passed, 1 pre-existing failure (test_library_registration_updates_the_screens_own_footer -- confirmed failing identically on the unmodified HEAD commit via git stash, unrelated to this change).

Files: tldw_chatbook/UI/Workbench/help.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Widgets/Console/console_session_surface.py, Tests/UI/test_console_fleet_discoverability.py (new).

### Round 1 (fleet-UX review): Critical fix + ruling + two minors

CRITICAL -- `WorkbenchHelpPanel`'s `#workbench-help-panel` was a plain `Vertical` with NO CSS anywhere, so it inherited Textual's own `Vertical` defaults (`height: 1fr`, `overflow: hidden hidden`): the overlay silently filled the screen and then HARD-CLIPPED anything past the fold with no scrollbar. The new Agents section and the Alt+W/Alt+1..9 hotkeys (AC#3's sole mechanism) were unreachable at every realistic terminal size; only the original test's exact 160x48 happened to fit all ~44 lines. Fixed by giving the body its own `VerticalScroll` (`#workbench-help-scroll`) with a bounded/centered panel and a pinned Close button below it, mirroring `ConsoleScopePickerModal`'s scroll-body-plus-fixed-footer shape. Styling lives in TWO places by design (same KEEP-IN-SYNC discipline `AppFooterStatus` already established): the production bundle (`css/components/_workbench.tcss`, rebuilt into `tldw_cli_modular.tcss`) carries the app's `$ds-*` design tokens and wins by origin in the real app; `WorkbenchHelpPanel.DEFAULT_CSS` bakes in the same STRUCTURAL rules using only built-in Textual variables (`$primary`/`$surface`) so the panel is still correctly bounded/scrollable in stylesheet-less test harnesses (`WorkbenchHelpPanel` is shared infra invoked from many screens' own lightweight harnesses, not just Console's -- discovered when the first version of the reachability test, run under the bare `ConsoleHarness`, showed the bundle-only CSS never applying at all).

Test rewrite: replaced the flawed single 160x48 test with a parametrized `test_console_f1_help_is_scrollable_and_reachable_at_realistic_sizes[80x24 / 160x40]` that drives real SVG compositor captures (`app.export_screenshot()`, rejoining per-segment `<text>` nodes -- mirrors `test_workbench_visual_snapshots.py`'s established idiom) instead of reading `Static.renderable` directly, which is blind to scroll-clipping and was the exact blind spot that let this bug ship unnoticed. Asserts: `scroll.max_scroll_y > 0` and a visible scrollbar at both sizes; the Close button's region fits on screen and is compositor-visible before AND after scrolling (pinned, not part of the scrolled body); the Agents section and marker-legend components render at rest (no scrolling needed); Alt+W/Alt+1..9 are confirmed ABSENT at rest (proving the fold problem is real) and reachable after `scroll_end()` (the reviewer's literal ask); and the borderline last Agents note ("Leaving Console cancels...", which needs a 1-row scroll at 80x24 but none at 160x40 -- multi-row wraps above it shift its position by a variable amount) is checked via a small scroll-checkpoint scan rather than a hardcoded position. Also discovered along the way: the full `CONSOLE_FLEET_MARKER_LEGEND` string cannot be asserted verbatim against wrapped, compositor-rendered text -- at these widths the Static wraps it across two rows and the hard line-wrap swallows the space at the break (observed: "...finished · ✗" / "failed — clears..."); asserted its components instead.

RULING on the dismiss-persisted-vs-shown-once judgment call: STANDS as implemented. AC#2's wording is now honest about this ("shown until acknowledged; never again after dismissal (persisted)") rather than "one-time", which had read as "shown exactly once, ever."

Minor (a): corrected a false claim in the AC#3 notes above -- alt+1..9 do NOT carry `show=True` (only alt+w does; alt+1..9 are `show=False`). The conclusion is unaffected: Console's footer is not driven by `Binding.show` at all regardless.

Minor (b): `_console_workbench_agents_notes` now pluralizes correctly -- "Up to 1 run in parallel" (not "1 runs"), since cap=1 is a supported floored value (`MIN_CONSOLE_MAX_PARALLEL_RUNS`). New pure-function test `test_console_workbench_agents_notes_pluralizes_a_cap_of_one` pins this.

Files (round 1 additions): tldw_chatbook/UI/Workbench/help.py (DEFAULT_CSS + VerticalScroll body), tldw_chatbook/css/components/_workbench.tcss (new `#workbench-help-*` rules), tldw_chatbook/css/tldw_cli_modular.tcss (rebuilt via `python3 tldw_chatbook/css/build_css.py`), tldw_chatbook/UI/Screens/chat_screen.py (pluralization fix), Tests/UI/test_console_fleet_discoverability.py (rewritten reachability test + new pluralization test).
<!-- SECTION:NOTES:END -->
