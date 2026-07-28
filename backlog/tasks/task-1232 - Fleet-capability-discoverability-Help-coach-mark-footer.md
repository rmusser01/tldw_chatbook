---
id: TASK-1232
title: 'Fleet capability discoverability: Help, coach-mark, footer'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 09:30'
updated_date: '2026-07-28 18:43'
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
- [x] #2 A one-time dismissible coach-mark on first second-tab creation states the parallel model and the cap.
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

Footer (AC#3): confirmed alt+w/alt+1..9 already carry show=True in BINDINGS, so flipping show flags was NOT the fix -- Console never mounts Textual's native Footer widget at all; the visible footer is the bespoke AppFooterStatus driven by the flat CONSOLE_WORKBENCH_SHORTCUTS tuple, which ignores Binding.show entirely. That tuple already renders ~120 chars for 7 hints as a single non-wrapping Static with no truncation, at the narrowest width (80 cols) this suite tests Console against -- so an 8th hint would just extend an already-overflowing line. Per the task's own permitted fallback ("Alt+W minimum in Help"), AC#3 is satisfied via the new Help shortcut group only; the footer tuple is unchanged.

Tests: new Tests/UI/test_console_fleet_discoverability.py (7 tests) -- pure-function Agents-notes content, the grouped-shortcuts hotkey list, a full ChatScreen F1-help end-to-end test proving the live cap (monkeypatched to a non-default 7) reaches the rendered panel body, coach-mark show-on-first-second-tab, dismiss-hides-and-persists, no-reshow-on-a-third-tab, and a real-config-seam test (tmp TLDW_CONFIG_PATH + save_setting_to_cli_config, not a mock) proving the flag survives a simulated app restart. Ran together with Tests/UI/test_workbench_focus_help.py, Tests/UI/test_screen_footer_hints.py, Tests/UI/test_console_parallel_runs.py, and Tests/UI/test_console_session_tab_strip.py in one foreground call: 57 passed, 1 pre-existing failure (test_library_registration_updates_the_screens_own_footer -- confirmed failing identically on the unmodified HEAD commit via git stash, unrelated to this change).

Files: tldw_chatbook/UI/Workbench/help.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/Widgets/Console/console_session_surface.py, Tests/UI/test_console_fleet_discoverability.py (new).
<!-- SECTION:NOTES:END -->
