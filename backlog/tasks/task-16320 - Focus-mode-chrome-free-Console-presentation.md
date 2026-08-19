---
id: TASK-16320
title: 'Focus mode: chrome-free Console presentation'
status: Done
assignee: []
created_date: '2026-08-16 13:34'
updated_date: '2026-08-19 19:07'
labels:
  - ui
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a 'focus mode' that presents only the Console content — message stream, composer, and a one-line status bar — hiding the MainNavigationBar and workbench header. Recreates a claude-code/codex-style UI for zen coding on desktop and phone use over --serve without fine pointer/touch affordances. Zen-not-kiosk with one navigation rule: any navigation to a non-chat route exits focus; ctrl+shift+f re-enters from anywhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console hides MainNavigationBar and DestinationHeader while focus mode is active; the one-line AppFooterStatus status bar remains visible
- [x] #2 --focus CLI flag and [general] focus_mode config launch straight into the chrome-free Console (first-run onboarding still wins)
- [x] #3 ctrl+shift+f is an app-level toggle with no conflicts with existing bindings; toggling on from a non-chat screen navigates to the Console and enters focus
- [x] #4 Any navigation to a non-chat route (destination hotkey or palette) exits focus mode and the destination mounts with normal chrome
- [x] #5 Footer shortcut context advertises the focus toggle truthfully in both states (focus / exit focus)
- [x] #6 Default presentation is unchanged when focus mode is off (existing console contract tests pass)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design approved + reviewed (2026-08-16). Spec: Docs/superpowers/specs/2026-08-16-focus-mode-design.md — ADR: backlog/decisions/067-focus-mode-chrome-free-console.md — Implementation plan: Docs/superpowers/plans/2026-08-16-focus-mode.md (7 TDD tasks). Ready to execute.

Implemented 2026-08-19 via the 7-task TDD plan (commits 5e23bfd5e, d9f595b79, 902034514, 365e78346, 252665ebe, a7ebecb99, plus this closeout). ADR-067 governs; no new ADR needed.

**Approach:** App-level `focus_mode` flag on `TldwCli` set at startup from `[general] focus_mode` config and/or the `--focus` CLI flag (flag wins; first-run onboarding beats both); `_resolve_initial_shell_route` forces the Console route when focus is requested. The Console screen mirrors the flag onto a `-focus` CSS class (`ChatScreen.-focus MainNavigationBar` / `ChatScreen.-focus #console-workbench-header { display: none }` — display:none only, ADR-042) via idempotent `_apply_focus_chrome()`, called at mount and on toggle; the one-line `#screen-footer-status` is intentionally kept. `Ctrl+Shift+F` is an app-level binding (show=False) driving `action_toggle_focus_mode` → `_set_focus_mode`, which duck-types the content screen (no ChatScreen import in app.py — avoids the circular import the screen registry exists to prevent) and navigates to the Console when enabled elsewhere. Single exit rule: `_clear_focus_if_leaving_console` in `_handle_screen_navigation_locked` drops the flag on any non-chat navigation. The footer registration always carries a `("Ctrl+Shift+F", "focus"|"exit focus")` pair (label names the action the key performs — truthfulness, ADR-031), and the palette gains "Quick Actions: Toggle Focus Mode".

**Files touched:** `tldw_chatbook/config.py` (template key), `tldw_chatbook/app.py` (arg parser extraction, focus attrs, route resolution, toggle/exit helpers, binding, palette QuickAction), `tldw_chatbook/UI/Screens/chat_screen.py` (`_apply_focus_chrome`, mount call, footer pair), `tldw_chatbook/css/components/_agentic_terminal.tcss` + regenerated `tldw_chatbook/css/tldw_cli_modular.tcss`, `Tests/UI/test_focus_mode.py` (new, 18 tests), `Tests/UI/test_command_palette_providers.py` (fixture counts), `Tests/UI/test_console_workbench_contract.py` (footer fixture gained the focus pair), `Docs/User_Guide/console.md` ("Focus mode" subsection).

**Verification:** `.venv/bin/python -m pytest Tests/UI/test_focus_mode.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_scope_row.py -v` → 120 passed; palette/footer-hint neighbors (`test_command_palette_providers.py`, `test_app_footer_shortcut_context.py`, `test_chrome_ux_fixes.py`) → 81 passed.

**Deviations from plan:**
1. The CSS harness needed `CSS_PATH` pointed at the full bundled stylesheet plus `run_test(size=(120, 40))` — at run_test's 80x24 default the existing `-console-compact` rule (engages below 35 rows) hides the workbench header independently of focus, masking the rules under test.
2. The palette test asserts on `hit.text`, not the plan's `hit.display` — Textual 8's `Hit` is a dataclass without a `.display` attribute.
3. Task-1's line anchors were re-based to current dev: app.py has two `TldwCli()` construction sites and the `__main__` block parsed no args today; the extracted `_build_arg_parser()` is now parsed there too via a try/except SystemExit guard so a bare `python3 -m tldw_chatbook.app` with an unknown flag still runs.
4. `Tests/UI/test_command_palette_providers.py` fixture counts updated 4→5 (discover) / 5→6 (search) for the new QuickAction.
5. Task 7: the console footer contract fixture (`test_console_registers_footer_workbench_shortcuts`) gained `Ctrl+Shift+F focus | ` before the trailing generic `Ctrl+Q quit` — the truthfulness rule means the fixture matches the real registration, not the other way around.

**Note:** status flip to Done is deferred to the main checkout (this worktree does not own the backlog DB).
<!-- SECTION:NOTES:END -->
