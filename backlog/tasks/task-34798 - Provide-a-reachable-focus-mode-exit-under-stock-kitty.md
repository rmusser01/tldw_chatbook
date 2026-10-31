---
id: TASK-34798
title: Provide a reachable focus-mode exit under stock kitty
status: To Do
assignee: []
created_date: '2026-10-11 23:33'
updated_date: '2026-10-11 23:34'
labels:
  - console
  - ux
  - terminal-compat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-09-24 report describes a Console launched with focus mode enabled under stock kitty 0.47.1. Its advertised Ctrl+Shift+F focus toggle was consumed by kitty, so that visible exit hint did not work. Navigation away from the Console and the palette still provide other exits.

Current dev still binds and advertises this chord. The application already has mounted key-dispatch tests; those cannot prove delivery through a terminal's own keymap. Provide a discoverable focus exit that works with the reported stock mapping while preserving existing focus behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Under stock kitty, at least one advertised, reachable affordance exits focus mode without requiring the user to override a terminal-intercepted shortcut.
- [ ] #2 The focus footer describes the reachable exit accurately under ADR-031; it does not present a known intercepted chord as the only visible route.
- [ ] #3 The chosen exit has mounted application key/action regression coverage, plus recorded stock-kitty delivery evidence. Existing Ctrl+Shift+F Pilot coverage is retained; Pilot alone is not terminal-delivery evidence.
- [ ] #4 The current Ctrl+Shift+letter bindings are checked against the reported terminal's stock mappings, with the surviving/colliding set documented in the relevant lesson and ADR-071 amendment or Console user guide.
- [ ] #5 Chrome suppression, clearing focus on non-Console navigation, the palette toggle, and ADR-210 region ownership retain their current behavior.
<!-- AC:END -->

## Historical incident and current review

The reporter's 2026-09-24 environment was kitty 0.47.1 without keymap overrides. This PR review did not rerun that historical binary or the reporter's private profile.

Kitty's stock Ctrl+Shift+F mapping invokes `move_window_forward`, which swaps kitty windows. It is not an OS-window frontmost action. `--allow-fallback=shifted,ascii` controls keyboard-layout matching; it does not forward a key after an action cannot move a window. Official references: [default mapping](https://sw.kovidgoyal.net/kitty/conf/#shortcut-kitty.Move-window-forward), [action](https://sw.kovidgoyal.net/kitty/actions/#action-move_window_forward), and [layout matching](https://sw.kovidgoyal.net/kitty/mapping/#non-latin-keyboard-layout-support).

Current application owners are `TldwCli.BINDINGS`, `TldwCli.action_toggle_focus_mode`, `TldwCli._clear_focus_if_leaving_console`, and `ChatScreen._register_console_footer_shortcuts`. `Tests/UI/test_focus_mode.py` already mounts the application and dispatches `pilot.press("ctrl+shift+f")`; that regression arrived in commit `350c4162e2`. The remaining gap is a reachable terminal-delivered exit and truthful discoverability, not absent in-process dispatch coverage.

The original report also named Ctrl+Shift+H/P/Z collisions. Treat those as historical leads for AC #4, not a completed census of present bindings after Console extraction. TASK-1733's Ctrl+Y alias concerns a separate non-Kitty-protocol collapse; TASK-18935 concerns the terminal/protocol matrix. Neither closes this issue.

The original fixture failed before key dispatch with `RecoveryRequired: raw_source_selection_changed`. That is historical environment evidence, not a fresh current-dev baseline. Future tests must use the repository's private profile/config ownership and compare the actual current baseline.

## Existing workarounds

Leaving the Console clears focus mode. The command palette's Toggle Focus Mode action also exits it. A reporter can disable startup focus mode with `[general] focus_mode = false`.

For kitty, `map ctrl+shift+f no_op` removes kitty's interception of this chord; see the official [no_op action](https://sw.kovidgoyal.net/kitty/actions/#action-no_op). These workarounds do not complete the product acceptance criteria.

## ADR check

ADR required: no new ADR for this issue filing and documentation reconciliation.
ADR path: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`, `backlog/decisions/071-focus-mode-chrome-free-console.md`, and `backlog/decisions/210-console-region-ownership.md`.
Reason: this PR changes no runtime or UX behavior. The eventual exit design must follow these existing contracts and amend ADR-071 if its shortcut or affordance changes.

## Renumbering provenance

This is the unresolved focus issue originally filed as TASK-32927 in PR #2826 at source `a2d70ede9a8c4a62a164b54740da630a7591ce59`, with historical report date 2026-09-24. Its source ID collides with the existing shell-filename task. A fresh scan of registered worktrees and local/remote branch backlog trees on 2026-10-11 allocated this new ID; the frozen source branch preserves the original record.

The same source PR's three Console capture incident additions were omitted because their fixes and recovery evidence already live in TASK-33940.3, TASK-33940.4, and TASK-32951 on dev (including PR #2975's RAG repair). They do not establish additional open work or new Done claims here. The source reference to TASK-32926 was removed because that ID now identifies unrelated keyring work.
