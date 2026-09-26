---
id: TASK-32927
title: >-
  Focus mode's advertised exit chord (Ctrl+Shift+F) never reaches the app under
  kitty
status: To Do
assignee: []
created_date: '2026-09-24 23:14'
labels:
  - console
  - ux
  - terminal-compat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reported 2026-09-24 by the repo owner: the Console came up chrome-free (`[general] focus_mode = true`) and Ctrl+Shift+F -- the only exit affordance the focus-mode footer shows -- did nothing, so there was no discoverable way back to normal chrome. The fault is not chatbook's key dispatch: in kitty, ctrl+shift+f is a stock terminal shortcut, so the key is consumed by the terminal emulator before Textual or the app ever sees it. The footer therefore advertises a key the terminal never forwards as the single visible escape. ADR-071 assessed this chord as conflict-free against chatbook's own bindings only, never against terminal emulators, so the same collision silently affects the Console's other ctrl+shift+<letter> bindings too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On kitty -- or any terminal that consumes ctrl+shift+<letter> -- at least one advertised, reachable affordance exits focus mode; the user is never left chrome-free with only a terminal-intercepted key.
- [ ] #2 The focus-mode footer hint names a key that is actually delivered in the terminal being run, or stops presenting the intercepted key as the exit route (ADR-031 truthfulness rule for advertised hints).
- [ ] #3 A regression test proves the toggle fires from a real key event on a mounted Console (a Pilot.press dispatch), not merely that the binding is present in TldwCli.BINDINGS; the existing test_ctrl_shift_f_binding_registered assertion is insufficient evidence.
- [ ] #4 Every chatbook ctrl+shift+<letter> binding is checked against the reported terminal's stock keymap, and the surviving/colliding set is recorded where the next contributor will look: a `backlog/docs/lessons-*.md` entry, plus the ADR-071 amendment or `Docs/User_Guide` console quirks entry for ctrl+shift+f, ctrl+shift+h and ctrl+shift+p.
- [ ] #5 Focus mode semantics are unchanged: chrome suppression, the single exit rule on non-Console navigation, and the palette QuickAction all keep their current behaviour.
<!-- AC:END -->

## Root Cause (evidence gathered 2026-09-24)

Reported environment: `tldw-cli` running under **kitty 0.47.1**, with no `~/.config/kitty/kitty.conf` overrides, so kitty's stock keymap applies.

- **kitty owns the chord.** `opt('kitty_mod', 'ctrl+shift', ...)` (`/usr/lib64/kitty/kitty/options/definition.py:2826`) and `map('Move window forward', 'move_window_forward --allow-fallback=shifted,ascii kitty_mod+f ...')` (`definition.py:3144`). `Ctrl+Shift+F` moves the OS window forward instead of reaching the program; when the window is already frontmost the action is invisible, which is exactly what the reporter saw. `--allow-fallback` only forwards the key when the window cannot move, so delivery is at best inconsistent.
- **chatbook's side is intact.** The binding is `Binding("ctrl+shift+f", "toggle_focus_mode", ...)` (`tldw_chatbook/app.py:7759`), the action is `TldwCli.action_toggle_focus_mode` (`app.py:13613`), and the Console footer advertises it as the only exit affordance (`tldw_chatbook/UI/Screens/chat_screen.py:5189`). No other binding claims the chord (a repo-wide `ctrl+shift+f` search hits `app.py` only), and the Console composer/screen key handlers fall through for it, so nothing in-process swallows it.
- **Kitty keyboard-protocol negotiation does not save it.** Textual's Linux driver does request `DISAMBIGUATE_ESCAPE_CODES | REPORT_ALL_KEYS | REPORT_ASSOCIATED_TEXT` (`textual/drivers/linux_driver.py`, written as `\x1b[>25u`), but a terminal's own keymap is resolved before that protocol decides what to forward. This is a different failure mode from TASK-1733's non-Kitty-protocol collapse, not a repeat of it.
- **Same collision, same terminal, other Console chords:** `ctrl+shift+h` (hands-free) = kitty `show_scrollback` (`definition.py:3019`); `ctrl+shift+p` (view context) is the prefix of kitty's `kitty_mod+p>…` sequences (`definition.py:3440-3500`), so kitty eats the chord and waits for a second key; `ctrl+shift+z` (redo) = kitty `scroll_to_previous_prompt` (`definition.py:3008`), already worked around by TASK-1733's portable `ctrl+y` alias.

## Resolution applied for the reporter (2026-09-24)

1. **Immediate escape, no chord needed** — any navigation to a non-Console destination clears focus mode (`app.py:14553` → `_clear_focus_if_leaving_console`, ADR-071's single exit rule): `F4`/`F2`/`F3`/`F5`/`F7`. In place: `ctrl+p` → "Quick Actions: Toggle Focus Mode" (`FOCUS_TOGGLE_PALETTE_ENTRY`, `app.py:1609`).
2. **Terminal-side fix for the advertised chord** — add to `~/.config/kitty/kitty.conf`, then reload with `ctrl+shift+f5` (kitty's `reload_config_file`):

   ```conf
   map ctrl+shift+f no_op
   ```

   kitty's own action documentation: "Mapping a shortcut to no_op causes kitty to not intercept the key stroke anymore, instead passing it to the program running inside it" (`/usr/lib64/kitty/kitty/actions.py:62`).
3. **Config-side fix** — `[general] focus_mode = false` in `~/.config/tldw_cli/config.toml`, so launches are no longer chrome-free (the runtime toggle is deliberately non-persistent). Safe to set while the app is running: config mutations apply exact keys against the locked on-disk snapshot (`tldw_chatbook/config.py:7926`), so a save from the live instance cannot revert it.

None of this fixes the product. The advertised exit chord stays dead for every kitty user until AC #1-#4 land.

## ADR check

ADR required: yes — amend `backlog/decisions/071-focus-mode-chrome-free-console.md`, whose "ctrl+shift+f is verified conflict-free" claim was checked against chatbook's own bindings only. Any replacement or alias chord must also satisfy ADR-031 (no terminal-convention keys; htop-style single letters for screen actions).

## Related

- TASK-18935 — terminal-matrix verification of the kitty keyboard protocol and Alt chords (same evidence-gathering angle, different failure mode).
- TASK-1733 — `ctrl+y` alias for `ctrl+shift+z` on non-Kitty-protocol terminals: the precedent for a portable alias, and for keeping the original chord where it is delivered.
- ADR-071 (focus mode), ADR-031 (keybinding and footer-hint conventions).

## Notes for whoever picks this up

The pilot harness used by `Tests/UI/test_focus_mode.py` (`_make_app_instance` → `Tests/UI/app_factory.py:112` → `load_settings()`) failed in the reporter's environment with `RecoveryRequired: raw_source_selection_changed` **before any key was pressed** — the same environmental fixture failure recorded in TASK-32926's notes, and reproducible at baseline (the existing `TestFocusChromeSuppression::test_focus_mode_hides_chrome_keeps_status_line` fails the same way). Establish baseline before attributing an AC #3 regression-test failure to a code change.
