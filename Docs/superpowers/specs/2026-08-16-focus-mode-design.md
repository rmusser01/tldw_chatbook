# Focus Mode — Design (chrome-free Console presentation)

Date: 2026-08-16
Status: Proposed (brainstorm-approved direction; revised after owner + self review)
Related Task: [TASK-16320](../../../backlog/tasks/task-16320%20-%20Focus-mode-chrome-free-Console-presentation.md)
Related ADR: [ADR-067](../../../backlog/decisions/067-focus-mode-chrome-free-console.md)

## Problem

The Console lives inside the full app shell: `MainNavigationBar` (3 rows,
top), the Console's own `DestinationHeader`, and `AppFooterStatus`
(bottom). For zen coding — and for using Chatbook from a phone, where
there is no fine pointer and every row is expensive — users want a
claude-code/codex-style surface: just the conversation and the composer,
plus a minimal status line.

## Goals

- Present the Console with the nav bar and workbench header hidden —
  message stream, composer, panes, and a **one-line status bar** only.
- Enter focus mode three ways: `--focus` CLI flag, `[general] focus_mode`
  config default, and an instant runtime toggle (`ctrl+shift+f`,
  app-level, + a command-palette QuickAction).
- Zen, not kiosk: ctrl+p palette and f1 help stay available; navigating
  to any non-chat route exits focus mode and restores chrome.
- Default presentation completely unchanged when focus mode is off.

## Non-goals

- No hard-lock/kiosk variant (explicitly not wanted).
- No new mobile subsystem: no touch theme, no on-screen keyboard helper
  buttons, no Termux-specific UI. The phone story is `--serve` browser
  mode plus a separate tappability audit follow-up.
- No auto-entering focus mode based on terminal size — explicit toggle
  only (mirrors ADR-043's explicit-toggle-over-implicit-sizing stance).
- No inline token/context meter in the composer — the retained status
  line already carries the token count (hides below 110 cols by existing
  width-tier rules).
- No write-through persistence of the runtime toggle — config is a
  default, not a lock; the session's last toggle state dies with it.

## Confirmed decisions

From the brainstorm session and review with the repo owner:

1. **Chrome-free except one status line** — `MainNavigationBar` and
   `DestinationHeader` are hidden; `AppFooterStatus` **stays**. It is
   already a single-row status bar (`dock: bottom; height: 1`,
   `AppFooterStatus.py:117-125`) showing the Console token count, the
   app-global key hints, and width-progressive right-cluster hiding —
   no reduced variant needed.
2. **`ctrl+shift+f` as the toggle key, app-level** — verified
   conflict-free: zero occurrences in app code or tests; existing
   ctrl+shift+ bindings are z/p/h/c/a only; not in ADR-031's forbidden
   (plain ctrl+c/v/x/s/d/z/a/r/w) or reserved (ctrl+p, ctrl+q, f1, f6)
   sets; follows the Console's existing ctrl+shift+h / ctrl+shift+p
   pattern. App-level (not console-scoped) so recovery from an exited
   focus state is one keypress from anywhere.
3. **Zen, not kiosk, with one navigation rule** — there is no hotkey
   gating. ALL navigation to a non-chat route exits focus mode at a
   single choke point. Re-entry is one keypress (decision 2). No
   shared-setup locking.
4. **Phone delivery = `--serve` browser mode primarily, Termux tolerated
   but not designed for.**

## Semantics

- `focus_mode` is **app-level state whose visible effect is scoped to
  the Console**. Other screens always render normal chrome.
- **Single exit rule:** any navigation to a non-chat route — via
  destination hotkeys (ctrl+1..0, F7–F9), the palette's tab commands, or
  any other `NavigateToScreen` path — clears `focus_mode` before the
  switch, so the destination arrives with normal chrome. There is no
  gating layer and no "invisible chrome with live hotkeys" state to
  keep in sync.
- **Single re-entry rule:** `ctrl+shift+f` (or the QuickAction) anywhere
  in the app — if not on the Console, navigate to the Console first,
  then apply focus; if on the Console, toggle chrome in place.
- ctrl+p palette, f1 help, and all Console-internal keys (console tabs,
  panes, popovers) are unaffected by focus mode.

## Visual scope

| Element | Source | In focus mode |
|---|---|---|
| `MainNavigationBar` | `BaseAppScreen.compose()` (`base_app_screen.py:216`) | hidden |
| `#console-workbench-header` (`DestinationHeader`) | `ChatScreen.compose_content` (`chat_screen.py:14474`) | hidden |
| `#screen-footer-status` (`AppFooterStatus`) | `BaseAppScreen.compose()` (`base_app_screen.py:232`) | **kept** (1 row: token count + key hints) |
| Message stream, composer, panes, popover overlays | Console content | unchanged |

- Suppression composes with existing responsive rules: `-console-compact`
  (< 35 rows, already hides the workbench header), single-pane (< 84
  cols), narrow prompt queue (< 92 cols) are orthogonal and stack.
- The footer's built-in width tiers (token count hides < 110 cols, word
  count < 100, DB size < 80) already adapt the status line to phone
  widths with no new work.
- **Exit-hint discoverability:** the Console's registered shortcut
  context (the `CONSOLE_WORKBENCH_SHORTCUTS` /
  `register_footer_shortcuts` mechanism, `chat_screen.py:1171`,
  `chat_screen.py:3023`) gains a `ctrl+shift+f` entry whose label
  reflects the target state ("focus" when off, "exit focus" when on).
  This rides the existing ADR-031-governed footer pipeline — no new hint
  surface, and the advertised key is a real binding.

## Mechanism

1. **State:** `TldwCli.focus_mode: bool` (default `False`), plus one
   helper `_set_focus_mode(enabled)` that stores the flag and applies
   the `-focus` CSS class to the Console screen when it is active.
2. **CSS:** two rules in `css/components/_agentic_terminal.tcss`
   (next to the `-console-compact` precedent at `:4895`; the app bundle
   `tldw_cli_modular.tcss` is generated from these sources by
   `css/build_css.py` and must never be hand-edited). The class must root
   at the screen, not `#console-shell` — `MainNavigationBar` is composed
   by `BaseAppScreen` as a sibling of `#console-shell`:
   ```css
   ChatScreen.-focus MainNavigationBar { display: none; }
   ChatScreen.-focus #console-workbench-header { display: none; }
   ```
   `display: none` only — no new raw visual values, so ADR-042 token
   governance is unaffected. `BaseAppScreen.compose()` is untouched
   (ADR-014's per-screen chrome model preserved).
3. **Apply points:** (a) `ChatScreen.on_mount` reads
   `self.app.focus_mode` (covers startup, palette navigation back to the
   Console, and the toggle-from-elsewhere path); (b) the app-level
   toggle action applies/removes the class in place when the Console is
   active; (c) `_handle_screen_navigation_locked` (`app.py:8478`)
   clears the flag before `switch_screen` when the target route is not
   `chat` — the single exit choke point.
4. **Toggle:** app-level `Binding("ctrl+shift+f", "toggle_focus_mode")`
   in `TldwCli.BINDINGS` (`app.py:5246`). Enabling while elsewhere posts
   `NavigateToScreen("chat")` first (the mount-time read applies the
   class); disabling clears the flag and removes the class if the
   Console is active. A `QuickActionsProvider` entry ("Toggle focus
   mode", `app.py:868` area) invokes the same action for pointer/touch
   users. The Console's footer shortcut-context label flips with state
   (see Visual scope).

## Config & CLI

- **Config:** `focus_mode = false` added to the `[general]` template
  (`config.py:2679` area, next to `default_tab`); read via
  `get_cli_setting("general", "focus_mode", False)` alongside the
  `default_tab` read (`app.py:5597`).
- **CLI:** `--focus` added to the existing argparse block
  (`app.py:12689`); passed into the app before `run_async`. Combines
  naturally with `--serve` as the phone launcher.
- **Precedence:** CLI flag > config > default `False`.
- **Route resolution:** when focus is requested at startup, the initial
  route is forced to `chat`, overriding `[general] default_tab` — except
  **first-run onboarding wins** (`_resolve_initial_shell_route`,
  `app.py:8206` keeps its first-run → `TAB_HOME` branch ahead of the
  focus override; focus applies when the Console is subsequently
  mounted, via the same mount-time read).

## Edge cases

- **Toggle during active streaming/worker:** no screen rebuild occurs
  (CSS class flip only), so sessions, workers, and streaming state are
  untouched.
- **Screen recompose:** the `-focus` class lives on the screen itself
  and survives widget-level recompose of children; `BaseAppScreen`
  re-seeds footer state independently (existing behavior).
- **Modals/popovers:** focus mode does not suppress overlays (model
  popover, approvals, help). They are content, not chrome. The
  app-level toggle binding is not shadowed by any modal (verified: no
  screen or widget binds ctrl+shift+f).
- **Splash screen:** `--focus` composes with the existing splash flow
  (`_run_no_splash_post_mount_setup` path included); focus applies when
  the Console is pushed.
- **`focus_mode = true` config + in-app toggle off:** the runtime toggle
  wins for the rest of the session (config is a default, not a lock).

## Testing

New `Tests/UI/test_focus_mode.py`, following the
`test_console_workbench_contract.py` harness patterns, plus config/CLI
unit tests:

- Mount the Console with `focus_mode` on → `MainNavigationBar` and
  `#console-workbench-header` are `display: none` / not visible;
  `#screen-footer-status` **remains visible**; content pane present.
- Default mount (flag off) → all chrome visible (guards the "unchanged
  by default" goal).
- App-level `ctrl+shift+f` from the Console → flag flips, class
  applied/removed, footer label flips focus/exit-focus.
- App-level `ctrl+shift+f` from a non-chat screen → navigates to the
  Console which mounts chrome-free.
- Simulated `NavigateToScreen(TAB_SETTINGS)` while focus is on — via
  the destination-hotkey path and the palette path — → flag cleared,
  target screen mounts with chrome.
- Footer shortcut context includes the `ctrl+shift+f` entry and only
  advertises real bindings (ADR-031 truthfulness).
- Startup: `--focus` resolves initial route to `chat`; first-run still
  resolves to `TAB_HOME`; `[general] focus_mode = true` config behaves
  like the flag; CLI flag overrides a `false` config.
- Existing console contract and keybinding-governance tests keep passing
  untouched.

## ADR check

ADR required: yes — long-lived UX/application structure decision.
ADR path: `backlog/decisions/067-focus-mode-chrome-free-console.md`
Reason: introduces a new presentation mode with policy (suppression vs
non-composition, single-exit-rule semantics, retained status line) that
amends how ADR-014's per-screen chrome is presented, and records why
the Textual `Mode` system and hotkey gating were rejected.

## Follow-ups (out of scope, separate tasks)

- Mobile tappability audit over `--serve`: send/approvals as buttons, no
  hover-only information, soft-keyboard escape flows for modal-Escape
  patterns.
- `Docs/User_Guide/console.md` update ships **with** the implementation
  task, not as a follow-up.
