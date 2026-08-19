# ADR-071: Focus mode — chrome-free Console presentation

Status: Proposed
Date: 2026-08-16
Related Task: [TASK-18812](../tasks/task-18812%20-%20Focus-mode-chrome-free-Console-presentation.md)
Related Spec: [Focus Mode — Design](../../Docs/superpowers/specs/2026-08-16-focus-mode-design.md)
Supersedes: N/A (complements [ADR-014](014-retire-legacy-navigation-chrome.md))

## Decision

Chatbook adds **focus mode**: an app-level boolean whose visible effect is
scoped to the Console (route `chat`). While active, the Console hides
`MainNavigationBar` and its `DestinationHeader` via CSS (`display: none`)
rather than by not composing them, and **keeps the one-line
`AppFooterStatus`** — the message stream, composer, panes, and a single
status row (token count + governed key hints) are the entire surface.

Mechanism:

- `TldwCli` holds the `focus_mode` flag. The Console screen carries a
  `-focus` CSS class; the two suppression rules live in
  `ChatScreen.BUNDLED_CSS` (the `-console-compact` precedent). They use
  `display: none` only, so they introduce no raw visual values (ADR-042
  token governance holds).
- A single helper sets the flag and applies/removes the class;
  `ChatScreen.on_mount` reads the flag so any Console mounted while focus
  is on starts chrome-free. `BaseAppScreen.compose()` itself is untouched
  — ADR-014's per-screen chrome model is preserved, only its presentation
  is suppressed.

Semantics are **zen, not kiosk, with one navigation rule**:

- Any navigation to a non-chat route — destination hotkeys, palette tab
  commands, any `NavigateToScreen` — clears focus mode at one choke
  point in the navigation handler, so the destination arrives with
  normal chrome. There is **no hotkey gating layer**.
- Re-entry is a single app-level keypress: `ctrl+shift+f` toggles focus
  from anywhere (enabling while elsewhere navigates to the Console
  first). A palette QuickAction invokes the same action for pointer/
  touch users.
- ctrl+p palette and f1 help always remain available; no hard-lock
  variant exists.

Entry points: a `--focus` CLI flag (combines with `--serve` as the phone
launcher), a `[general] focus_mode` config default, and the runtime
toggle. `ctrl+shift+f` is verified conflict-free (existing ctrl+shift+
bindings are z/p/h/c/a only) and is not in ADR-031's forbidden or
reserved sets; the Console advertises it through the existing
footer shortcut-context pipeline so the advertised key is a real
binding.

CSS suppression is chosen over structural alternatives because
`ChatScreen` carries heavy live state (sessions, workers, streaming); the
toggle must be instant, reversible, and single-code-path. The cost of
mounted-but-hidden widgets is negligible.

## Context

Users want a claude-code/codex-style "just the conversation" surface for
zen coding, and a Console usable from a phone (via the existing
`--serve` browser mode) where there is no fine pointer and every screen
row is expensive. The Console is already the default destination
(`[general] default_tab`), and it already hides its workbench header
below 35 rows — focus mode is the explicit, user-controlled generalization
of that idea (explicit toggle over implicit sizing, mirroring ADR-043).
The owner review (2026-08-16) confirmed the retained one-line status
bar: `AppFooterStatus` is already `height: 1` with width-progressive
hiding, so keeping it costs nothing and preserves the visible token
count on desktop widths.

## Alternatives Considered

- **Chrome-less screen variant** (skip chrome inside
  `BaseAppScreen.compose`): two compose paths in the base class every
  screen inherits, and a runtime toggle forces a recompose or
  `switch_screen` of the most stateful screen in the app — risk of losing
  console state for no user-visible gain over suppression.
- **Textual `Mode` system** (`SCREEN_MODES`/`switch_mode`): the
  first-class mechanism, but the repo has zero Mode usage, and it implies
  a second `ChatScreen` instance (state disaster) or delicate
  screen-stack sharing. Highest risk, same user-visible result.
- **Hide the footer too (fully chrome-free) + composer-placeholder
  hints:** rejected at owner review — the one-line status bar is kept,
  which also removes the need for a placeholder-hint mechanism (the
  footer's governed shortcut-context pipeline already advertises keys).
- **Gate destination hotkeys while focused (muscle-memory nav inert,
  palette as the only deliberate escape):** rejected in self review — it
  requires `check_action` wiring across thirteen app-level bindings and a
  permanent "hidden chrome vs live hotkeys" invariant to keep in sync,
  to protect against an accident (ctrl+digit mid-zen) whose recovery is
  already one keypress once the toggle is app-level. One navigation rule
  beats two escape semantics.
- **Config-only entry (no runtime toggle)**: rejected — users move
  between zen and the full shell within a session; forcing a restart
  would make the mode a curiosity instead of a habit.

## Consequences

- **Positive:** instant toggle with zero screen rebuilds; one compose
  path; one exit rule and one re-entry rule (no gating state to drift);
  composes cleanly with the existing small-terminal CSS
  (`-console-compact`, single-pane, narrow queue) and the footer's own
  width tiers; token count and key hints survive on desktop widths;
  foundation for phone use over `--serve`.
- **Trade-offs:** hidden widgets stay mounted (negligible); an accidental
  ctrl+digit mid-zen exits focus — accepted in exchange for the simpler
  single-rule semantics, with one-keypress re-entry; the status line
  costs one row versus fully chrome-free (owner-accepted); runtime
  toggles do not persist across sessions (config is a default, not a
  lock).
- **Mobile:** focus mode is necessary but not sufficient for phone use; a
  follow-up audit (buttons for send/approvals, no hover-only information,
  soft-keyboard escape flows) is tracked separately.

## Links

- [ADR-014: Retire legacy navigation chrome](014-retire-legacy-navigation-chrome.md)
- [ADR-031: TUI keybinding and footer hint conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-042: Design token system](042-design-token-system-and-design-language.md)
- [ADR-043: Rail collapse yields to explicit toggle](043-console-rail-compact-collapse-yields-to-explicit-toggle.md)
- [Spec: Focus Mode — Design](../../Docs/superpowers/specs/2026-08-16-focus-mode-design.md)
