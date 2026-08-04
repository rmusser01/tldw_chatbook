# ADR-031: TUI Keybinding and Footer-Hint Conventions

- Status: Accepted
- Date: 2026-08-03
- Context: UX critique of Lab/Schedules/Logs (`.impeccable/critique/2026-08-03T17-35-45Z__tldw-chatbook-ui-screens.md`, issues `output/ux-review/ux-issues-lab-schedules-logs.md`). The Schedules screen bound `ctrl+c` to "create reminder" (and taught it in first-run empty-state copy), shadowed the global `ctrl+p` command palette with a screen-level stub, and used `ctrl+s`/`ctrl+d`/`ctrl+r`, which collide with terminal flow-control (XOFF), EOF, and readline history-search reflexes. Two of five footer-advertised shortcuts were stubs that only toasted "Not yet available".
- Decision:
  1. **Reserved globals.** `ctrl+q` (quit) and `ctrl+p` (command palette) are app-global. Screens and widgets MUST NOT bind them. `f1` (help) and `f6` (next pane) are also app-global.
  2. **Never bind terminal-convention keys for app actions.** No screen/widget bindings for `ctrl+c`, `ctrl+v`, `ctrl+x`, `ctrl+s`, `ctrl+d`, `ctrl+z`, `ctrl+a`, `ctrl+r`, `ctrl+w`. These mean interrupt/paste/cut/XOFF-freeze/EOF/suspend/select-all/history-search/delete-word to users' muscle memory; overriding them trains dangerous habits and breaks terminal behavior.
  3. **Screen actions use single-letter, htop-style bindings** (e.g. `c` create, `d` delete, `s` sync). Printable keys are safely consumed first by focused text inputs, so single letters are safe outside inputs and fast everywhere else. Destructive single-letter actions MUST be guarded by a confirmation dialog (as delete already is).
  4. **Truthful footer hints.** The footer may only advertise bindings that are implemented and functional on the current screen. A binding whose action is a stub MUST be removed (binding + hint together) until the feature lands — no "Not yet available" toasts for advertised keys. Footer hints and `BINDINGS` must be kept 1:1 (enforce with a test).
  5. **Empty states and onboarding copy MUST NOT teach convention-breaking keys.** First-run copy references the same single-letter bindings shown in the footer.
- Alternatives considered:
  - Keep `ctrl+` chords but pick non-colliding ones (e.g. `ctrl+e`): rejected — chords are scarce, hard to discover, and still compete with readline; single letters are the established TUI pattern (htop, lazygit, ranger).
  - Implement the stub actions instead of removing them: rejected for now — the scheduler service has no run-now/pause-resume methods; when it does, bindings return per rule 4.
  - Leave footer stubs as "coming soon" affordances: rejected — advertised-but-dead keys burn trust in the hint system, which is the app's primary discoverability mechanism.
- Consequences: Schedules rebinds to `c`/`d`/`s`; the run-now and pause/resume stubs are removed until the scheduler service supports them; future screens follow rules 1–5; a test asserts footer-hint truthfulness. The intern-era bindings are retired.
- Links: critique snapshot above; issues UX-001–UX-004 in `output/ux-review/ux-issues-lab-schedules-logs.md`.
