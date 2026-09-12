# ADR-152: Roleplay mode keys move to single letters; screens never bind the shell nav layer

- Status: Accepted
- Date: 2026-09-12
- Amends: ADR-031 (TUI keybinding and footer-hint conventions), narrowing the
  task-32458 refinement's destination-local carve-out
- Task: task-32500

## Context

The task-32458 refinement to ADR-031 made the shell-destination hotkey layer a
left-to-right keyboard walk (`ctrl+1`…`ctrl+9`, `ctrl+0`, then `f2` `f3` `f4`
`f5` `f7`) and recorded one destination-local carve-out: the Roleplay mode chips
keep `ctrl+1`–`ctrl+4` on their own screen, judged "harmless since the user is
already there."

That judgment only covers `ctrl+4` (Roleplay's own slot). On the Roleplay
screen, `ctrl+1`, `ctrl+2`, and `ctrl+3` switch modes instead of navigating to
Home, Console, and Library — while the nav bar keeps painting `⌃1 Home`,
`⌃2 Console`, `⌃3 Library`. The labels lie on exactly the screen where a
newcomer is most likely to try them, and the user guide carries the cost: an
exception block in the index, caveat notes on every Roleplay sub-page, and a
published troubleshooting entry for "Ctrl+2 didn't take me to Console."

ADR-031 rule 1 already reserves global keys from screens; the carve-out made
the nav layer the one global layer a screen may still borrow.

## Decision

1. **The shell destination hotkey layer is fully reserved.** No `BaseAppScreen`
   or widget may bind any key in `SHELL_DESTINATION_SHORTCUTS`. The task-32458
   carve-out for destination-local shadowing is withdrawn. (Modal-scoped
   bindings keep their own task-32458 treatment — a modal owns its keyboard
   while open.)
2. **Roleplay mode switching moves to single letters** per ADR-031 rule 3
   (htop-style printable keys): `c` Characters, `p` Personas, `d` Dictionaries,
   `l` Lore, zipped positionally against `MODE_CHIP_ORDER` via a new
   `MODE_HOTKEYS` tuple so the two can never desync. `[` / `]` cycling stays.
   Printable keys are consumed first by focused text inputs, so the letters
   only act from list/button focus. The one known shadow: the persona-buddy
   overlay binds `c` (collapse) while the buddy itself holds DOM focus — a
   narrow, pre-existing exception documented at the binding site.
3. **Guard test.** A test introspects every `BaseAppScreen` subclass's BINDINGS
   and fails if any binds a shell destination hotkey, so the next shadowing
   regression fails CI instead of shipping another lying label.

## Alternatives considered

- Keep the carve-out (status quo): rejected — it answers "ctrl+4 is harmless"
  but says nothing for ctrl+1/2/3, which are the keys the nav bar advertises
  most heavily; documentation does not fix a label that lies on one screen out
  of fifteen.
- `alt+1..4` for modes: rejected — still a chord layer, and Console already
  owns `alt+1..9` for session tabs; two digit-chord meanings would remain.

## Consequences

- Nav labels are truthful on every screen; the user guide's exception block,
  per-page caveats, and the "Ctrl+2 didn't take me to Console" troubleshooting
  entry are removed.
- Roleplay users' `ctrl+1..4` muscle memory breaks once, replaced by
  ADR-031-conformant single letters shown in the footer hint and chip tooltips.
- The footer hint changes from `ctrl+1-4 mode` to `c/p/d/l mode`.
