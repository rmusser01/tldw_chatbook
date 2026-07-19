# ADR-016: Palette liveness, the Ctrl+N destination hotkey layer, and BINDINGS-driven help

Status: Accepted
Date: 2026-07-17
Related Task: N/A (UX efficiency review, cycle 2)
Supersedes: N/A

## Decision

Three truthfulness and speed fixes land together, all anchored in `app.py`:

**Palette commands must be live.** Every command-palette entry either performs its stated action or is deleted. Notify-only ("...requested" / "...initiated") placebo commands are removed rather than half-wired. A command is only kept wired when the real action is a short call into existing machinery:

- Wired: LLM "Switch to X" now sets the same `chat_api_provider_value` reactive the Settings screen drives; "Show Current Provider" reads that reactive instead of a nonexistent `current_provider` attribute; "Show Database Stats" navigates to the Statistics screen; "Show Keybindings" opens a `WorkbenchHelpPanel` generated from the app's real `BINDINGS`.
- Deleted: "Test API Connection", the three temperature presets, "Toggle Streaming Mode", "Reload Configuration", "Reset to Default Settings", Quick Actions "Clear Current Chat" / "Export Chat as Markdown" / "Refresh Database", Media "Recent Media Files" / "Show Ingested Content" / "Open Media Database" / "Refresh Media Library" / "Export Media List", Character "Switch/Edit/Delete/Import/Export Character", and Developer "Clear Cache" / "Debug Mode Toggle" / "Memory Usage" / "Database Integrity Check" / "Export Debug Info". These need in-screen state or per-provider machinery that a palette hit cannot truthfully complete.

**Ctrl+N destination hotkey layer.** `TldwCli.BINDINGS` gains `ctrl+1`..`ctrl+9` then `ctrl+0`, built by one loop that zips the key list against `SHELL_DESTINATION_ORDER`. Every binding maps to a single parameterized action, `shell_destination(index)`, which posts `NavigateToScreen(destination.primary_route)`. The layer binds the first ten destinations (Home through ACP); Lab and Settings have no hotkey because the approved key set stops at `ctrl+0`. Hotkeys are `show=False` (no footer noise) and no index numbers appear in nav labels.

**Generic contextual help.** When the active screen has no custom `action_show_workbench_help`, F1 now opens a `WorkbenchHelpPanel` listing that screen's own `BINDINGS` as key/description pairs (falling back to the app-level bindings when the screen declares none) instead of toasting "No contextual help is available for this screen." Every screen gets truthful help from data that already exists. `WorkbenchHelpPanel` also gains an Escape binding so it can be dismissed from the keyboard.

## Context

A UX efficiency review (P1 trust, P1 speed, P2 help parity) found palette commands that reported success without acting, no keyboard path between shell destinations, and an F1 handler that produced a dead toast on 10 of 12 destinations.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep placebo commands for discoverability | A command that claims to act but only notifies is a lie that costs trust every time it is discovered; the palette alias model (ADR-015) already covers discovery through navigation commands. |
| Half-wire commands to nearest-related machinery | A temperature preset that does not reach the active session's settings is still a placebo; deletion is preferred over wiring that only works in some contexts. |
| Per-screen custom help everywhere | Twelve hand-maintained help texts drift out of date; screen `BINDINGS` are the source of truth the keys already obey, so generated help cannot go stale. |
| Hardcoded per-key handlers for the hotkey layer | Twelve near-identical actions duplicate what one indexed action and a zipped binding list express directly; the loop also keeps the key map mechanically tied to `SHELL_DESTINATION_ORDER`. |

## Consequences

- Palette search/discover lists shrink to commands that do something; tests assert the deleted entries stay absent.
- `SHELL_DESTINATION_ORDER` is the single ordering source for both the nav rail and the hotkey layer.
- `WorkbenchHelpPanel` is now also the keybindings viewer for the Developer palette command and the generic F1 fallback; it gains `escape -> dismiss`.
- F6 is unchanged: screens without pane support keep their truthful "no pane focus target" toast.
- Copy note: no em dashes are used in any user-facing text added here (notifications, help titles, binding descriptions).

## Links

- [ADR-015: Complete the shell destination IA](015-shell-destination-ia.md)
- [ADR-014: Retire the legacy navigation chrome](014-retire-legacy-navigation-chrome.md)
