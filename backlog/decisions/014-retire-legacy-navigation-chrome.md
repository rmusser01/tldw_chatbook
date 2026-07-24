# ADR-014: Retire the legacy navigation chrome and re-platform status writes

Status: Accepted (amended 2026-07-17 on rebase onto origin/dev)
Date: 2026-07-16
Related Task: N/A (UX remediation plan, step 1 — see Links)
Supersedes: N/A

## Decision

Commit exclusively to the destination-based master-shell chrome (`MainNavigationBar` + Textual `Footer`, mounted per screen by `BaseAppScreen`) and delete the legacy parallel chrome: the `TitleBar` widget, the three legacy 18-tab navigation widgets (`TabBar` / `TabLinks` / `TabDropdown`), and the dead `general.use_dropdown_navigation` / `general.use_link_navigation` config switches.

Status readouts the user relies on render on visible surfaces: chat token/cost usage and database sizes render in the per-screen `AppFooterStatus`, the active conversation title renders in the Console transcript header (`ConsoleSessionSurface.set_session_title`), and Console F1/F6 hints render through the visible footer.

## Amendment (2026-07-17, rebase onto origin/dev, PR #685)

Upstream `origin/dev` solved the same occluded-footer problem in parallel (task-264): `BaseAppScreen` now mounts a per-screen `AppFooterStatus`, and a `ShortcutContext` channel (`shortcut_context.py`, plus `CONSOLE_WORKBENCH_SHORTCUTS` / `LIBRARY_SHORTCUTS` style registrations) feeds it per-screen hint contexts. That upstream system is newer and integrated across Console, Library, MCP, and Personas, so this ADR is amended on rebase:

- **`AppFooterStatus` and `shortcut_context.py` are NOT deleted.** The per-screen mounting fixes the same occlusion this ADR targeted; our replacement widget (`AppStatusLine`) was dropped in favor of it.
- **`tab_events.py` is NOT deleted**: upstream `app.py` still routes legacy window buttons through `tab_events.handle_tab_button_pressed`.
- `event_dispatcher.py` stays deleted (zero consumers on both sides).
- Everything else in this ADR stands: the three legacy nav widgets, `TitleBar`, and the dead config switches are gone; token/DB writes (`chat_token_events.py`, `db_status_manager.py`) keep targeting `AppFooterStatus`, which is now visible on every screen.

## Context

A design critique (see Links) found the app composes two parallel chrome systems. `_create_main_ui_widgets` mounted `TitleBar`, one of three legacy 18-tab navs (selected by the `use_dropdown_navigation` / `use_link_navigation` config switches), and `AppFooterStatus` onto Textual's default screen — but startup immediately pushes a `BaseAppScreen` screen over the default screen and nothing ever pops back, so all of that chrome is permanently invisible (verified at runtime via headless Textual Pilot). Meanwhile live status was still written to the occluded widgets: token/cost and DB-size updates went to `AppFooterStatus`, conversation titles went to `TitleBar`, and Console F1/F6 hints were registered on the invisible footer. Users therefore saw no token/cost, persistence, or conversation-title feedback anywhere, and the two config switches switched between three kinds of nothing.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Re-wire the legacy navs as a user-selectable option | Keeps two chrome systems alive for zero visible benefit; the legacy navs duplicate the shell's `MainNavigationBar` with an 18-destination IA the product has already moved away from, and every navigation improvement would have to be implemented twice. |
| Keep both chrome systems mounted | The legacy chrome is permanently occluded by the pushed screen; keeping it only preserves dead code, dead config switches, and misleading write targets. |
| Put token/DB status in `MainNavigationBar` | Viable, but the nav bar's destination layout is owned by later IA work; a one-line status bar owned by `BaseAppScreen` is independent of that and keeps the update cadence trivially cheap (two `Static` text refreshes). |
| (Amendment) Keep our `AppStatusLine` replacement and delete `AppFooterStatus` | Would have required re-pointing upstream's newer per-screen shortcut-context consumers (Console, Library, MCP, Personas) against the current of task-264's integration, for no user-visible gain. |

## Consequences

- The `general.use_dropdown_navigation` and `general.use_link_navigation` config options are removed (including the Settings-window switch). Users who set them lose nothing visible: the options only selected which of three occluded nav widgets was mounted.
- Token/cost, DB-size, and conversation-title readouts are visible on every screen (per-screen `AppFooterStatus`, Console transcript header).
- Dead CSS for `#app-titlebar` is removed from the live stylesheets (`css/components/_widgets.tcss`, rebuilt `tldw_cli_modular.tcss`); the `AppFooterStatus` rules stay (the widget survives, per the amendment); the unused `Constants.css_content` legacy string is left untouched.
- (Amendment) The Console's rail Model readouts, transcript copy blocks, and footer-shortcut registration machinery from upstream are kept; the PR's Console dedup experiment was superseded by upstream's expanded Console internals.

## Links

- [UI critique motivating this change](../../.impeccable/critique/2026-07-17T01-08-49Z__k-ui-tab-bar-screen-headers-app-wide-console-first.md)
