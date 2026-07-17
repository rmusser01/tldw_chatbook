# ADR-015: Complete the shell destination IA (Lab, folds, palette aliases, identity headers)

Status: Accepted (amended 2026-07-17 on rebase onto origin/dev)
Date: 2026-07-17
Related Task: N/A (UX remediation plan, step 2 — see Links)
Supersedes: N/A

## Decision

Finish the information architecture started with the master shell so that every registered route has a home and every screen says what it is.

**Destination taxonomy (12, as amended).** `SHELL_DESTINATION_ORDER` gains one destination, **Lab**, seated between ACP and Settings (`navigation_priority=45`). Lab's primary route is `llm` (the Models screen) and it folds `llm_management`, `stts`, and `evals` as legacy routes. The full order is: Home, Console, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Lab, Settings. There is no Skills destination: upstream retired it into Library (Skills sub-project tasks 1-5, route alias `skills` -> `library`), and this ADR adopts that retirement on rebase.

**Fold map.** Every formerly orphaned route resolves to an owning destination while keeping its own screen as the canonical route:

| Route | Destination | Canonical route |
| --- | --- | --- |
| `writing`, `research` | Library | unchanged |
| `logs`, `stats` | Settings | unchanged |
| `llm`, `stts`, `evals` | Lab | unchanged |
| `llm_management` | Lab | `llm` |
| `coding` | Console | `chat` |
| `prompts`, `skills` | Library | unchanged (upstream retirement) |

**Coding is merged into Console, not kept.** The standalone Coding screen is deleted: `CodingScreen`, its `coding` route in `screen_registry.py`, and `UI/Coding_Window.py` (whose only live consumers were the screen and one integration test). Legacy `coding` links land on Console via two seams: the shell destination model (`coding` in Console's `legacy_routes`, canonical `chat`) and a screen-registry alias (`"coding": "chat"`), mirroring how the retired Notes route folds into Library. `CodeRepoCopyPasteWindow` survives untouched — it is a standalone modal with its own tests and never depended on `CodingWindow`.

**Palette alias model.** The command palette shows exactly one navigation command per destination (12), driven by `SHELL_DESTINATION_ORDER` instead of `NAVIGATION_TABS + ALL_TABS`. Legacy route names (`ccp`, `tools_settings`, `media`, `search`, `study`, `writing`, `research`, `logs`, `stats`, `llm`, `llm_management`, `stts`, `evals`, `coding`, `ingest`, `notes`, `chatbooks`, `subscriptions`, `customize`, `characters`, `prompts`, `conversation`, `subscription`, `conversations_characters_prompts`) are alias terms on their owning destination's single command: searchable, and activating the hit lands on the destination's primary route. This eliminates the duplicate "Personas" (`TAB_CCP` vs `TAB_PERSONAS`) and "MCP" (`TAB_TOOLS_SETTINGS` vs `TAB_MCP`) entries. `switch_tab()`/`route_for_tab()` keep legacy tab ids navigable for non-palette callers.

**DestinationHeader is the standard identity header.** `DestinationHeader` (title + subtitle + text-labeled status badge, driven by `WorkbenchHeaderState`, live via `sync_state()`) is the one component screens use to identify themselves:

- Console: the previously hidden `DestinationHeader` becomes visible. The legacy `#console-title` / `#console-purpose` / `#console-status-row` / `#console-mode-bar` statics stay mounted but hidden: they are a tested contract seam (mode-bar freshness, parity geometry, gate1 region list) and remain copy-consistent with the visible header.
- The folded/orphan screens mount it with per-screen copy: Search, Media, Study, Writing, Research, Models (llm), Speech (stts), Logs, Stats, and Evals (seat only; its internal push flow is owned by later work).
- Personas' hand-rolled title line (`Roleplay | Author the pieces that shape a chat | Ready`) migrates to the component; the live editing state moves into the subtitle and the mode-descriptor/counts statics stay.
- `StatsScreen` moves onto `BaseAppScreen`, gaining nav/footer/status chrome like every other screen, and its header status badge goes live (`loading` / `error` / `ready` / `empty`) off the existing statistics reactives.
- Home keeps its dashboard header; Console's control bar/chips are untouched (owned by later steps).

## Context

A design critique (see Links, P1) found that 8 screens (evals, llm, stts, writing, research, logs, stats, coding) routed but mapped to no shell destination — the nav highlighted nothing there and offered no button to reach them — while screens folded into Library (media, search, study) showed Library's active state with no screen-level identity. The command palette listed destinations and legacy tabs side by side, producing visibly duplicated "Personas" and "MCP" commands.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Grouped nav of ~20 destinations | A rail of ~20 tabs is unscannable and recreates the retired 18-tab chrome (ADR-014). Folding related screens under 12 destinations keeps the rail readable and gives each fold a clear owner. |
| Promote Coding to a destination | The Coding screen is a sidebar shell around one working tool (`CodeRepoCopyPasteWindow`) plus placeholder panes; it does not carry a top-level destination. Console is where code work happens (agent chat with tools), so Coding merges there and its route folds into Console. |
| Keep the Console `DestinationHeader` hidden and show the legacy statics instead | The legacy statics are one-off markup; the whole point of the step is one consistent identity component. The statics stay mounted (hidden) only because contract tests read them. |
| Per-screen hand-rolled headers (status quo for artifacts/mcp/settings) | Keeps N header implementations drifting apart. New and migrated headers use the shared component; the older destination screens keep their existing statics until their own passes. |

## Consequences

- `MainNavigationBar` renders 12 destinations; Lab is reachable from chrome for the first time, and Models/Speech/Evals have a home.
- Nav active state is truthful on folded screens: on Search the Library button boxes and the header says Search.
- The palette shows one entry per destination; typing a legacy name (`coding`, `media`, `stts`, `logs`, …) finds the owning destination and lands there.
- `TAB_GROUPS` is removed from `Constants.py` (fully dead after this change); `TAB_CODING`, `ALL_TABS`, and `TAB_DISPLAY_LABELS` stay as the legacy-route vocabulary that aliases and startup-config validation read.
- `WORKBENCH_ROUTE_OWNERS` gains a `lab` owner so route-inventory coverage stays complete.
- The Coding feature's CSS (`css/features/_coding.tcss`) is left in place; it now styles nothing live and can be removed in a later cleanup pass.
- Startup configs naming `coding` still work: the route resolves to Console.

## Links

- [UI critique motivating this change](../../.impeccable/critique/2026-07-17T01-08-49Z__k-ui-tab-bar-screen-headers-app-wide-console-first.md)
- [ADR-014: Retire the legacy navigation chrome](014-retire-legacy-navigation-chrome.md)
