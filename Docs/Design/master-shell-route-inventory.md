# Master Shell Route Inventory

Date: 2026-05-03

## Purpose

This inventory maps current routes and UI surfaces onto the approved master shell IA. It is a compatibility ledger for the migration from legacy tab labels to the new destination model.

## Destination Map

| Master destination | Legacy routes | Existing screen/wrapper | Current user-facing label | Compatibility requirement |
| --- | --- | --- | --- | --- |
| Home | `home` | new | Home | New default for first-run users |
| Console | `chat` | `ChatScreen` | Console | Route remains `chat`; live work stores `pending_console_launch` and opens Console |
| Library | `notes`, `media`, `ingest`, `search`, `conversation`, conversation browsing | `LibraryScreen` plus legacy screens | Library | Wrapper links to source routes; staged source context uses Chat handoff payloads |
| Artifacts | `chatbooks` | `ArtifactsScreen` plus `ChatbooksScreen` | Artifacts | Wrapper owns Chatbooks and generated/portable outputs; staged artifacts use Chat handoff payloads |
| Personas | `ccp`, character/persona/prompt/lore subviews | `PersonasScreen` plus `ConversationScreen` | Personas | Personas owns behavior and identity management; staged persona context uses Chat handoff payloads |
| W+C | `subscriptions` plus collections services | `WatchlistsCollectionsScreen` plus `SubscriptionScreen` | W+C | Wrapper separates Watchlists from Collections and can follow live work in Console |
| Schedules | schedule surfaces | `SchedulesScreen` | Schedules | Wrapper owns when-runs and can follow timing/recovery work in Console |
| Workflows | workflow surfaces | `WorkflowsScreen` | Workflows | Wrapper owns what-runs and can launch live work in Console |
| MCP | `tools_settings`, tools/MCP settings | `MCPScreen` | MCP | Wrapper owns MCP tool/server capability control; `tools_settings` is an MCP alias, not global Settings |
| ACP | ACP surfaces | `ACPScreen` | ACP | Wrapper shows honest ACP runtime-unconfigured state and can follow agent work in Console |
| Skills | skills services | `SkillsScreen` | Skills | Wrapper exposes Agent Skills pack boundaries, `SKILL.md`, local skills dir, and staged skill context |
| Settings | `settings`, `customize` | `SettingsScreen` plus `CustomizeScreen` | Settings | Settings owns global preferences, appearance, accounts/auth, storage, and app behavior |

## Shortcut And Command Palette Inventory

| Shortcut/command | Current target | Master destination | Keep/change |
| --- | --- | --- | --- |
| Command palette tab navigation | `ALL_TABS` direct tab IDs | Console plus legacy direct routes | Keep all direct legacy commands searchable while adding primary shell destinations first |
| Chat route | `chat` | Console | Keep route ID; change user-facing label to Console |
| Tools/settings route | `tools_settings` | MCP | Keep as legacy MCP alias, not global Settings |
| Models shortcut | `llm` alias and `llm_management` tab ID | Legacy direct command | Keep alias to `TAB_LLM` until a later Models/MCP decision |
| Subscription route | `subscription`, `subscriptions` | W+C | Keep both aliases |

## Import/Export Boundary

Library import/export means source import/export.

Artifacts import/export means bundle/output import/export.

## Design-System Boundary

Top navigation is global primary destination navigation only.

Destination context, source authority, readiness, and recovery belong inside destination headers or local panels.

Status labels, source authority, approvals, staged source roles, recovery callouts, and shortcuts use the agentic terminal design-system contract.

Top navigation may use compact labels where terminal width requires it, but `Home` and `Console` must remain visible, full destination names must remain available through tooltips and command palette, and overflow must be explicit rather than silently hiding destinations.

## Deferred Surfaces

- ACP launch remains an honest unavailable/capability state until an ACP runtime is configured.
- Workflow runtime execution is not implemented in the shell wrapper; the wrapper exposes ownership and Console launch boundaries only.
- Rich MCP management still lives in the existing Unified MCP panel/service surface; the top-level MCP wrapper prevents `tools_settings` from acting as global Settings.
- Library sub-surfaces remain linked to legacy Notes, Media, Ingest, Search/RAG, and conversation screens until those screens are split into Library-native views.
