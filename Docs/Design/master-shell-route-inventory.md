# Master Shell Route Inventory

Date: 2026-05-03

## Purpose

This inventory maps current routes and UI surfaces onto the approved master shell IA. It is a compatibility ledger for the migration from legacy tab labels to the new destination model.

## Destination Map

| Master destination | Legacy routes | Existing screen/wrapper | Current user-facing label | Compatibility requirement |
| --- | --- | --- | --- | --- |
| Home | `home` | new | Home | New default for first-run users |
| Console | `chat` | `ChatScreen` | Chat | User-facing label becomes Console; route remains `chat` |
| Library | `notes`, `media`, `ingest`, `search`, `conversation`, conversation browsing | multiple | Notes/Media/Ingest/Search | New wrapper links to source routes; `conversation` means saved conversation browsing/source access |
| Artifacts | `chatbooks` | `ChatbooksScreen` | Chatbooks | New wrapper owns Chatbooks and generated/portable outputs |
| Personas | `ccp`, character/persona/prompt/lore subviews | `ConversationScreen` | Library | User-facing label becomes Personas for behavior and identity management |
| Watchlists+Collections | `subscriptions` plus collections services | `SubscriptionScreen` | Subscriptions | New wrapper has Watchlists and Collections sections |
| Schedules | schedule surfaces | existing scheduler surfaces | mixed | New wrapper owns when-runs |
| Workflows | workflow surfaces | future/existing workflow code | mixed | New wrapper owns what-runs |
| MCP | `tools_settings`, tools/MCP settings | `ToolsSettingsScreen` and MCP panels | legacy Settings/tools label | New wrapper owns MCP/tool capability control; `tools_settings` becomes an MCP alias |
| ACP | ACP surfaces | new | ACP | New wrapper with honest unavailable state if needed |
| Skills | skills services | new | Skills | New wrapper around local/server skills |
| Settings | `settings`, `customize` | existing screens | Settings/Customize | Settings owns global app preferences only; do not route global preferences through `tools_settings` |

## Shortcut And Command Palette Inventory

| Shortcut/command | Current target | Master destination | Keep/change |
| --- | --- | --- | --- |
| Command palette tab navigation | `ALL_TABS` direct tab IDs | Console plus legacy direct routes | Keep all direct legacy commands searchable while adding primary shell destinations first |
| Chat route | `chat` | Console | Keep route ID; change user-facing label to Console |
| Tools/settings route | `tools_settings` | MCP | Keep as legacy MCP alias, not global Settings |
| Models shortcut | `llm` alias and `llm_management` tab ID | Legacy direct command | Keep alias to `TAB_LLM` until a later Models/MCP decision |
| Subscription route | `subscription`, `subscriptions` | Watchlists+Collections | Keep both aliases |

## Import/Export Boundary

Library import/export means source import/export.

Artifacts import/export means bundle/output import/export.

## Design-System Boundary

Top navigation is global primary destination navigation only.

Destination context, source authority, readiness, and recovery belong inside destination headers or local panels.

Status labels, source authority, approvals, staged source roles, recovery callouts, and shortcuts use the agentic terminal design-system contract.

## Open Questions For Implementation

- None. Add only implementation-time discoveries here.
