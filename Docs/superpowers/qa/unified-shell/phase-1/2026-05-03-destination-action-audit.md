# Phase 1.2 Destination Action Audit

Date: 2026-05-03
Task: `TASK-2.2`
Branch: `codex/unified-shell-phase1-destination-audit`
Base: `origin/dev` at `2d0fd041`

Phase 1.3 update: `TASK-2.3` resolves the false Console-launch affordance findings by disabling generic W+C, Schedules, Workflows, and ACP Console controls until actionable payloads exist. See `2026-05-03-phase-1-3-false-affordance-fix.md`.

## Purpose

Audit every top-level Unified Shell destination for real action ownership, honest disabled states, focus/navigation behavior, and workflow-level usability beyond render/click coverage.

This audit treats "renders" and "posts a navigation event" as partial evidence only. A destination is only considered a working workflow when the user can reach a useful existing surface or stage meaningful context into Console.

## Status Legend

- Working workflow - `working-workflow`: the primary action reaches a useful existing surface, stages meaningful Console context, or performs the promised shell behavior.
- Honest blocked state - `honest-blocked`: the shell clearly says the capability is unavailable or disabled and avoids pretending work was launched.
- False affordance - `false-affordance`: the control is clickable or labeled as executable, but current code only opens a generic placeholder or lacks actionable payload/service backing.

## Running-App QA Evidence

Focused checks were run against Textual `App.run_test(...)` harnesses that mount the actual navigation, destination screens, Home screen, Console handoff card, and app navigation handlers.

- Red test first: python -m pytest Tests/UI/test_unified_shell_destination_action_audit.py -q
- Expected red result before this document existed: `4 failed`
- Mounted shell walkthrough set: python -m pytest Tests/UI/test_master_shell_navigation.py Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_home_screen.py Tests/UI/test_shell_destinations.py -q
- Mounted shell walkthrough result: `61 passed, 3 warnings`
- Warning boundary: dependency/version warnings and splash string syntax warnings were unrelated to shell behavior.

Residual risk: these checks exercise running Textual screens and event handlers, not live LLM provider calls, live scheduler services, ACP runtimes, MCP server sessions, or real local content databases.

## Destination Matrix

| Destination | Primary route | Destination ID | Action owner | Usability status | Classification | Evidence | Follow-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Home | `home` | `home` | `HomeScreen`, `HomeDashboardInput`, `summarize_home_dashboard`, app home control hooks | Dashboard sections and next-best-action routing are verified. Approve, reject, pause, resume, and retry hooks currently notify that no active run service is connected when surfaced. | Honest blocked state - `honest-blocked` | `Tests/UI/test_home_screen.py`; `tldw_chatbook/UI/Screens/home_screen.py`; `tldw_chatbook/app.py` | Phase 2 parent `TASK-4` should wire real active-work adapters before treating controls as operational. |
| Console | `chat` | `console` | `ChatScreen`, `ChatWindowEnhanced`, app `open_chat_with_handoff`, app `open_console_for_live_work` | Console navigation works, pending live-work cards render, and staged context can be delivered to chat. Live model/API success remains outside this shell audit. | Working workflow - `working-workflow` | `Tests/UI/test_master_shell_navigation.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/chat_screen.py` | Phase 3 parent `TASK-3` owns full live-agent control completion. |
| Library | `library` | `library` | `LibraryScreen`, legacy Notes/Media/Ingest/Search/Conversation screens, app chat handoff helper | Notes, Media, Conversations, Import/Export, Search/RAG, and Use in Console actions are mounted and route or stage context as promised. The surface is still a wrapper over legacy Library subroutes. | Working workflow - `working-workflow` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/library_screen.py` | Phase 4 parent `TASK-5` owns Library-native service adoption. |
| Artifacts | `artifacts` | `artifacts` | `ArtifactsScreen`, `ChatbooksScreen`, app chat handoff helper | Chatbooks opens an existing artifact surface and Use in Console stages artifact context. Generated output listing is only explanatory copy. | Working workflow - `working-workflow` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/artifacts_screen.py` | Phase 4 parent `TASK-5` owns generated output service adoption. |
| Personas | `personas` | `personas` | `PersonasScreen`, `ConversationScreen`, app chat handoff helper | Open Personas reaches the existing character/persona/prompt management surface and Attach to Console stages persona context. | Working workflow - `working-workflow` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/personas_screen.py` | Phase 4 parent `TASK-5` owns deeper Persona-native refinement. |
| W+C | `watchlists_collections` | `watchlists_collections` | `WatchlistsCollectionsScreen`, `SubscriptionScreen` | Watchlists routes to the existing subscriptions surface. Collections are described but not actionable. Console follow is disabled with recovery copy until watchlist/collection payloads exist. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`; `2026-05-03-phase-1-3-false-affordance-fix.md` | Resolved by `TASK-2.3`; broader W+C service adoption remains under `TASK-5`. |
| Schedules | `schedules` | `schedules` | `SchedulesScreen` | The screen honestly says no scheduler data is available and disables Console recovery until schedule run payloads are wired. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/schedules_screen.py`; `2026-05-03-phase-1-3-false-affordance-fix.md` | Resolved by `TASK-2.3`; real scheduler adapters remain under `TASK-5`. |
| Workflows | `workflows` | `workflows` | `WorkflowsScreen` | The screen says no workflow service is wired and disables Console launch until workflow execution payloads are wired. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/workflows_screen.py`; `2026-05-03-phase-1-3-false-affordance-fix.md` | Resolved by `TASK-2.3`; real workflow execution remains under `TASK-5`. |
| MCP | `mcp` | `mcp` | `MCPScreen`, existing MCP module surfaces | MCP management is disabled with explicit unavailable copy. `tools_settings` correctly aliases to MCP rather than global Settings. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_shell_destinations.py`; `tldw_chatbook/UI/Screens/mcp_screen.py` | Phase 4 parent `TASK-5` owns embedding or routing the full MCP management surface. |
| ACP | `acp` | `acp` | `ACPScreen` | Launch ACP Agent is disabled with clear runtime-unconfigured copy. Console follow is also disabled until ACP session payloads are wired. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/acp_screen.py`; `2026-05-03-phase-1-3-false-affordance-fix.md` | Resolved by `TASK-2.3`; runtime integration remains under `TASK-5`/`TASK-6`. |
| Skills | `skills` | `skills` | `SkillsScreen`, app chat handoff helper, local skills service hook | Import Skill is disabled with explicit unavailable copy. Attach to Console stages skills context and local skills directory status is visible when available. | Honest blocked state - `honest-blocked` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_console_live_work_handoffs.py`; `tldw_chatbook/UI/Screens/skills_screen.py` | Phase 4 parent `TASK-5` owns list/detail/import/validation adoption. |
| Settings | `settings` | `settings` | `SettingsScreen`, `CustomizeScreen` | Open Appearance reaches the existing customization route. Settings clearly excludes MCP/tool-control ownership. | Working workflow - `working-workflow` | `Tests/UI/test_destination_shells.py`; `Tests/UI/test_shell_destinations.py`; `tldw_chatbook/UI/Screens/settings_screen.py` | No new follow-up from this audit. |

## Findings

- P1 resolved by `TASK-2.3`: W+C, Schedules, Workflows, and ACP no longer expose clickable Console follow/launch actions without actionable live-work payloads.
- P1: Home active-work controls are not operational because the app only exposes placeholder notification hooks. This is already a Phase 2 scope item under `TASK-4`; creating a second broad task here would duplicate the phase plan.
- P2: Library, Artifacts, Personas, Skills, and MCP are mostly compatibility wrappers over legacy or unavailable surfaces. The wrappers are honest enough for Phase 1 except for service-adoption depth, which belongs under `TASK-5`.
- P2: Console is structurally the right live-work hub, but end-to-end agent execution depends on provider/model/runtime state and remains a Phase 3 validation target.

## Follow-on Scope Boundary

Created Backlog child task:

- `TASK-2.3` - Remove false Console-launch affordances from skeletal destinations.

No broader service-adoption Backlog child tasks were created in this audit because those would not be PR-sized. They are already covered by phase parent tasks:

- `TASK-3` - Console Live Work Hub.
- `TASK-4` - Home Operational Control.
- `TASK-5` - Destination Service Adoption.
- `TASK-6` - Capability And Recovery System.

## QA Conclusion

Phase 1.2 moved the shell from "we have render/click tests" to "we know which destination actions are real, blocked, or misleading." `TASK-2.3` removes the known false Console-launch affordances; Phase 1 still needs a final shell-contract replay before it is marked verified.
