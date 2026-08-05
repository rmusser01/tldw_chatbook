---
id: TASK-333
title: Fix incorrect and stale developer documentation
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [docs]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CLAUDE.md and README contain claims that contradict the code and will actively mislead contributors following the "Adding Features" recipes. Grouped as one documentation pass (naturally a single PR); each item is an acceptance criterion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schema version corrected: pinned to the `_CURRENT_SCHEMA_VERSION` symbol in `DB/ChaChaNotes_DB.py` (not a bare integer, so it won't re-rot on the next schema bump)
- [x] #2 Tool calling documented as fully implemented (detection + execution wired in `worker_events.py:407` and `chat_streaming_events.py:211`)
- [x] #3 "New LLM Provider" recipe points at `chat_api_call()` (`Chat/Chat_Functions.py:646`) + API_CALL_HANDLERS/PROVIDER_PARAM_MAP dispatched correctly
- [x] #4 "New Tool" recipe uses the property-based `Tool(ABC)` with name/description/parameters properties + async execute + `ToolExecutor.register_tool()`/`get_tool_executor()`
- [x] #5 Splash count/location corrected (~90 effects under `Utils/Splash_Screens/`; `Utils/splash_animations.py` is a compat shim)
- [x] #6 Pre-commit hook path corrected to `Helper_Scripts/fixed_auto_review.py` (not deprecated `auto_review.py`)
- [x] #7 "Main Windows" section corrected: Screen-per-tab pattern via `UI/Screens/*` + `UI/Navigation/screen_registry.py` (Chat_Window_Enhanced and SearchRAGWindow are Containers; all others extend Screen)
- [x] #8 File paths corrected: `model_capabilities` is top-level `tldw_chatbook/model_capabilities.py` (NOT under `LLM_Calls/`); `custom_splash_cards` at `Helper_Scripts/Examples/custom_splash_cards/`
- [x] #9 Data-layer DB list updated to include AgentRuns/Workspace/Library/Research/Writing/Mindmap and all other DB modules
- [x] #10 The `Agents/` subsection expanded with Architecture details (control loop + `chat_api_call`/tool-provider seam)
- [x] #11 The React/Tailwind "gotchas" (localStorage, Tailwind) removed as irrelevant doc-rot in a Python TUI
- [x] #12 README.md verified clean (no conflicting or stale claims); no changes required
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Comprehensive documentation corrections applied to CLAUDE.md:

**Schema**: pinned to the `_CURRENT_SCHEMA_VERSION` symbol in `DB/ChaChaNotes_DB.py` rather than a bare integer, so the doc tracks the code automatically as the schema evolves.

**Tool System**: Tool detection + execution both implemented and documented; `execute_tool_call()` wired into both `worker_events.py:407` (agent-event dispatch) and `chat_streaming_events.py:211` (streaming message handling); updated "New Tool" recipe to reflect current `Tool(ABC)` pattern with name/description/parameters properties + async execute + `ToolExecutor.register_tool()`/`get_tool_executor()`.

**Provider Integration**: "New LLM Provider" recipe corrected to show real pattern: `chat_with_<provider>()` functions + API_CALL_HANDLERS/PROVIDER_PARAM_MAP registries dispatched by `chat_api_call()` (`Chat/Chat_Functions.py:646`).

**UI Architecture**: "Main Windows" section clarified: all Screens registered via `UI/Navigation/screen_registry.py`; `Chat_Window_Enhanced` and `SearchRAGWindow` are Containers; all others extend Screen directly.

**File Paths**: `model_capabilities` corrected to top-level `tldw_chatbook/model_capabilities.py` (NOT under `LLM_Calls/`); pre-commit hook corrected to `Helper_Scripts/fixed_auto_review.py`; custom-card reference corrected to `Helper_Scripts/Examples/custom_splash_cards/`; splash effects corrected to `Utils/Splash_Screens/` (`splash_animations.py` is a compat shim).

**Database Modules**: expanded DB list to include AgentRuns_DB, Workspace_DB, Library_Collections_DB, Library_Ingest_Jobs_DB, Research_DB, Writing_DB, Mindmap_DB, search_history_db, Sync_Client, plus the existing Evals/Prompts/Subscriptions modules.

**Agents Runtime**: added Architecture subsection documenting control loop, `chat_api_call` dispatcher, tool-provider seam, and event-handler integration.

**React/Tailwind Gotchas**: removed (py-only TUI, not relevant).

**README.md**: verified clean; no corrections required (contains only general project pitch, not feature recipes).

Files: `CLAUDE.md` (comprehensive multi-section pass). All ACs satisfied via code review + path verification.
<!-- SECTION:NOTES:END -->
