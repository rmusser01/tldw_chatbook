---
id: TASK-333
title: Fix incorrect and stale developer documentation
status: To Do
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
- [ ] #1 Schema version corrected: ChaChaNotes is v21 (`DB/ChaChaNotes_DB.py:143`), not "v7"
- [ ] #2 Tool calling documented as implemented (execution is wired in `worker_events.py:407` and `chat_streaming_events.py:211`), not "execution pending"
- [ ] #3 "New LLM Provider" recipe points at `chat_api_call()` (`Chat/Chat_Functions.py:646`) + `chat_with_<provider>()`, not the removed `chat_with_provider()`
- [ ] #4 "New Tool" recipe uses the property-based `Tool` ABC + `register_tool()` (`tool_executor.py:27-44,265`), not `get_*()`/`AVAILABLE_TOOLS`
- [ ] #5 Splash count/location corrected (~90 effects under `Utils/Splash_Screens/`; `Utils/splash_animations.py` is a compat shim)
- [ ] #6 Pre-commit hook path corrected to `Helper_Scripts/fixed_auto_review.py`
- [ ] #7 Data-layer DB list updated to include AgentRuns/Workspace/Library/Research/Writing/Mindmap and the other DB modules
- [ ] #8 The `Agents/` runtime gains an Architecture subsection (control loop + `chat_api_call`/tool-provider seam)
- [ ] #9 The React/Tailwind "gotchas" (localStorage, Tailwind) are removed as irrelevant doc-rot in a Python TUI
<!-- AC:END -->
