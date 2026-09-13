---
id: TASK-24601
title: Live work sources reports hardcoded Connected for five of seven rows
status: Done
assignee:
  - '@claude'
created_date: '2026-08-30 00:53'
updated_date: '2026-08-30 01:39'
labels:
  - console
  - ux
  - inspector
  - critique-2026-08-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under a heading that reads as measured readiness, Watchlists, Workflows, Schedules, RAG and Artifacts are literal status="Connected" string constants; only ACP derives from a runtime snapshot and MCP is a constant "Not wired". from_acp_runtime_status is the sole builder, so no code path ever measures the other five. A user who discovers this has reason to distrust every other status line in the rail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No row in Live work sources displays a readiness word that was not derived from a runtime check
- [x] #2 Sources that are not probed render an explicit not-checked state rather than Connected
- [x] #3 A test fails if a readiness row's status is a literal in the builder rather than derived from an input
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The card mixed three different kinds of claim under one heading that reads as measured readiness, and used one word for all of them. Separating the kinds is the fix; probing everything was neither possible nor meaningful.

- Probed connections (ACP, MCP) are the ONLY rows that may say 'Connected', and only from a real input. MCP was previously the literal 'Not wired'; it now derives from the caller's tool count: None -> 'Not checked', 0 -> 'Not wired', N -> 'Connected - N tools ready.'
- RAG is a local capability gated on optional extras, so 'Connected' there was not merely unmeasured, it could be FALSE with the extras absent. It now reports Ready / Unavailable / Not checked from DEPENDENCIES_AVAILABLE['embeddings_rag'].
- Watchlists, Workflows, Schedules and Artifacts are in-app handoff DESTINATIONS -- navigation targets that always exist locally. There is nothing to probe. They now say 'Available', which is true, instead of 'Connected', which claimed a connection that was never made.

AC interpretation worth stating: 'no row displays a readiness word not derived from a runtime check' is enforced as 'no row says Connected without evidence'. 'Available' remains a literal, deliberately -- it is a statement about a static in-app capability, not a readiness claim, and the docs now say so.

An existing test asserted 'Watchlists: Connected - Home run details.' and friends: it pinned the defect rather than catching it, and was updated. DEPENDENCIES_AVAILABLE is imported lazily inside the method, not at module scope, because this screen is on the boot path and ADR-097's import-weight budget is a ratchet.

Pre-existing dev failures in this area, verified identical on a pristine origin/dev worktree and NOT caused by this change: 4 in test_console_live_work_handoffs.py and 1 in test_unified_shell_phase6_first_time_replay.py (Textual SelectCurrent '#label' NoMatches at mount).

Modified: tldw_chatbook/Chat/console_live_work.py, tldw_chatbook/UI/Screens/chat_screen.py, Docs/User_Guide/console/context-and-rag.md, Docs/Development/release-recovery-setup.md, and 4 test files.
<!-- SECTION:NOTES:END -->
