---
id: TASK-15473
title: Timer gating: non-blocking Ollama probe and change-gated nav overflow tick
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two verified steady-state loop-stealers from the audit. (1) The Models screen's 3 s status timer ends in a BLOCKING `socket.create_connection(("127.0.0.1", 11434), timeout=0.25)` on the event loop when no app-owned Ollama process exists (`UI/LLM_Management_Window.py:525` -> `UI/Screens/llm_screen.py:90-98`) — instant on ECONNREFUSED but up to 250 ms of frozen UI per probe if the port blackholes (firewalled/container setups). (2) The nav bar re-measures overflow hints every 0.5 s forever on the active screen and schedules two extra callbacks per tick (`UI/Navigation/main_navigation.py:396/:445/:598`) — scroll math, hint toggles, re-center, ghost-button geometry — with no change detection, on every screen, app-lifetime.

Fix direction: probe via `asyncio.open_connection` (same 0.25 s cap) or a thread; nav tick gets a cheap change signature (scroll_x, container width, button count) and skips no-op ticks — or becomes resize/scroll-event-driven if that stays simple. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No blocking connect ever runs on the event loop (evidence); Models availability UX unchanged
- [ ] #2 The nav tick performs no measurement/ghost work when nothing changed (evidence); hints still correct after resize, overflow, and scroll (tests)
<!-- AC:END -->
