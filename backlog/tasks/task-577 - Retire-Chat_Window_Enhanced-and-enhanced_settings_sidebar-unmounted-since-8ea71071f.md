---
id: TASK-577
title: Retire Chat_Window_Enhanced and enhanced_settings_sidebar (unmounted since 8ea71071f)
status: To Do
assignee: []
created_date: '2026-07-25 15:10'
labels:
  - chat
  - dead-code
  - tech-debt
dependencies:
  - task-562
---
## Description

Follow-up to task-562 / ADR-026. The task-562 scout established that the entire
`ChatWindowEnhanced` surface has been unmounted since commit `8ea71071f`
("Move Console transcript and composer to native surface", 2026-05-06):
`_ensure_chat_window()` (chat_screen.py) has zero callers, `self.chat_window`
stays `None` for the process lifetime, and `#chat-window` / `#chat-log` /
`EnhancedSettingsSidebar` never exist in the live tree. task-562 retired the
conversation-entry chain but deliberately KEPT the window family to bound its
blast radius.

Remaining retirement audit (~2,600+ production LOC + ten test suites):
`UI/Chat_Window_Enhanced.py` (~1,163), `Widgets/enhanced_settings_sidebar.py`
(~1,429), `UI/Chat_Modules/`, the `use_enhanced_window` config flag +
Tools/Settings checkbox (a no-op toggle today), the `#chat-window` dead-end
consumers (`app.py`, `worker_events.py`, `chat_events.py`, `chat_events_tabs.py`),
the chat_events send-path liveness question (the `use_enhanced_window` reads at
chat_events.py ~:792/:1067/:1125/:1266/:1661/:2760 and the tab wrappers in
`chat_events_tabs.py` :99-294 serve surfaces that may all be unreachable), the
chat-tabs subsystem (`ChatTabContainer`/`chat_session.py` — composed only inside
the unmounted window), plus any unit task-562's gates DEFERRED (recorded in
task-562's Implementation Notes). Same method as task-562: per-unit grep-gates
(ids composed nowhere live + zero direct callers), retirement-guard pins in
`test_legacy_entrypoints_retired.py`, defer on gate failure.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every unit above is either deleted behind a passing grep-gate or explicitly recorded as live/deferred with the gate evidence
- [ ] #2 Retirement-guard pins cover the deleted modules and symbols (test_legacy_entrypoints_retired.py pattern)
- [ ] #3 No live behavior regresses: full test suite green, app boots, Console chat and all cross-screen handoffs unaffected
<!-- AC:END -->
