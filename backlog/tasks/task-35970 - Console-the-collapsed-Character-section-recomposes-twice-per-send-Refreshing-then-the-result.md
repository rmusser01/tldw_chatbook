---
id: TASK-35970
title: >-
  Console: the collapsed Character section recomposes twice per send
  (Refreshing..., then the result)
status: To Do
assignee: []
created_date: '2026-10-21 04:37'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while measuring TASK-33620.15.7 (2026-10-20, live, 160x45, a 36-turn chat, Anthropic claude-haiku-4-5 through the review proxy). After the run-tick fixes there, about 12-13 whole-screen reflows remain per send window; two of them come from ConsoleCharacterContext alone. Every send ends with a Character refresh that publishes a REFRESHING state (loading=True) and then the loaded result (data_revision and scope_fingerprint move), and ConsoleCharacterContext.sync_state recomposes the whole section, search Input included, for each, whether or not the Character section is expanded. Measured: 2 recomposes per send live (8.4-74 ms reflows each); in the mounted test harness the refresh errors and the section recomposed every few poll ticks while a run was held (Tests/UI/test_console_run_tick_layout_requests.py exempts it and names this task).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send with the Character section collapsed causes no Character-section recompose or layout request
- [ ] #2 Expanding the Character section after such a send shows the current chats with no stale Refreshing line
- [ ] #3 The held-run layout-request test in Tests/UI/test_console_run_tick_layout_requests.py no longer exempts the Character section
<!-- AC:END -->
