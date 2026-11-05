---
id: TASK-38949
title: "Console cold-config deferral entry outlives the config fix until another owner reads warm config"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- Tests/UI/test_console_projection_reads_off_loop.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-36511.4's notes list it as "Not changed (proposed follow-up)". The helper is unchanged since 983f363c8c (#3385).

- `defer_cold_live_read` keeps its per-owner episode set, `_console_cold_read_deferrals`, in `tldw_chatbook/UI/Console_Modules/console_spend_projection.py:1170-1188`. It clears the set (`:1183`) only when some owner calls it while config is warm, checks for re-entry at `:1185`, and adds the owner at `:1187`.
- The credential poll returns before its locked section unless its pre-gate changed (`tldw_chatbook/UI/Screens/chat_screen.py:10779`; the call is at `:10819`). The other callers are `config_sync.py:289` and `console_spend_projection.py:1220`.
- So once a broken config is fixed, an owner's entry can stay until another owner happens to read warm config. If config goes cold again before that, the owner's first cold read reads live on the UI loop instead of deferring. TASK-36511.4 and TASK-36511 measured that read at 147-222 ms for the poll and about 80 ms for the identity refresh.
- The existing test covers only the path where a warm call clears the set (`Tests/UI/test_console_projection_reads_off_loop.py:491`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Once config is warm again, whichever owner or path warmed it, a new cold episode makes each owner's first cold read defer even when no owner was called while config was warm; a unit test in Tests/UI/test_console_projection_reads_off_loop.py is red on dev
- [ ] #2 The bound still holds: with an unparseable config.toml each owner defers at most once per episode, and the existing deferral tests pass, including test_an_unparseable_config_leaves_an_idle_console_idle and test_the_credential_poll_defers_a_cold_read_once_and_then_stays_idle
<!-- AC:END -->
