---
id: TASK-280
title: >-
  Gate the Console 0.2s sync tick (stop per-tick DB queries and unconditional
  recomposes)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 03:23'
labels:
  - performance
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During any active run, the 0.2s _poll_transcript tick runs 10 sub-syncs, most unconditional: _sync_console_workspace_context re-queries the DB per scope per tick ON the event loop (measured 11.2ms/tick @3k convs/8 workspaces; ~70ms/tick with an active search) and ConsoleWorkspaceContextTray.sync_state recomposes unconditionally; ConsoleSettingsSummary/agent-section/system-line update ~13 Statics per tick with no equality guard; ConsoleRunInspector recomposes wholesale per change (streaming-excerpt selection = 5 teardowns/s) and syncs while hidden. In-file templates exist: the subagent-badge TTL cache (chat_screen 4037-4068) and RunInspector's own equality guard. Preserve: transcript fingerprint gate, incremental row reconciler, twice-per-turn persist. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No DB query executes from the 0.2s tick when the conversation set is unchanged (TTL/signature gate or removal from the tick)
- [x] #2 Settings-summary/agent/system-line/guidance sub-syncs no-op when their state is unchanged (equality guards)
- [x] #3 Inspector no longer full-recomposes per streaming tick when the streaming message is selected (static "Streaming…" excerpt). AMENDED: the skip-while-hidden half was implemented, broke 6 pinning tests, and was REVERTED — Console keeps hidden inspector content fresh by design (display:none subtrees stay queryable and flows depend on it); a regression test now pins that contract
- [x] #4 Live QA: streaming remains smooth and all synced surfaces still update on real changes (send/finish/rename/switch)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TTL cache (CONSOLE_PERSISTED_ROWS_CACHE_TTL_SECONDS=2.0, Finding-A pattern) wraps _sync_persisted_console_browser_rows keyed (query, current_conversation_id), invalidated on submit-return, run-leaves-active, search change, rename, delete, new-tab — per-tick DB queries during streams drop from ~30/s to ≤0.5/s. Equality no-op guards: ConsoleSettingsSummary.sync_state, ConsoleWorkspaceDetailsTray.sync_state, agent-section/system-line/guidance payload caches, rail-state computed once per tick. Workspace tray recompose gated at the SCREEN-side call site (tick path only). Inspector: static 'Streaming…' excerpt during streams (deliberate UX change, user-approved at gate). Two brief items reverted with root causes + pinning tests: inspector hidden-skip (hidden-DOM freshness is a design contract — 6 tests) and widget-level tray equality guard (recompose is load-bearing for click-target layout; screen-side compare achieves the win). Live QA: gating transparent — rail 2.8s post-finish, rename 0.5s, excerpt static→real, 0 stalls, 0 worker-cancels. Suites 1048/69/0. Files: UI/Screens/chat_screen.py, Widgets/Console/console_settings_summary.py, console_workspace_context.py, console_workspace_details.py, Tests/UI/test_console_tick_gating.py. Evidence: Docs/superpowers/qa/console-tick-2026-07/.
<!-- SECTION:NOTES:END -->
