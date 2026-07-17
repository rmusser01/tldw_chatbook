---
id: TASK-280
title: Gate the Console 0.2s sync tick (stop per-tick DB queries and unconditional recomposes)
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, console]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During any active run, the 0.2s _poll_transcript tick runs 10 sub-syncs, most unconditional: _sync_console_workspace_context re-queries the DB per scope per tick ON the event loop (measured 11.2ms/tick @3k convs/8 workspaces; ~70ms/tick with an active search) and ConsoleWorkspaceContextTray.sync_state recomposes unconditionally; ConsoleSettingsSummary/agent-section/system-line update ~13 Statics per tick with no equality guard; ConsoleRunInspector recomposes wholesale per change (streaming-excerpt selection = 5 teardowns/s) and syncs while hidden. In-file templates exist: the subagent-badge TTL cache (chat_screen 4037-4068) and RunInspector's own equality guard. Preserve: transcript fingerprint gate, incremental row reconciler, twice-per-turn persist. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P1 B1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No DB query executes from the 0.2s tick when the conversation set is unchanged (TTL/signature gate or removal from the tick)
- [ ] #2 Settings-summary/agent/system-line/guidance sub-syncs no-op when their state is unchanged (equality guards)
- [ ] #3 Inspector skips sync while hidden and no longer full-recomposes per streaming tick when the streaming message is selected
- [ ] #4 Live QA: streaming remains smooth and all synced surfaces still update on real changes (send/finish/rename/switch)
<!-- AC:END -->
