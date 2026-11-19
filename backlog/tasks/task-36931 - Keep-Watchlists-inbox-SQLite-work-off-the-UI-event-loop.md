---
id: TASK-36931
title: Keep Watchlists inbox SQLite work off the UI event loop
status: Done
created_date: 2026-10-25 23:37
labels:
- textual-book
- watchlists
- notifications
- performance
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md
assignee:
- '@codex'
updated_date: 2026-10-26 03:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the application responsive while the Watchlists inbox reads or updates its local notification store, including when another SQLite writer is active.

### Verified scope

Verified on merged dev `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05` after PR #3248. With the production mounted destination and an explicit private file-backed ClientNotificationsDB, a real competing writer held for 0.7 seconds made Mark read execute on the main thread for about 0.75 seconds and delayed a 10ms heartbeat by about 0.84 seconds. The ordinary control also persisted the read flag; it is a correctness control, not a storage-speed guarantee. A later merged-tree uncontended repeat measured a 1.31-second write and a 1.91-second heartbeat gap on the same UI thread. No fake slow DB result was injected; natural contention frequency and native terminal frames are unmeasured. Source also shows list and dismiss evaluated synchronously before _maybe_await.

Affected source: tldw_chatbook/UI/Subscription_Modules/notifications_inbox_controller.py:79-90,141,156-158; UI/Screens/watchlists_collections_screen.py:1209-1211,7940-8066.

Book: Chapter 17 PDF335 (printed326), threaded work for synchronous blocking calls; Chapter 7 PDF145-146/158 (printed136-137/149), async handlers alone do not make blocking work safe.

### Existing ownership and decisions

TASK-33283 owns incremental panes and the original PERF-24 Schedules DB/automation-result Mark-all-read sites. Its original appendix names Notifications table recomposition but no inbox DB offload. TASK-15466 covers connection lifecycle, TASK-282 Home reads, TASK-21238 corrupt-store isolation, and TASK-19559/19870 worker-group/refresh ordering; retain those contracts.

Related tasks: [TASK-33283](./task-33283%20-%20PERF-24-Watchlists-and-Schedules---incremental-panes-DB-off-the-event-loop.md), [TASK-15466](./task-15466%20-%20Port-the-held-connection-idiom-to-Library_Collections-RAG_Indexing-and-client_notifications.md), [TASK-282](./task-282%20-%20Home-dashboard-thread-and-cache-seam-queries.md), [TASK-21238](./task-21238%20-%20A%20corrupt%20notifications%20database%20can%20now%20fail%20unrelated%20Media%2C%20Research%20and%20Watchlists%20flows.md), [TASK-19559](./task-19559%20-%20exclusive%3DTrue%20without%20group%3D%20makes%20every%20worker%20on%20a%20node%20cancel%20its%20siblings.md), [TASK-19870](./task-19870%20-%20Mutation-workers-refresh-inline-instead-of-dispatching-into-the-loader-group.md), [TASK-36900](./task-36900%20-%20Keep-Watchlists-navigation-responsive-during-article-body-reads.md).

ADR required: no.
ADR paths: [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md), [ADR-042](../decisions/042-watchlists-reader-first-ia.md).
Reason: Use existing controller/store boundaries and preserve policy gates plus file-backed and in-memory connection semantics; no new storage architecture or dependency is proposed.

[Post-merge audit and reproducible evidence](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md).
<!-- SECTION:DESCRIPTION:END -->
## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 File-backed inbox list, mark-read and dismiss operations leave the UI loop responsive during controlled slow reads or competing SQLite writes.
- [x] #2 Ordinary mark-read and dismiss operations persist the intended notification state and refresh through the established inbox loader ownership.
- [x] #3 Existing policy refusals, corrupt/unavailable-store feedback and nonfatal worker behavior remain intact, without reporting a failed mutation as successful.
- [x] #4 Targeted coverage verifies file-backed responsiveness and compatible in-memory behavior without violating SQLite connection ownership or weakening lifecycle guards.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add controlled regression tests and observe the reported defect before production edits.
2. Dispatch synchronous file-backed ClientNotificationsDB list/read/dismiss calls with asyncio.to_thread before evaluation. Keep thread-affine in-memory stores valid and retain awaitable seams, policy gates, nonfatal errors and existing loader serialization. Test real private file-backed durable state and controlled blocking/competing writes, in-memory compatibility and refusal/error paths. Do not repair Notifications pane rebuilding (TASK-33283).
3. Run focused regression and adjacent owner tests, lint/format checks, update Docs/Architecture/notifications.md, and self-review.
4. Complete Superpowers review, register new UI coverage in the existing gate and verify CI before closure.

ADR required: no
ADR path: existing canonical ADRs 011, 033, 042 (linked in Description).
Reason: bounded implementation of existing UI/session/store ownership; no new contract or architecture.

Implementation plan: Docs/superpowers/plans/2026-10-25-textual-followup-fixes.md (Task 2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Canonical bound list/read/dismiss calls on the exact file-backed notifications repository now run through the existing finite-worker ownership helper. In-memory, custom synchronous and awaitable stores retain their caller-thread/loop contracts; policy gates and nonfatal error handling are preserved. Each admitted screen mutation keeps its completion-to-refresh path using noncancelling workers, while the shared loader retains latest-wins ownership. This bounded admission correction addresses the whole-branch review's native-write/awaiter cancellation race.

Changed UI/Subscription_Modules/notifications_inbox_controller.py, the two mutation admissions in UI/Screens/watchlists_collections_screen.py and Docs/Architecture/notifications.md; added Tests/UI/test_notifications_inbox_responsiveness.py. All 12 final cases pass (6 actual mounted private children plus 6 contracts), alongside 11 existing controller/store/nonfatal cases and 7 native finite-worker lifecycle outcomes. Real SQLite two-row A-held/B-commits-first regressions for Mark read and Dismiss failed at the intended stale-table assertion before the two-line fix and pass after it. Screen length is unchanged; no new lint diagnostics. The historical source-path typo is corrected. TASK-33283 retains pane rebuilding/focus ownership.

ADR required: no. Existing [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md) and [ADR-042](../decisions/042-watchlists-reader-first-ia.md) apply; no storage, connection or service boundary changes.

Superpowers task, whole-branch, fix-wave and CI-delta reviews are complete with no open findings; Qodo was replaced as requested. Current implementation-head hosted CI is green: [40261190139](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190139); [40261190218](https://github.com/rmusser01/tldw_chatbook/actions/runs/40261190218). Its 17 UI shards executed all 36 new exact cases once; maximum wall time 1094s under the unchanged 1200s cap. The gate preserves all 899 live-dev selectors plus 36 new cases (floor935). Rebase onto dev e554f25812 preserved all nine reviewed source/test blobs.

[Verification and explicit scope decisions](../../Docs/superpowers/qa/2026-10-25-textual-followup-fixes.md) records exact targeted evidence and the retained baseline profile/bootstrap, Chat-size and initial Chunking timing qualifications. No full local suite or native/live-server qualification is claimed.

[PR #3251](https://github.com/rmusser01/tldw_chatbook/pull/3251) targets dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
