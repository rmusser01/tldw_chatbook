---
id: TASK-32673
title: Stop and revoke plugin work within the requested scope
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:20'
updated_date: '2026-09-16 16:11'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32672
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users immediately stop plugin activity while preserving unrelated authorized work and reporting persistence honestly.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disable here, global-default changes, Disable everywhere and uninstall invalidate exactly their specified scopes; old approvals and callbacks never revive on re-enable.
- [ ] #2 Live admission closes and host cancellation starts before waiting for trust unlock or storage writes; affected plugin cleanup callbacks are suppressed immediately.
- [ ] #3 Durable disable/removal success is distinct from a session-only block and confirmed local stop; retry and failure cannot reopen the live scope or release surviving resource ownership.
- [ ] #4 Uninstall removes only installation-owned grants and registrations, retains data by default and leaves independent credentials/connections owned by their existing services.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Direct implementation of the accepted scoped revocation, retained lifecycle cleanup and authenticated persistence contracts. No separate permission/execution runtime or new ADR is proposed.

1. Read task-6-brief.md, the final reviewed F5 report and exact admission/service/run-owner APIs, lifecycle research, and the relevant testing/isolation lessons. Confirm current-scope authority and mutation ownership before changing code.
2. Establish RED through real PluginService/coordinator revocation with two authorized workspaces, an owned controlled child, exact captured run/approval identities and a blocked storage worker. Demonstrate the same production entry succeeds before the fence; isolate all profile/keyring/process fixtures.
3. Add Plugins/revocation.py and minimal admission/service/coordinator/runtime-owner integration. Express Disable here, global-default edit, Disable everywhere and uninstall as distinct validated requests. Seal affected live generations and transfer exact cancellation to retained host cleanup before any trust/storage/coordinator await. Retain terminal evidence independently of caller cancellation; preserve unrelated authorized work.
4. Use the existing certificate publication pipeline for durable scoped activation, revocation and uninstall tombstones. Keep session-only block, durable outcome, confirmed stop and cleanup pending separate. Retries cannot reopen admission, and a later explicit enable cannot revive old approval/output/callback generations. Keep data by default and remove only installation-owned registrations/grants.
5. R17 allows minimal Chat/console_chat_controller.py, Chat/console_agent_bridge.py and Agents/agent_service.py adapters if F5's retained callbacks do not already suffice; console_runtime.py only if it is an actual owner seam. Record exact used paths before code. F5's Plugins/service.py owns the persistent worker and live runtime custody; synchronous live sealing must remain independent of that worker being busy. MCP request/connection ownership remains M4, hook-v2 consumers remain their planned tasks, with explicit fenced interfaces/handoffs rather than duplicate runtimes.
6. Run Tests/Plugins/test_revocation.py and test_revocation_persistence.py, plus exact affected F5/Console/approval/lifecycle regressions chosen from changed code. Cover failed/locked persistence, cancelled waiter, surviving and confirmed-dead child, A/B/default inheritors, stale re-enable approvals and idempotent uninstall. No full suite. Run changed-file syntax/Ruff/Black and git diff --check, preserving established shared-file baselines rather than reformatting unrelated code.
7. Update plugin operation documentation, report concrete output and qualification limits in task-6-report.md, self-review, then return exact paths for controller commit/review. Controller closes acceptance criteria/Done only after independent clean review and resolves cross-task cannot-verify items explicitly.
8. R36: minimally extend Plugins/review.py with closed revoke/uninstall review kinds and additive independent operation outcome dimensions, reusing the one existing coordinator commit/certificate pipeline and existing authority OperationResult/Tombstone schema. Retain original persistence/cancellation failure if cleanup also fails, with content-free public diagnostics; ephemeral runtime outcomes do not become authenticated commitment proof. Update Plugins/README.md operational guidance. Disable everywhere records default false and all explicit overrides disabled, advances installation authority and live generations; later explicit scoped enable admits fresh work only after reconciliation, preserving stale callback/approval refusal.
9. R17 exact adapter selected by actual Console cancellation regression: minimally extend Chat/console_chat_controller.py direct-provider run ownership to wake/cancel its captured consumer task on its owning loop when scoped revocation closes queue delivery. Keep actual provider_work completion custody and generation identity intact; consumer cancellation does not prove underlying work ended. Cancel only while that exact consumer is active, so a retained provider callback cannot cancel later unrelated work in a reused caller task. Extend the real Console regression in Tests/Plugins/test_revocation.py and exact affected provider lifecycle/stop neighbors; baseline static comparison against task BASE for the shared controller.
<!-- SECTION:PLAN:END -->
