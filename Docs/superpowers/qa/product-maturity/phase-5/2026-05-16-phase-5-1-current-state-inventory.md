# Phase 5.1 Server Parity Current-State Inventory

Date: 2026-05-16
Branch: `codex/phase5-server-parity-planning`
Backlog task: TASK-12.1

## Scope

Reconcile Product Maturity Phase 5 with the current `dev` branch before implementing live integrations. The older April server-parity plans are useful design references, but `dev` already contains much of that foundation and should not be reimplemented.

## Current Dev Inventory

| Area | Current anchor | State | Product-maturity consequence |
| --- | --- | --- | --- |
| Active server/auth | `RuntimeServerContextProvider`, `server_connection_contracts.py`, `server-client-provider-migration-audit.md` | Foundation exists and is tracked as done or safe in `backend-parity-phase-tracker.md`. | Phase 5 should verify running-app server status and recovery, not build a second server selector or credential path. |
| Event state | `EventStateRepository`, `server_notification_events.py`, `event_observer.py` | Foundation exists with durable cursor/dedupe/presentation contracts. | Phase 5 should surface event/feed states and replay gaps in user workflows. |
| Sync dry-run | `Sync_Interop`, `SyncReadinessReport`, mirror-report helpers | Read-only dry-run foundation exists. | Phase 5 may expose readiness and reports, but write sync remains deferred. |
| Domain parity contracts | `domain_edge_contracts.py`, domain scope services, backend parity tracker | Primary domains and remote utilities are mostly ready-for-UX with unsupported reports. | Phase 5 should pick high-value workflow gaps, not chase endpoint count. |
| UX contracts | `server_parity_contracts.py`, backend handoff packet | Contract fixtures exist. | Phase 5 screens should consume these contracts rather than inferring capability from current UI state. |

## Residual Risks From Earlier Phases

- ACP runtime launch remains a Phase 4 accepted residual.
- Schedules and Workflows run-control services remain accepted residuals unless backend contracts are already available.
- write sync remains deferred; Phase 5 may expose dry-run mirror state only.
- Live server/auth behavior needs running-app QA because backend unit coverage does not prove the product workflow is usable.
- Any visible UI changes still require actual screenshot evidence and user approval.

## Phase 5 Slice Recommendation

1. `TASK-12.1`: current-state inventory and child-task split.
2. `TASK-12.2`: active-server/auth live status and recovery in the running app.
3. `TASK-12.3`: server event and notification live-feed surfacing.
4. `TASK-12.4`: sync mirror dry-run workflow surfacing without write sync.
5. `TASK-12.5`: high-value domain parity workflow closures selected by product impact.
6. `TASK-12.6`: live integration closeout replay.

## Verification Evidence

Focused planning regression:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase5_server_parity_plan.py --tb=short
```

Result: `1 passed in 0.16s`.

Backend parity contract smoke:

```bash
python -m pytest -q Tests/RuntimePolicy/test_server_context_provider.py Tests/Notifications/test_event_state_repository.py Tests/Sync_Interop/test_sync_readiness.py Tests/UX_Interop/test_server_parity_contracts.py --tb=short
```

Result: `75 passed in 7.51s`.

Diff hygiene:

```bash
git diff --check
```

Result: passed with no output.

## Closeout Decision

Phase 5 is now activated but not verified. The next implementation slice should start with active-server/auth live status because it is the safest dependency for later event, sync, and domain workflow work.
