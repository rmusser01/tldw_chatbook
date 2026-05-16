# Phase 5.3 Server Events And Notifications Live Feed QA

Date: 2026-05-16
Status: verified
Backlog: `TASK-12.3`

## Scope

Phase 5.3 verifies that Home can present local notification queue state and server-owned event-feed state as distinct signals. This gate uses the existing `NotificationsScopeService.list_observed_server_feed()` projection; it does not add a new transport, WebSocket client, server mutation path, or local read-state authority for server resources.

Out of scope for this gate: server event polling, notification push delivery, marking server events as read, server-side notification mutation, and automatic recovery actions.

## Workflow Check

Verified by focused Home adapter and Notifications regressions:

1. Build the Home dashboard from local notification queue records and an observed server event feed.
2. Confirm unread local notifications are counted separately from observed server events.
3. Confirm Home status copy includes both `Local notifications: 1 unread` and `Server events: 1 observed via server event feed`.
4. Confirm replay retention gaps tell the user to requery server events instead of pretending the local cache is complete.
5. Confirm missing active-server scope tells the user to reconnect or select an active server.
6. Confirm unavailable server-event backend renders as unavailable, while local notification state remains local.

## State Coverage

| State | Expected Home status | Verification |
| --- | --- | --- |
| Local notification queue plus server event feed | Local and server signals render as separate status rows | Focused Home adapter regression |
| Server feed replay retention gap | `Server events: Replay gap - requery server events` | Focused Home adapter regression |
| Missing active server scope | `Server events: Reconnect required` | Focused Home adapter regression |
| Server event backend unavailable | `Server events: Unavailable` | Focused Home adapter regression |
| Existing event repository replay metadata | Retention-gap and replay metadata behavior remains covered | Notifications repository/service regressions |

## Source Authority Boundary

- Local notification count is derived from the local notification queue only.
- Server event count/state is derived from the observed server feed projection only.
- `mark_presented=False` is used when Home builds its dashboard so rendering Home does not mutate server-owned presentation/read state.
- Replay gaps are shown as recovery states requiring a server requery; they are not normalized into a complete local feed.

## Visual Evidence

- Actual rendered screenshot: `Docs/superpowers/qa/product-maturity/screen-qa/home/phase-5-3-home-server-events-2026-05-16.png`
- Screenshot capture method: `textual-web` on `127.0.0.1:8837` with `PYTHONPATH` pinned to this worktree and an isolated temporary HOME/XDG profile.
- Screenshot file verification: `PNG image data, 2050 x 1240, 8-bit/color RGB, non-interlaced`.
- Visual approval: approved by user on 2026-05-16 after actual rendered screenshot review; recaptured and reapproved after PR review fixes changed clean local-mode server-event copy from reconnect-required to unavailable.

## Verification Commands

Focused commands run for this gate:

```bash
python -m pytest -q Tests/Home/test_active_work_adapter.py -k 'server_events or server_event' --tb=short
```

Result: `4 passed, 22 deselected`.

```bash
python -m pytest -q Tests/Home/test_active_work_adapter.py Tests/Notifications/test_event_state_repository.py Tests/Notifications/test_notifications_scope_service.py --tb=short
```

Result: `50 passed`.

```bash
python -m pytest -q Tests/Home/test_active_work_adapter.py Tests/Home/test_dashboard_state.py Tests/UI/test_home_screen.py Tests/Notifications/test_event_state_repository.py Tests/Notifications/test_notifications_scope_service.py Tests/UI/test_product_maturity_phase5_server_parity_plan.py --tb=short
```

Result: `97 passed, 8 warnings`.

```bash
git diff --check
```

Result: pass.

## Functional Defects

No P0/P1 functional defects remain in the verified Phase 5.3 scope.

Accepted residual functional gaps:

- The event feed is an observed-state projection only; no live transport is introduced by this task.
- Server event read/presentation mutation remains server-owned and out of scope.

## UX Defects

No P0/P1 UX defects remain in the verified Phase 5.3 scope.

Accepted residual UX risks:

- Home presents feed readiness/recovery status, but does not yet provide a full event inbox drilldown.
- The recovery action is copy-only for reconnect/requery states until later server parity slices wire direct event-feed workflows.

## Visual/UI Defects

No P0/P1 visual/UI defects remain in the verified Phase 5.3 scope.

Visible UI change is bounded to Home status-copy rows in the existing `System Status` pane. Actual screenshot approval is complete for the clean local-mode Home capture.

## Result

Phase 5.3 passes for the implemented gate scope because Home distinguishes local notifications from server-owned observed events and exposes replay-gap, reconnect, and unavailable states without making local state authoritative for server resources.
