# Phase 2 Home Operational-Control Closeout

Date: 2026-05-05
Task: `TASK-4.8`
Parent Task: `TASK-4`
Branch: `codex/unified-shell-phase2-home-closeout`
Base: `origin/dev` at `32ac2bdc`

## Current Baseline

Phase 2 is replayed after the merged Home operational-control slices:

- `TASK-4.1` - Home active-work adapter boundary and unavailable default controls.
- `TASK-4.2` - adapter-owned `Open details` and `Open in Console` actions.
- `TASK-4.3` - item-scoped Home control target identity.
- `TASK-4.4` - local unread notification snapshot.
- `TASK-4.5` - notification review routing into the notifications inbox.
- `TASK-4.6` - local W+C watchlist runs surfaced as Home active work.
- `TASK-4.7` - W+C watchlist run details opened from Home.
- `TASK-4.8` - maturity-gate replay and closeout.

The implemented product boundary is explicit-adapter control, not full schedule or agent-service execution. Unsupported approve, reject, pause, resume, and retry controls are valid only when they return honest unavailable messages with a recovery route. W+C watchlist run detail and Console paths are handled when the local watchlist adapter can resolve the target run.

## Workflow Matrix

| Workflow | Running-app path | Result | Severity |
| --- | --- | --- | --- |
| Approve active Home item | Mounted Home screen renders `#home-approve`; click calls `approve_active_home_item`; app hook delegates to `HomeControlAction.APPROVE`. | Functional through adapter boundary; unavailable default is explicitly recoverable when no active-run service is wired. | none |
| Reject active Home item | Mounted Home screen renders `#home-reject`; click calls `reject_active_home_item`; app hook delegates to `HomeControlAction.REJECT`. | Functional through adapter boundary; unavailable default is explicitly recoverable when no active-run service is wired. | none |
| Pause active Home item | Mounted Home screen renders `#home-pause`; item-scoped click passes `target_id` to the app hook and adapter. | Functional through adapter boundary; unavailable default is explicitly recoverable when no active-run service is wired. | none |
| Resume active Home item | Mounted Home screen renders `#home-resume`; click calls `resume_active_home_item`; app hook delegates to `HomeControlAction.RESUME`. | Functional through adapter boundary; unavailable default is explicitly recoverable when no active-run service is wired. | none |
| Retry failed Home item | Failed W+C active-work row selects `#home-retry` with `target_id=local:watchlist_run:*`; click passes the target to the app hook and adapter. | Functional through adapter boundary; current local W+C retry remains recoverable via W+C detail/Console rather than fake retry execution. | none |
| Open details | `#home-open-details` delegates to the adapter; handled W+C targets stage `pending_subscription_initial_tab=watchlist-runs` and navigate to `subscriptions`. | Functional for local W+C run details; unavailable default does not navigate falsely and shows recovery copy. | none |
| Open in Console | `#home-open-in-console` delegates to the adapter; handled W+C targets create `HomeConsoleLaunch` and call `open_console_for_live_work`. | Functional for local W+C Console launch payloads; unavailable default does not create fake Console work. | none |
| Notification review | Home primary action for unread notifications stages `pending_subscription_initial_tab=notifications` and opens the notifications inbox. | Functional local notification recovery path. | none |

## Focused Verification

Commands run from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-unified-shell-phase2-home-closeout`:

- Baseline replay: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Baseline replay result before closeout tracker updates: `57 passed, 2 failed, 10 warnings`
- Baseline failures were tracker drift only:
  - `Tests/UI/test_unified_shell_phase2_home_adapter.py::test_phase_two_home_adapter_is_linked_from_index_roadmap_and_task` still expected Phase 2 `in-progress`.
  - `Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_two_three_four_closeout_tasks_keep_parent_phases_open` still expected `TASK-4.8` to remain `To Do`.
- Red closeout contract: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase234_maturity_gate.py::test_phase_two_closeout_doc_records_verified_workflows_and_task_completion -q`
- Red closeout result before evidence existed: `1 failed` with `FileNotFoundError` for this closeout document.
- Final focused replay: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Home/test_dashboard_state.py Tests/Home/test_active_work_adapter.py Tests/UI/test_home_screen.py Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services Tests/UI/test_unified_shell_phase2_home_adapter.py Tests/UI/test_unified_shell_phase234_maturity_gate.py -q`
- Final focused replay result after closeout updates: `60 passed, 8 warnings`

Warning boundary: final warnings are existing dependency/import warnings. Baseline replay also emitted train splash string warnings. None are Home operational-control behavior failures.

## UX Notes

- Home remains a usable status and control dashboard without pretending unavailable backend actions are wired.
- Visible controls are not render-only: mounted Textual tests click controls and verify app hooks, adapter calls, target IDs, target routes, notifications, navigation staging, and Console launch payloads.
- Beginner recovery copy is explicit when no live run service owns an action.
- Power-user density and speed are preserved: the Home controls remain directly clickable and item-scoped controls carry target identity instead of forcing users through a generic detail page.

## Defects And Blockers

No Phase 2 closeout blocker remains.

The only issues found during replay were stale tracker tests created by the transition from implementation slices to maturity-gate QA. Those are resolved by this closeout test update and evidence document.

## Residual Risk

- Schedule and agent-service adapters still need future implementation; Phase 2 verifies the Home adapter/control contract and local W+C/notification flows, not every future backend.
- Local W+C retry remains recoverable via W+C detail or Console context rather than a direct retry service action.
- Full first-time-user and power-user shell replay remains Phase 6 scope.

## Closeout Decision: verified

Phase 2 is verified because Home approve, reject, pause, resume, retry, open-detail, notification review, and Console handoff paths are covered by mounted running-app tests and adapter-level tests. Unsupported backend actions are explicitly recoverable, and handled local W+C and notification workflows route to concrete app state instead of only rendering or firing empty click handlers.
