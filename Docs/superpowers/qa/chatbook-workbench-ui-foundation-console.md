# Chatbook Workbench UI Foundation Console QA

## Responsiveness Baseline

- Command: `PATH=.venv/bin:$PATH pytest Tests/UI/test_ui_responsiveness.py Tests/UI/test_ui_responsiveness_artifacts.py -q`
- Evidence: `12 passed`; monitor records heartbeat lag, active timers, active workers, mount count, remove count, disabled-diagnostics behavior, heartbeat shutdown, and footer timer shutdown. The run emitted one existing `requests` dependency warning.
- Command: `PATH=.venv/bin:$PATH python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 1`
- Evidence: route switch soak completed with `route switches: 6, failures: 0, focus failures: 0, workers before: 0, workers after: 0`.
- Responsiveness artifacts:
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/ui_heartbeat.log`: `max_heartbeat_lag_ms=163`, `stalled=False`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/worker_snapshot.log`: `active_workers=0`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/timer_registry.log`: `active_timers=3`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/mount_churn_summary.log`: `mounts=12`, `removes=4`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/route_switch_soak_result.txt`: route and focus failures are zero.
