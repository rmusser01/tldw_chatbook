# Chatbook Workbench UI Foundation Console QA

## Responsiveness Baseline

- Command: `PATH=.venv/bin:$PATH pytest Tests/UI/test_ui_responsiveness.py Tests/UI/test_ui_responsiveness_artifacts.py -q`
- Evidence: included in the targeted Workbench verification suite below. The monitor records heartbeat lag, active timers, active workers, mount count, remove count, disabled-diagnostics behavior, heartbeat startup baseline reset, heartbeat shutdown, and footer timer shutdown.
- Command: `PATH=.venv/bin:$PATH python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10`
- Evidence: route-switch soak completed with `route switches: 6, failures: 0, focus failures: 0, workers before: 0, workers after: 0`.
- Responsiveness artifacts:
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/ui_heartbeat.log`: `max_heartbeat_lag_ms=54`, `stalled=False`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/worker_snapshot.log`: `active_workers=0`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/timer_registry.log`: `active_timers=3`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/mount_churn_summary.log`: `mounts=12`, `removes=4`
  - `Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts/route_switch_soak_result.txt`: route and focus failures are zero.

## Verification

- Workbench route inventory, responsiveness monitor, shared widgets, focus/help, visual snapshots, Console Workbench contract, parity matrix, and footer shortcut context:
  - Command: `PATH=.venv/bin:$PATH pytest Tests/UI/test_workbench_route_inventory.py Tests/UI/test_ui_responsiveness.py Tests/UI/test_ui_responsiveness_artifacts.py Tests/UI/test_workbench_state.py Tests/UI/test_workbench_widgets.py Tests/UI/test_workbench_focus_help.py Tests/UI/test_workbench_visual_snapshots.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_workbench_parity_matrix.py Tests/UI/test_app_footer_shortcut_context.py -q`
  - Result: `58 passed, 1 warning` in 16.88s. Warning: existing `requests` dependency warning.
- Console regression and parity set:
  - Command: `PATH=.venv/bin:$PATH pytest Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_chat_image_attachment.py Tests/UI/test_command_palette_basic.py Tests/UI/test_command_palette_providers.py Tests/Chat/test_console_message_actions.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_display_state.py Tests/Chat/test_console_provider_support.py Tests/Chat/test_console_provider_endpoints.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py::TestChatFunction::test_chat_with_image_and_rag Tests/integration/test_chat_tool_flow.py -q`
  - Result: `673 passed, 8 warnings` in 541.79s. Warnings: existing `requests` dependency warning plus SWIG deprecation warnings.
  - Note: the first run exposed a stale legacy-selector expectation for missing-model recovery; the test now targets the visible Workbench recovery action, and the full suite was rerun successfully.
- Route/navigation smoke:
  - Command: `PATH=.venv/bin:$PATH pytest Tests/UI/test_shell_destinations.py Tests/UI/test_screen_navigation.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_command_palette_shell_routes.py -q`
  - Result: `54 passed, 1 warning` in 10.13s. Warning: existing `requests` dependency warning.
- CSS build:
  - Command: `PATH=.venv/bin:$PATH python3 tldw_chatbook/css/build_css.py`
  - Result: exit 0. Known warning: `features/_evaluation_v2.tcss` is missing; generated `tldw_chatbook/css/tldw_cli_modular.tcss`.
- Diff hygiene:
  - Command: `git diff --check`
  - Result: exit 0.
- Route-switch soak:
  - Command: `PATH=.venv/bin:$PATH python3 Tests/UI/run_workbench_soak.py --output Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts --route-switches 6 --idle-seconds 10`
  - Result: `route switches: 6, failures: 0, focus failures: 0, workers before: 0, workers after: 0`.
- Artifact presence:
  - Command: `test -s` for `ui_heartbeat.log`, `worker_snapshot.log`, `timer_registry.log`, `mount_churn_summary.log`, and `route_switch_soak_result.txt`
  - Result: exit 0.

## Residual Risks

- The automated soak uses synthetic Console interactions. A real provider streaming soak can be added when a provider endpoint is available.
- Later destinations are tracked by route owner but not migrated in this task.
- Full project `pytest -q` was not run; final verification used the scoped Workbench, Console, navigation, CSS, diff, and soak gates from the implementation plan.
