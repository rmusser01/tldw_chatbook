# Post-Release UX/HCI Walkthrough Evidence: ACP Runtime Launch

## Metadata

- Task: `TASK-60.4.1`
- Screen or workflow: ACP runtime launch and Console follow readiness
- Date: 2026-05-22
- Branch: `codex/task6041-acp-runtime-launch`
- App command: textual-web on `127.0.0.1:8891` with isolated HOME/XDG profile
- Evidence method: focused mounted Textual regressions plus textual-web/CDP screenshot capture
- Actual screenshot path: `Docs/superpowers/qa/product-maturity/post-release-ux-hci/acp-schedules-style-polish-final-2026-05-22.png`
- Screenshot approval: approved by user on 2026-05-22

## User Goal

Start an ACP-owned local runtime, see the resulting session as a first-class ACP work item, and follow the session into Console only when a real session payload exists.

## Steps Attempted

1. Configured an isolated QA ACP runtime command that runs a short local Python process.
2. Opened ACP through textual-web/CDP.
3. Activated the launch control.
4. Confirmed ACP moved to `Runtime running`, exposed process/session details, hid Launch/Restart while running, and kept Follow/Stop available.
5. Captured the approved running-state screenshot.

## What Worked

- ACP owns runtime launch configuration and does not move ACP setup into Settings.
- Missing runtime remains an honest blocked state.
- Configured runtime launch creates a session payload with process id, command, cwd, args, and start time.
- ACP, Console, and Home consume the same runtime readiness signal.
- The running-state ACP screen matches the approved destination style: three framed columns, visible selected session row, center detail pane, and right-side compatibility/actions inspector.

## Remaining Limits

- This tranche proves local process launch and Console-followable session payloads, not full ACP protocol execution.
- Diffs and terminal streams remain explicitly unavailable until a later ACP runtime-depth tranche adds those payloads.
- Server-backed ACP task/run package handoff remains future service-depth work.

## Verification

```bash
python -m pytest -q Tests/ACP/test_runtime_process.py Tests/UI/test_destination_shells.py::test_acp_missing_runtime_explains_acp_owned_setup_recovery Tests/UI/test_destination_shells.py::test_acp_configured_runtime_without_session_disables_console_follow Tests/UI/test_destination_shells.py::test_acp_configured_runtime_process_enables_launch_and_creates_session_payload Tests/UI/test_destination_shells.py::test_acp_failed_runtime_process_surfaces_recovery_and_restart_action Tests/UI/test_destination_shells.py::test_acp_runtime_and_session_labels_are_markup_escaped Tests/UI/test_destination_shells.py::test_acp_session_payload_enables_console_follow_live_work_handoff Tests/UI/test_destination_shells.py::test_acp_running_runtime_presents_actionable_hierarchy_without_dead_actions Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_source_readiness_reflects_acp_runtime_state Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_primary_action_routes_acp_session_details Tests/UI/test_home_screen.py::test_home_screen_acp_readiness_uses_runtime_process_state --tb=short
```

Result: 13 passed.

```bash
git diff --check
```

Result: passed.

## Acceptance Decision

- Accepted: yes.
- Reason: runtime launch, source-honest blocked states, cross-screen readiness consistency, and actual rendered ACP running-state screenshot were verified and approved.
- Follow-up: full ACP protocol execution, runtime terminal streaming, diffs, and server ACP package handoff remain outside this tranche.
