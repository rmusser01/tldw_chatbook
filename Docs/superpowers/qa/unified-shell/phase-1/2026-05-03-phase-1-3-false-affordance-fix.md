# Phase 1.3 False Console Affordance Fix

Date: 2026-05-03
Task: `TASK-2.3`
Branch: `codex/unified-shell-phase1-false-affordances`
Base: `origin/dev` at `6a2ef7f8`

## Purpose

Close the Phase 1.2 audit finding that W+C, Schedules, Workflows, and ACP exposed Console follow/launch actions without actionable live-work payloads.

## Changes Verified

- W+C keeps the real `Open current Watchlists` route, but its Console follow control is disabled until watchlist/collection live-work payloads exist.
- Schedules now shows `Console recovery unavailable` with disabled recovery control and schedule-run payload recovery copy.
- Workflows now shows `Console launch unavailable` with disabled launch control and workflow-execution payload recovery copy.
- ACP keeps `Launch ACP Agent` disabled for runtime-unconfigured state and now disables generic Console follow until ACP session payloads exist.

## Running-App QA Evidence

Focused checks were run against Textual `App.run_test(...)` harnesses that mount the destination screens and exercise button state/click behavior.

- Red test first: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_skeletal_destination_console_actions_are_disabled_with_recovery_copy -q`
- Expected red result before screen changes: `4 failed`
- Focused green check: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py::test_skeletal_destination_console_actions_are_disabled_with_recovery_copy -q`
- Focused green result: `4 passed, 1 warning`
- Destination shell check: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py -q`
- Destination shell result: `46 passed, 1 warning`

Warning boundary: the remaining warning is the existing `requests` dependency warning and is unrelated to shell affordance behavior.

## Walkthrough Notes

- Visual usability: unavailable controls include `unavailable` in the visible button label and are paired with nearby recovery copy.
- Keyboard/mouse behavior: disabled Console controls do not trigger `open_console_for_live_work`.
- Functional result: the user can still reach real legacy surfaces where they exist, and cannot accidentally launch a generic Console placeholder for skeletal workflows.
- Residual risk: this does not implement live scheduler, workflow, ACP, or collection payload services. That remains Phase 4/5 work.

## Conclusion

`TASK-2.3` removes the known Phase 1 false Console-launch affordances without pretending the underlying services are complete.
