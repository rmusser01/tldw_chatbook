# Product Maturity Phase 1.2 Clean First-Run Walkthrough

## Environment

- Date: 2026-05-05
- Branch: codex/product-maturity-phase1-2-exec
- Commit: Phase 1.2 execution worktree before final PR commit
- Python version: Python 3.12.11
- Runtime source: running Textual app through the app test pilot
- Config/home directory: fresh pytest temporary directories

## Task Or Phase

- Backlog task: TASK-8.2
- Phase: Phase 1.2
- Destination or workflow: Clean first-run launch and configuration orientation

## Entry Path

- Launch command, direct route, command palette path, focused mounted test, or manual terminal replay: clean first-run app test with `_first_run=True`, configured default route set to Console, and splash disabled

## Terminal Size

- Category: minimum supported compact
- Dimensions: 100x32
- Category: common laptop terminal
- Dimensions: 140x40
- Category: large power-user workspace
- Dimensions: 180x50

## Clean-Run Setup

- Fresh HOME: `<tmp>/home`
- XDG_CONFIG_HOME: `<tmp>/xdg-config`
- XDG_DATA_HOME: `<tmp>/xdg-data`
- XDG_CACHE_HOME: `<tmp>/xdg-cache`
- Reused state: none

## Steps Attempted

1. Launched the Textual app from fresh HOME and XDG directories.
2. Forced first-run routing while the configured default route pointed at Console.
3. Verified the app opened Home first instead of dropping into Console.
4. Verified Home exposed dashboard purpose, Console setup guidance, and command-palette overflow copy.
5. Activated Console, Library, and Settings navigation buttons from the first-run context.
6. Ran shallow compact and large terminal probes to verify Home loaded without exceptions and retained first-run setup affordances.

## Visual/Focus Notes

- Layout: Home rendered the dashboard, navigation bar, and primary setup action at 100x32, 140x40, and 180x50.
- Clipping: no clipping was detected by mounted Textual assertions; full manual screenshot review remains a later Phase 1 visual gate.
- Focus indication: button activation was verified through Textual pilot controls; full keyboard focus traversal remains a later Phase 1 gate.
- Labels: Home, Console, Library, and Settings labels were visible and matched the shell product model.
- Disabled or blocked states: first-run Home correctly showed `Set up Console model` instead of `Start in Console`, avoiding a false affordance when model readiness is blocked.
- Information hierarchy: Home made setup/status orientation visible before deeper product workflows.

## Keyboard Path Result

- Completed, blocked with recovery, failed, or not tested: partially completed by mounted button/navigation assertions; full keyboard traversal is not tested in this gate.

## Mouse/Click Path Result

- Completed, blocked with recovery, failed, or not tested: completed through Textual pilot button activation for Home to Console, Library, and Settings.

## Functional Result

- Completed workflow: clean first-run launch opens Home and exposes setup-oriented routes to Console, Library, and Settings.
- Honest blocked state: direct `Start in Console` is withheld until model readiness exists; Home routes the user to Console setup instead.
- Failed workflow: none found for this Phase 1.2 scope.
- Recovery path: configure the Console model/provider through the setup path, then later replay the core Console workflow in Phase 2.

## Defects Found

- `blocker` / P0: none.
- `workflow-degradation` / P1: none.
- `recoverability` / P2: full keyboard traversal and screenshot-based visual review remain untested in this gate.
- `polish` / P3: none.

## Evidence

- Screenshots: not captured; mounted app assertions were sufficient for this setup-orientation gate.
- Logs: pytest captured standard Textual startup logs.
- Probe JSON: not applicable.
- Test commands: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q` -> 5 passed
- Harness regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q` -> 6 passed
- Adjacent first-time replay: `../../.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q` -> 2 passed
- Related PRs or commits: Phase 1.2 execution branch

## Residual Risk

- Untested live server/API paths: all live model/provider calls remain outside this Phase 1.2 gate.
- Optional dependency limits: optional dependency setup screens were not exhaustively replayed.
- Environment limits: manual screenshot and full keyboard/focus sweeps remain later Phase 1 gates.
- Follow-up tasks: top-level navigation smoke, keyboard/focus sweep, visual broken-state audit, empty/error/setup-state coverage, and narrow core-loop proof.

## Exit Decision

- Pass for Phase 1.2 scope.

## Product QA Boundary

This walkthrough verifies clean first-run launch and setup orientation only. It proves the app is usable, not merely rendered, for entering Home and finding the Console, Library, and Settings setup paths. It does not complete the full Phase 1 top-level navigation, keyboard/focus, visual, empty/error/setup, or narrow core-loop gates, and it does not start the Phase 2 grounded Console to Artifact/Chatbook loop.
