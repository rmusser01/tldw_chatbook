# Product Maturity Phase 1.4 Keyboard And Focus Sweep

## Environment

- Date: 2026-05-05
- Branch: codex/product-maturity-phase1-4-keyboard-focus
- Commit: Phase 1.4 execution worktree before final PR commit
- Python version: Python 3.12.11
- Runtime source: running Textual app through the app test pilot
- Config/home directory: fresh pytest temporary directories

## Task Or Phase

- Backlog task: TASK-8.4
- Phase: Phase 1.4
- Destination or workflow: Keyboard And Focus Sweep

## Entry Path

- Launch command, direct route, command palette path, focused mounted test, or manual terminal replay: clean first-run app test with `_first_run=True`, configured default route set to Console, and splash disabled

## Terminal Size

- Category: common laptop terminal
- Dimensions: 140x40

## Clean-Run Setup

- Fresh HOME: `<tmp>/home`
- XDG_CONFIG_HOME: `<tmp>/xdg-config`
- XDG_DATA_HOME: `<tmp>/xdg-data`
- XDG_CACHE_HOME: `<tmp>/xdg-cache`
- Reused state: none

## Steps Attempted

1. Launched the Textual app from fresh HOME and XDG directories.
2. Verified first-run routing opened Home before the configured Console default.
3. Walked focus with the Tab key from the initial Home focus state through every top-level navigation button.
4. Verified the first-run primary setup action is reachable by Tab after top-level navigation.
5. Verified the global `Ctrl+P` binding is registered to the command-palette action.
6. Verified the command-palette navigation provider exposes every top-level shell destination in canonical order with descriptive help text.

## Tab Order Probe

| Order | Focus target |
| --- | --- |
| 1 | `nav-home` |
| 2 | `nav-console` |
| 3 | `nav-library` |
| 4 | `nav-artifacts` |
| 5 | `nav-personas` |
| 6 | `nav-watchlists_collections` |
| 7 | `nav-schedules` |
| 8 | `nav-workflows` |
| 9 | `nav-mcp` |
| 10 | `nav-acp` |
| 11 | `nav-skills` |
| 12 | `nav-settings` |
| 13 | `home-primary-action` |

## Command-Palette Fallback Probe

| Destination | Keyboard fallback result |
| --- | --- |
| Home | `Ctrl+P` exposes Home with dashboard/status purpose text |
| Console | `Ctrl+P` exposes Console with live agent work purpose text |
| Library | `Ctrl+P` exposes Library with source/import/search purpose text |
| Artifacts | `Ctrl+P` exposes Artifacts with generated output purpose text |
| Personas | `Ctrl+P` exposes Personas with behavior-profile purpose text |
| W+C | `Ctrl+P` exposes Watchlists+Collections with monitored-source purpose text |
| Schedules | `Ctrl+P` exposes Schedules with timing/trigger purpose text |
| Workflows | `Ctrl+P` exposes Workflows with reusable-procedure purpose text |
| MCP | `Ctrl+P` exposes MCP with server/tool capability purpose text |
| ACP | `Ctrl+P` exposes ACP with agent/session/runtime purpose text |
| Skills | `Ctrl+P` exposes Skills with discovery/validation purpose text |
| Settings | `Ctrl+P` exposes Settings with preferences/storage purpose text |

## Visual/Focus Notes

- Layout: Home loads with navigation first and setup orientation visible.
- Clipping: no clipping was detected by mounted Textual assertions; screenshot-based clipping review remains a later Phase 1 visual gate.
- Focus indication: Tab order reaches all top-level navigation buttons and then the first-run setup action.
- Labels: top-level destination names remain discoverable through visible navigation and command-palette labels/help.
- Disabled or blocked states: `home-primary-action` remains reachable and labelled `Set up Console model`; no mouse-only setup blocker was found.
- Information hierarchy: keyboard users can move across the shell first, then reach the primary first-run setup action.

## Keyboard Path Result

- Completed, blocked with recovery, failed, or not tested: completed for Tab order through top-level navigation and primary setup action; completed for `Ctrl+P` registration/provider coverage.

## Mouse/Click Path Result

- Completed, blocked with recovery, failed, or not tested: not the target of this gate; Phase 1.3 already covered top-level button activation.

## Functional Result

- Completed workflow: a keyboard user can reach all top-level navigation buttons and the first-run setup action from clean-run Home.
- Honest blocked state: direct Console work remains model-setup gated, but the setup action is keyboard reachable.
- Failed workflow: none found for this Phase 1.4 scope.
- Recovery path: use `Ctrl+P` when a top-level destination is not visible or Tab traversal is inefficient.

## Defects Found

- `blocker` / P0: none.
- `workflow-degradation` / P1: none. No P0/P1 keyboard or focus blockers were found.
- `recoverability` / P2: full live command-palette overlay interaction is not replayed in this gate; provider/binding coverage verifies the fallback contract.
- `polish` / P3: screenshot-based visual focus styling remains unverified in this gate.

## Evidence

- Screenshots: not captured; mounted app focus assertions were sufficient for this keyboard/focus gate.
- Logs: pytest captured standard Textual startup logs.
- Probe JSON: not applicable.
- Test commands: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_keyboard_focus.py -q` -> 4 passed
- Harness regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q` -> 6 passed
- First-run regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q` -> 12 passed
- Navigation smoke regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_navigation_smoke.py -q` -> 4 passed
- Related PRs or commits: Phase 1.4 execution branch

## Residual Risk

- Untested live server/API paths: no live backend calls are part of this gate.
- Optional dependency limits: optional dependency setup screens were not exhaustively replayed.
- Environment limits: manual screenshot focus-ring review, empty/error/setup-state coverage, and narrow core-loop proof remain open.
- Follow-up tasks: Remaining Phase 1 gates are visual broken-state audit, empty/error/setup-state coverage, and narrow core-loop proof.

## Exit Decision

- Pass for Phase 1.4 scope.

## Product QA Boundary

This walkthrough verifies keyboard reachability and focus/fallback affordances for clean-run Home and the top-level product model. It proves primary shell navigation and first-run setup are keyboard usable, not merely rendered. It does not complete screenshot-based visual focus styling, empty/error/setup-state coverage, detailed workflows inside each destination, or the Phase 2 grounded Console to Artifact/Chatbook loop.
