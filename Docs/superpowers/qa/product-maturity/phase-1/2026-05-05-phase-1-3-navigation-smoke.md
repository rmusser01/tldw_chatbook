# Product Maturity Phase 1.3 Top-Level Navigation Smoke

## Environment

- Date: 2026-05-05
- Branch: codex/product-maturity-phase1-3-nav-smoke
- Commit: Phase 1.3 execution worktree before final PR commit
- Python version: Python 3.12.11
- Runtime source: running Textual app through the app test pilot
- Config/home directory: fresh pytest temporary directories

## Task Or Phase

- Backlog task: TASK-8.3
- Phase: Phase 1.3
- Destination or workflow: Top-Level Navigation Smoke

## Entry Path

- Launch command, direct route, command palette path, focused mounted test, or manual terminal replay: clean first-run app test with `_first_run=True`, configured default route set to Console, and splash disabled

## Terminal Size

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
2. Verified first-run routing opened Home before the configured Console default.
3. Enumerated the mounted master-shell navigation bar and compared it to the canonical destination order.
4. Activated each top-level navigation button through the Textual pilot.
5. Waited for each target screen and current-tab value to match the router's expected destination.
6. Verified each reached screen exposed non-empty orientation or control text.
7. Confirmed the navigation overflow hint exposes `Ctrl+P` as the keyboard fallback for destinations that are not comfortably visible in narrower terminals.

## Destination Smoke Matrix

| Destination | Route | Result |
| --- | --- | --- |
| `home` | `home` | reached Home screen with dashboard/setup orientation |
| `console` | `chat` | reached Console screen with live-work source/status orientation |
| `library` | `library` | reached Library screen with source/import/search orientation |
| `artifacts` | `artifacts` | reached Artifacts screen with generated-output orientation |
| `personas` | `personas` | reached Personas screen with loading/empty/service state surface |
| `watchlists_collections` | `watchlists_collections` | reached W+C screen with watchlist/collection state surface |
| `schedules` | `schedules` | reached Schedules screen with scheduler empty state |
| `workflows` | `workflows` | reached Workflows screen with procedure/run orientation |
| `mcp` | `mcp` | reached MCP screen with tool/server control orientation |
| `acp` | `acp` | reached ACP screen with agent/session/runtime orientation |
| `skills` | `skills` | reached Skills screen with discovery/validation state surface |
| `settings` | `settings` | reached Settings screen with preferences orientation |

## Visual/Focus Notes

- Layout: all top-level navigation buttons were mounted in canonical order at 180x50.
- Clipping: no route-switching exceptions or empty rendered screens were detected by the mounted smoke test; screenshot-based clipping review remains a later Phase 1 visual gate.
- Focus indication: pilot activation verified every top-level route can be reached by button activation; full tab-order traversal remains a later Phase 1 focus gate.
- Labels: the product model remains visible through Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, and Settings.
- Disabled or blocked states: no top-level destination presented a dead or false navigation affordance during this smoke pass.
- Information hierarchy: this gate verifies destination reachability and first-level orientation only, not full workflow completion inside each destination.

## Keyboard Path Result

- Completed, blocked with recovery, failed, or not tested: partially completed through the persistent `Ctrl+P` overflow hint. Full command-palette execution and tab-order traversal remain later Phase 1 gates.

## Mouse/Click Path Result

- Completed, blocked with recovery, failed, or not tested: completed through Textual pilot button activation for every top-level destination.

## Functional Result

- Completed workflow: clean-run Home can reach every top-level destination exposed by the master shell.
- Honest blocked state: destinations with no local data expose orientation/loading/empty states instead of silent dead ends.
- Failed workflow: none found for this Phase 1.3 scope.
- Recovery path: use `Ctrl+P` when the navigation bar overflows or the target destination is not visible in a compact terminal.

## Defects Found

- `blocker` / P0: none.
- `workflow-degradation` / P1: none. No P0/P1 navigation blockers were found.
- `recoverability` / P2: command-palette execution and full keyboard traversal remain unverified in this gate.
- `polish` / P3: screenshot-based visual polish and clipping review remain unverified in this gate.

## Evidence

- Screenshots: not captured; mounted app assertions were sufficient for this top-level navigation smoke gate.
- Logs: pytest captured standard Textual startup logs.
- Probe JSON: not applicable.
- Test commands: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_navigation_smoke.py -q` -> 4 passed
- Harness regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py -q` -> 6 passed
- First-run regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_first_run.py -q` -> 12 passed
- Related PRs or commits: Phase 1.3 execution branch

## Residual Risk

- Untested live server/API paths: destination service calls and live backend actions remain outside this Phase 1.3 gate.
- Optional dependency limits: optional dependency setup screens were not exhaustively replayed.
- Environment limits: manual screenshot, compact-terminal visual audit, full keyboard/focus traversal, empty/error/setup-state coverage, and narrow core-loop proof remain open.
- Follow-up tasks: Remaining Phase 1 gates are keyboard/focus sweep, visual broken-state audit, empty/error/setup-state coverage, and narrow core-loop proof.

## Exit Decision

- Pass for Phase 1.3 scope.

## Product QA Boundary

This walkthrough verifies top-level destination reachability and first-level orientation only. It proves the master shell is usable, not merely rendered, for moving across the product model. It does not complete detailed workflows inside each destination, the full keyboard/focus sweep, screenshot-based visual audit, empty/error/setup-state coverage, or the Phase 2 grounded Console to Artifact/Chatbook loop.
