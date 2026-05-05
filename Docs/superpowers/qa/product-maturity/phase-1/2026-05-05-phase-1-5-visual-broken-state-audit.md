# Product Maturity Phase 1.5 Visual Broken-State Audit

## Environment

- Date: 2026-05-05
- Branch: codex/product-maturity-phase1-5-visual-audit
- Commit: Phase 1.5 execution worktree before final PR commit
- Python version: Python 3.12.11
- Runtime source: running Textual app through the app test pilot
- Config/home directory: fresh pytest temporary directories

## Task Or Phase

- Backlog task: TASK-8.5
- Phase: Phase 1.5
- Destination or workflow: Visual Broken-State Audit

## Entry Path

- Launch command, direct route, command palette path, focused mounted test, or manual terminal replay: clean first-run app test with `_first_run=True`, configured default route set to Console, splash disabled, then screen-navigation routing to every top-level destination.

## Terminal Size

- compact: 100x32
- laptop: 140x40
- large: 180x50

## Clean-Run Setup

- Fresh HOME: `<tmp>/home`
- XDG_CONFIG_HOME: `<tmp>/xdg-config`
- XDG_DATA_HOME: `<tmp>/xdg-data`
- XDG_CACHE_HOME: `<tmp>/xdg-cache`
- Reused state: none

## Steps Attempted

1. Launched the Textual app from fresh HOME and XDG directories.
2. Verified first-run routing opened Home before the configured Console default.
3. Routed through every top-level shell destination at compact, laptop, and large terminal sizes.
4. Waited for stable shell chrome after async destination refreshes before recording visual state.
5. Exported an in-memory SVG screenshot for each destination and size.
6. Checked each rendered destination for non-empty body content, intact navigation chrome, `Ctrl+P` fallback hint, active destination state, traceback/unhandled-exception text, and raw object repr leakage.

## Visual Size Matrix

| Size | Dimensions | Destination count | SVG screenshot export | Result |
| --- | --- | ---: | --- | --- |
| compact | 100x32 | 12 | in-memory per destination | pass |
| laptop | 140x40 | 12 | in-memory per destination | pass |
| large | 180x50 | 12 | in-memory per destination | pass |

## Destination Probe

| Destination | Visual result |
| --- | --- |
| `home` | non-empty content, shared chrome, fallback hint, active state |
| `console` | non-empty content, shared chrome, fallback hint, active state |
| `library` | non-empty content, shared chrome, fallback hint, active state |
| `artifacts` | non-empty content, shared chrome, fallback hint, active state |
| `personas` | non-empty content, shared chrome, fallback hint, active state |
| `watchlists_collections` | non-empty content, shared chrome, fallback hint, active state |
| `schedules` | non-empty content, shared chrome, fallback hint, active state |
| `workflows` | non-empty content, shared chrome, fallback hint, active state |
| `mcp` | non-empty content, shared chrome, fallback hint, active state |
| `acp` | non-empty content, shared chrome, fallback hint, active state |
| `skills` | non-empty content, shared chrome, fallback hint, active state |
| `settings` | non-empty content, shared chrome, fallback hint, active state |

## Visual/Focus Notes

- Layout: top-level destinations mount body content inside `#screen-content` with shared navigation chrome.
- Clipping: no blank, traceback, or unmounted-content state was detected by SVG export and DOM assertions across compact, laptop, and large sizes.
- Async refresh: Schedules and Workflows can refresh after mount; the audit waits for full navigation chrome before snapshot export.
- Labels: the `Ctrl+P` overflow/fallback hint remains present across the size matrix.
- Disabled or blocked states: blocked Console follow/launch actions remain visible as disabled controls with explanatory copy.

## Keyboard Path Result

- Completed, blocked with recovery, failed, or not tested: not retested here; Phase 1.4 already verifies keyboard reachability. This gate confirms the same destinations remain visually inspectable after routing.

## Mouse/Click Path Result

- Completed, blocked with recovery, failed, or not tested: not the target of this gate; Phase 1.3 already covers top-level button activation at the large layout, and Phase 1.4 covers keyboard fallback for overflow.

## Functional Result

- Completed workflow: all top-level destination shells can be visually rendered and exported across the supported size matrix.
- Honest blocked state: optional backend/data-dependent states remain visibly blocked with recovery copy where available.
- Failed workflow: none found for this Phase 1.5 scope.
- Recovery path: use `Ctrl+P` when compact layouts make direct top navigation inefficient.

## Defects Found

- `blocker` / P0: none.
- `workflow-degradation` / P1: none. No P0/P1 visual broken-state blockers were found.
- `recoverability` / P2: live optional service/database setup states still need a dedicated empty/error/setup-state pass.
- `polish` / P3: no screenshot files were retained because in-memory SVG screenshot export plus structural assertions were sufficient for this gate.

## Evidence

- Screenshots: in-memory SVG screenshot export for each top-level destination at compact, laptop, and large sizes; no file artifacts retained because no visual defect required illustration.
- Logs: pytest captured standard Textual startup logs.
- Probe JSON: not applicable.
- Test command: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_visual_audit.py -q` -> 5 passed.
- Phase 1 regression: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_harness.py Tests/UI/test_product_maturity_phase1_first_run.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py Tests/UI/test_product_maturity_phase1_keyboard_focus.py Tests/UI/test_product_maturity_phase1_visual_audit.py -q` -> 31 passed.
- Related shell regression: `../../.venv/bin/python -m pytest Tests/UI/test_command_palette_basic.py Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_app_footer_shortcut_context.py -q` -> 14 passed.
- Related PRs or commits: Phase 1.5 execution branch.

## Residual Risk

- Untested live server/API paths: no live backend calls are part of this gate.
- Optional dependency limits: optional dependency setup screens were not exhaustively replayed.
- Environment limits: screenshot files were not saved because no specific visual defect needed annotation.
- Follow-up tasks: Remaining Phase 1 gates are empty/error/setup-state coverage and narrow core-loop proof.

## Exit Decision

- Pass for Phase 1.5 scope.

## Product QA Boundary

This walkthrough verifies that the top-level shell destinations are visually inspectable and do not collapse into blank, traceback, or unexplained broken states across compact, laptop, and large terminal sizes. It proves visual/chrome integrity, not detailed workflows inside every destination or the Phase 2 grounded Console to Artifact/Chatbook loop.
