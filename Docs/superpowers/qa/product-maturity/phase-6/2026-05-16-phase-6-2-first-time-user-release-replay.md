# Phase 6.2 First-Time User Release Replay

<!-- PHASE_6_2_FIRST_TIME_RELEASE_REPLAY_METADATA:BEGIN -->
```json
{
  "task": "TASK-13.2",
  "parent_task": "TASK-13",
  "persona": "first-time-user",
  "decision": "first_time_release_replay_recorded",
  "verified_routes": ["home", "console", "library", "settings"],
  "clean_environment": ["HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"],
  "p0_p1_findings": [],
  "final_focused_replay_result": {
    "passed": 2,
    "failed": 0
  }
}
```
<!-- PHASE_6_2_FIRST_TIME_RELEASE_REPLAY_METADATA:END -->

## Environment

- Date: 2026-05-16
- Scope: Product Maturity Phase 6.2 release-hardening replay
- App under test: running Textual app through the mounted `TldwCli` harness
- Terminal size: `140x42`
- Persona: first-time user with no prior product knowledge

## Clean-Run Setup

The replay used a clean environment with isolated placeholders for:

- `HOME`: `<tmp>/home`
- `XDG_CONFIG_HOME`: `<tmp>/xdg-config`
- `XDG_DATA_HOME`: `<tmp>/xdg-data`
- `XDG_CACHE_HOME`: `<tmp>/xdg-cache`

The splash screen was disabled through the test settings seam so the replay starts directly at the shell.

## Entry Path

The app was configured as first-run with the legacy initial tab set to `chat`. The shell correctly routed the user to `Home`, confirming the release front door is the dashboard/status surface rather than a blank Console.

## Steps Attempted

1. Launch the running Textual app in the clean environment.
2. Verify Home appears and exposes first-run orientation.
3. Verify the top navigation exposes Home, Console, Library, and Settings.
4. Open Console and verify live-work/source orientation plus provider setup recovery.
5. Open Library and verify source entry points, Import/Export, and Search/RAG are visible.
6. Open Settings and verify global preferences and appearance/setup recovery are discoverable.

## First-Time Orientation Result

- Home exposes the product model as dashboard, notifications, status, active work, and next actions.
- Home primary action is `Set up Console model`, which is appropriate for a first-time user with no configured model.
- The shell exposes the command-palette hint with `Ctrl+P`, preserving a power-user path without requiring recall.
- Console, Library, and Settings are reachable from visible top-level navigation.

## Setup And Recovery Result

- Console shows provider setup recovery through `Provider setup needed`, rather than presenting generation as ready.
- Library exposes `Import/Export Sources` and `Search/RAG`, giving a first-time user clear source-ingestion and retrieval entry points.
- Settings exposes global preferences and Appearance, giving the user an obvious setup/control destination.

## Defects Found

No P0 or P1 release blockers were found in this replay.

P2/P3 residual risks:

- This replay verifies mounted first-run orientation, not a full terminal screenshot approval pass. Visual screenshot approval remains in `TASK-13.4`.
- This replay verifies setup recovery copy without requiring live provider credentials. Live provider generation remains covered by later power-user/recovery gates.

## P0/P1 Decision

No P0/P1 findings require fixes or explicit acceptance for `TASK-13.2`.

## Residual Risk

Release readiness still depends on the remaining Phase 6 gates:

- `TASK-13.3`: power-user workflow release replay.
- `TASK-13.4`: keyboard/focus/accessibility and visual sweep.
- `TASK-13.5`: recovery/setup/documentation alignment.
- `TASK-13.6`: packaging/configuration/data-safety validation.
- `TASK-13.7`: public roadmap release closeout.

## Verification

- `python -m pytest -q Tests/UI/test_product_maturity_phase6_first_time_release_replay.py --tb=short`
- Regression file: `Tests/UI/test_product_maturity_phase6_first_time_release_replay.py`
