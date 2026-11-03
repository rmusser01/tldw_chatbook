---
id: TASK-36529
title: Perf Guard ignores changes to the test helpers its steps import, and two keystroke census pins run nowhere
status: Done
assignee: ['@claude']
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 10:41
labels:
- follow-up
- ci
- tests
- performance
priority: low
references:
- .github/workflows/perf-guard.yml
- Tests/Performance/test_console_keystroke_work_census.py
- Tests/private_profile.py
- Tests/UI/console_controller_stubs.py
- Tests/UI/app_factory.py
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390's review (PR #3230, 2026-10-23). Perf Guard is the only PR run of the latency and keystroke guards.
- **Trigger paths.** perf-guard.yml's `pull_request` paths are `tldw_chatbook/**.py`, `tldw_chatbook/css/**`, `Tests/Performance/**`, one Tests/Utils file, `pyproject.toml` and the workflow itself; PR #3230 adds `Packaging/speculative_voice_*.py`. The files its steps run import helpers outside those paths:
  - `Tests/private_profile.py`: five of them;
  - `Tests/UI/console_controller_stubs.py`: the keystroke census;
  - `Tests/UI/app_factory.py`: the CSS fast-path test;
  - and every test loads the root `Tests/conftest.py`.

  A PR that changes only these files merges without a Perf Guard run.
- **Two passing pins that nothing runs.** `test_settled_400_message_typing_traverses_no_history` (:1090) and `test_typing_does_not_rebuild_the_provider_derivation` (:1185) pass on dev (2 passed). PR #3230 adds only the census's other two behaviour nodes to Perf Guard.
- **Detection runs through the avatar.** The census sees a full Console sync during typing by counting `messages_for_session` calls. TASK-36390 found that a full sync costs 3 such calls: 2 from the avatar path (`UI/Console_Modules/character.py:368`) and 1 from the transcript. The avatar call runs only when expression reactions are on (`if react and store is not None`, character.py:366). So with reactions off, or a quieter avatar path, a full sync inside the burst counts 1 or 0, and the guard passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A PR that changes only a file a Perf Guard step imports (including Tests/private_profile.py, Tests/UI/console_controller_stubs.py, Tests/UI/app_factory.py and Tests/conftest.py) triggers Perf Guard
- [x] #2 test_settled_400_message_typing_traverses_no_history and test_typing_does_not_rebuild_the_provider_derivation run on a PR lane
- [x] #3 The keystroke census fails when a full Console sync runs inside the typing burst with avatar expression reactions off, shown by a negative control
<!-- AC:END -->

## Implementation Plan

1. AC#1: add a CI contract test next to the standalone-workflow trigger tests in `Tests/CI/test_ci_queue_pressure_contract.py`. It reads every pytest target from perf-guard.yml's steps, adds the conftest.py chain and package `__init__.py` chain pytest loads for each, follows their imports transitively by AST (stopping at `tldw_chatbook/`, which `tldw_chatbook/**.py` covers wholesale), and fails naming each loaded file that the `pull_request` or the `push` paths miss (GitHub glob semantics: `*` stops at `/`). Show it red at dev, then derive the missing `paths` entries from its output and add them to both triggers.
2. AC#2: add the two pins to the "Run Console keystroke work census" step; measure their cost against the job's 15-minute timeout and current runtime.
3. AC#3: count full Console sync passes in `_census` directly (a seam the full pass reaches once, not identity-recorded), with an anti-vacuity check that setup reached it, and assert none ran in the typing burst. Negative control: force a full sync inside the burst with `react_character_expressions = false` and show the census fails, then revert.
4. Run the CI contract tests, the census nodes and preflight through the wrapper.

## Implementation Notes

Perf Guard's triggers now cover every file its steps load, a contract test derives that list from the steps' imports, the two pins run in the keystroke step, and the census counts full syncs directly.

- **AC#1.** New `test_perf_guard_runs_when_a_file_its_steps_load_changes` in `Tests/CI/test_ci_queue_pressure_contract.py` (the fast lane runs all of `Tests/CI`). It takes every pytest target in perf-guard.yml, adds the `conftest.py` and package `__init__.py` files pytest loads for each, and follows imports by AST transitively. It does not follow `tldw_chatbook/`, which `tldw_chatbook/**.py` covers. It then checks each file against both triggers' `paths`, using GitHub glob rules (`*` stops at `/`). At dev it failed listing 15 files, including `Tests/conftest.py`, `Tests/private_profile.py` and `Tests/UI/console_controller_stubs.py`. Those 15 went into both triggers and the test passed. Negative control: a temporary `from Tests.UI.app_factory import _build_test_app` in `console_controller_stubs.py` failed it, naming `Tests/UI/app_factory.py` and `Tests/app_module_patches.py`; then reverted. `Tests/UI/app_factory.py` is not in the list because no Perf Guard step imports it at this base. The CSS fast-path test dropped that import in 4ff3c65ba9 (2026-10-19). If it comes back, this test fails until the file is listed.
- **AC#2.** Both pins were added to "Run Console keystroke work census". Cost: on CI that step took 85.5 s for 3 nodes (4 app boots) in run 41611706329. A single-boot node there takes 18-23 s, so the step gains about 40 s. The job took 9m58s and 11m11s in the two latest green runs, against `timeout-minutes: 15`. Locally (`-n 4`, load 8-12) they took 16.9 s and 14.4 s.
- **AC#3.** `_census` wraps `config_sync.sync_console_core_state` and counts it as `full_syncs`. Every full pass that is not held or coalesced calls that seam once, through the module attribute; nothing records it by identity. Setup must have counted at least one, so a moved seam fails instead of reading 0. `test_typing_never_snapshots_the_transcript` and `test_settled_400_message_typing_traverses_no_history` assert 0 in the burst; the scale test's exact comparison includes it. Negative control: temporary copies forced `_sync_native_console_chat_ui()` after key 12.
  - Dev, reactions on: failed with 2 snapshots (`character.py:425`, `archive_recovery.py:166`).
  - Dev, `react_character_expressions = false`: failed with 1 snapshot (`archive_recovery.py:166`). The transcript's display-token gate skipped its own snapshot.
  - Dev, reactions off and archive recovery's sync quiet: **passed**, with `messages_for_session` at 0.
  - This branch in both reactions-off variants: failed with `1 full Console syncs ran inside the typing burst`.
- **Checks:** `Tests/CI` 1177 passed, 1 skipped. The five census nodes passed twice (5 passed, 34 s each, `-n 4`). preflight passed. ruff check is clean.
- **Pre-existing failure:** `test_console_storage_units_stay_within_their_ratchets` fails locally with `trace GC pass os_opens: 57 > ceiling 54 x 1.05`. It fails the same way at dev with the same command. The wrapper's deeper basetemp adds 3 opens per extra path component, as that pin's comment says.
- **Files:** `.github/workflows/perf-guard.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/Performance/test_console_keystroke_work_census.py`.
