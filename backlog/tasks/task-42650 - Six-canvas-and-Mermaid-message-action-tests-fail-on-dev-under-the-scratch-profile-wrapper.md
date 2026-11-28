---
id: TASK-42650
title: "Six canvas and Mermaid message-action tests fail on dev under the scratch-profile wrapper"
status: Done
assignee:
- '@claude'
created_date: 2026-11-27 18:00
updated_date: 2026-11-28 17:34
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_message_actions.py
- tldw_chatbook/Chat/console_message_actions.py
- tldw_chatbook/Canvas/config_policy.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 27 (2026-11-27) while rerunning, on the stack's final top, the test files that dev's #3733 touched. Six tests in `Tests/Chat/test_console_message_actions.py` fail on dev 9c4c8f9768 exactly as on the stack, run whole or each alone, under the scratch-profile wrapper:

- `test_mermaid_fences_keep_html_ordinals_and_use_distinct_replay_identity`
- `test_oversized_mermaid_keeps_discovery_safe_and_refuses_open`
- `test_mermaid_compile_repair_does_not_advertise_v1_diagram_execution`
- `test_canvas_html_actions_use_parsed_fences_and_stable_block_identity`
- `test_canvas_candidate_defers_compatibility_validation_until_open`
- `test_canvas_action_discovery_and_dispatch_do_not_compile_candidates`

They build `ConsoleMessageActionService()` without a `canvas_enabled_reader`, so the service uses `Canvas/config_policy.py::get_canvas_execution_enabled`. Inside these tests that reader raises `RecoveryRequired('raw_source_selection_changed')` (traced 2026-11-27 with the reader wrapped after conftest loads), and `ConsoleMessageActionService._read_canvas_enabled` turns any exception into "disabled". It is not the config default: the shipped default and the code default both turn Canvas on. Each test then sees Canvas as disabled: dispatch returns `blocked` with "Canvas is disabled. Restart Chatbook after re-enabling it.", discovery lists no canvas actions, and two get `None` where they expect a block reference (`AttributeError: 'NoneType' object has no attribute 'language'`).

No PR lane runs the file whole (the fast PR contract runs four of its nodes). The workflows that do run it whole (`nightly-deep.yml`, and `test.yml`'s core tests) have not run in the last month (last runs 2026-09-27 and 2026-10-03), so whether the six are red on CI or only under the local wrapper is unknown. A scheduled `nightly-deep.yml` run tests `main` (scheduled workflows run on the default branch), so evidence for dev needs a run dispatched on `dev` or a PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Why the default Canvas reader raises `RecoveryRequired('raw_source_selection_changed')` in these tests is found, and whether CI's environment hits it too is checked
- [x] #2 The six tests pass on dev under the scratch-profile wrapper, without enabling Canvas in any shared or real config and without hiding a reader failure the product should report
- [x] #3 The whole file runs in a PR lane, so a red there shows on PRs
<!-- AC:END -->

## Implementation Plan

1. AC#1: trace the reader's `RecoveryRequired` (conftest's collection-time config binding against `isolate_test_environment`'s per-test redirect), compare `run.sh`'s environment with the CI job's, and find a CI log of one of the six nodes.
2. AC#2: give the six tests an explicit `canvas_enabled_reader` (the precedent of `Tests/Canvas/test_canvas_kill_switch.py`); keep every assertion; no product or config change. Show the six red before and green after.
3. AC#3: replace the four node ids in `pr-fast-lane`'s "Run fast PR contract" and in `FAST_LANE_TARGETS` with the whole file, in place; update the TASK-33621.34 comment with the file's measured time.
4. Run the whole file, the covering tests, `Tests/CI` at `-n 4` and preflight through the wrapper.

## Implementation Notes

**AC#1, the cause.** `Tests/conftest.py` sets HOME, XDG and `TLDW_CONFIG_PATH` to its bootstrap profile at import, then imports `Chunking`, `RAG_Search.config_profiles` and `model_capabilities`. Their guarded config reads register the `tldw_chatbook.config` source as a raw participant whose recorded selection is the bootstrap config. For a plain test, `isolate_test_environment` then points `TLDW_CONFIG_PATH` at `tmp_path/test_data/config/config.toml`. The default reader, `get_canvas_execution_enabled`, misses the warm cache (keyed by the effective path) and enters `_load_cli_config_bootstrap` (guarded), where `_register_raw_participant` compares the recorded selection with the new one and raises `RecoveryRequired('raw_source_selection_changed')` (traced with a reader probe through the wrapper). `ConsoleMessageActionService._read_canvas_enabled` reads that as Canvas off and latches it. This is the ADR-126 single-profile guard working as designed (TASK-36381 reached the same verdict). The test was wrong, not the fixture.

**AC#1, CI.** CI hits it too. `run.sh` sets HOME, XDG_*, `TLDW_CONFIG_PATH` and `PYTHON_KEYRING_BACKEND`. Conftest overwrites every one that matters before the binding. The CI step's env block sets none of them (only `PYTEST_TIMEOUT`, `PYTHON*` and the setup-python paths). CI log: Release candidate qualification run 44955437105, job 137004879842 ("Core Tests (all but UI) - shard 2/6", hosted Linux, 2026-11-22, head 241b9256c1). It shows `FAILED Tests/Chat/test_console_message_actions.py::test_oversized_mermaid_keeps_discovery_safe_and_refuses_open - AssertionError: assert '8 KiB' in 'Canvas is disabled. Restart Chatbook after re-enabling it.'`, the local failure word for word. The shard runs with `--maxfail=1`, so it stopped there and the other five never ran. An older CI log shows the same refusal raised visibly: UI Fast Lane (7), run 39319900193, 2026-10-19, TASK-34982.

**AC#2.** The six tests build the service through `_canvas_on_service()`, which passes `canvas_enabled_reader=lambda: True` (the `Tests/Canvas/test_canvas_kill_switch.py` precedent). They are about fences, identity and dispatch, not config. No product code changed, no config is written, and every assertion is kept. The reader itself stays covered by `Tests/Canvas/test_canvas_settings.py` and the kill-switch suite. Red/green, same command (`run.sh ... pytest -n 4 --timeout=180 Tests/Chat/test_console_message_actions.py`): before, 6 failed and 131 passed; after, 137 passed. Each of the six also passes alone.
- `Tests/Backup_Recovery/test_inventory.py::test_normal_config_and_pure_resolvers_agree` is not cleared. It calls the guarded `config.get_user_data_dir()` itself, so no reader seam applies. Under a temporary `bootstrap_profile` marker (reverted) it fails differently, with `config_directory_selection_changed`, because it patches `get_cli_setting` to move `data_dir`. It needs a selected config source (`select_config_source`). It stays with TASK-41116's open AC#1 and its CI `--deselect`.

**AC#3.** `pr-fast-lane`'s "Run fast PR contract" and `FAST_LANE_TARGETS` now list `Tests/Chat/test_console_message_actions.py` whole, in place of the four node ids, beside the approval riders. The TASK-33621.34 comment is kept and extended. No other `Tests/CI` pin lists those nodes. Measured: 137 tests in 5.4 s of test time serially (load 10); the four nodes were 13 tests in 0.8 s. The whole step, run locally with the lane's flags (`-p no:xdist --timeout=180 --tb=short`): 5176 passed, 4 skipped in 24:07 (load 8.6-11.9). Pin red/green: with the old `FAST_LANE_TARGETS` against the new workflow, `test_fast_lane_target_set_is_exact_and_non_overlapping` failed; with the new pin it passed.

**Checks.** `Tests/CI -n 4`: 2080 passed, 2 skipped. Preflight passed. `Tests/lint/ruff_gate.py` passed (RATCHET OK). `ruff --select F` on both changed test files passed.

**Files.** `Tests/Chat/test_console_message_actions.py`, `Tests/CI/test_ci_queue_pressure_contract.py`, `.github/workflows/derived-artifacts.yml`, `backlog/docs/lessons-testing-evidence.md` (new entry).
