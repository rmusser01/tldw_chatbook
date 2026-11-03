---
id: TASK-36533
title: Five test suites stub json.loads process-wide with a one-argument function
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:44
updated_date: '2026-11-03 10:54'
labels:
- follow-up
- tests
priority: low
references:
- Tests/Sync_Interop/test_sync_crypto.py
- Tests/Prompts_DB/test_prompts_db_retained_history.py
- Tests/Agents/test_run_log_service_wiring.py
- Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py
- Tests/Image_Generation/test_comfyui_image_adapter.py
dependencies:
- TASK-36381
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36381's implementation and review (branch fu/h-settingsreds, 2026-10-23). `monkeypatch.setattr(module.json, "loads", stub)` replaces the stdlib function for the whole process, not just for the module under test.

TASK-36381's red came from exactly this. A probe's storage admission called `json.loads(data, object_pairs_hook=...)` and got a `TypeError` from a stub (lesson in lessons-testing-evidence.md).

These stubs accept only one positional argument, so any guarded read that runs inside their window fails the same way:
- `Tests/Sync_Interop/test_sync_crypto.py:58-61`
- `Tests/Prompts_DB/test_prompts_db_retained_history.py:433-455`
- `Tests/Agents/test_run_log_service_wiring.py:688-704`
- `Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py:157-162`
- `Tests/Image_Generation/test_comfyui_image_adapter.py:1034-1036`

They all pass on dev today; each is a trap for the next change near it. Other process-wide stubs (MCP, Backup_Recovery, vllm, chat dictionary) take `*args, **kwargs` and are not affected.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 None of the five suites replaces json.loads for the whole process with a stub that refuses json.loads' keyword arguments
- [x] #2 Each changed test still fails when the behaviour it pins is removed
<!-- AC:END -->

## Implementation Plan

1. In each of the five suites, replace `monkeypatch.setattr(<module>.json, "loads", stub)` (which rebinds the stdlib function for every module) with `monkeypatch.setattr(<module>, "json", SimpleNamespace(loads=stub, ...))`, so only the module under test sees the stub. The namespace carries the other `json` names that module reads (`dumps`, `JSONDecodeError`, `load`), following the precedent in `Tests/Chat/test_local_server_discovery.py` and `Tests/Agents/test_hook_permissions.py`.
2. Keep the run_log change local to its one test (owner draft #3345 edits that file).
3. AC#1 evidence: a temporary probe inside each stub window calling `json.loads(s, object_pairs_hook=dict)` is red (`TypeError`) on the old stubs and green on the new patches; revert the probe.
4. AC#2 evidence: for each changed test, temporarily remove the source behaviour it pins and show the test red, then revert.
5. Run the five files with `-n 4`, plus the suites for the modules under test, and preflight.

## Implementation Notes

Each of the seven tests now patches the module under test's own `json` name: `monkeypatch.setattr(<module>, "json", SimpleNamespace(loads=stub, ...))`. Before, they patched `<module>.json.loads`, which is the stdlib function every module shares. No stub reaches another module, so its one-argument signature no longer matters. The namespace carries the other `json` names that module reads (`dumps`, `JSONDecodeError`, and `load` for the ComfyUI adapter). A namespace without `JSONDecodeError` turns `crypto.py`'s propagating `RuntimeError` into `AttributeError` (probed); the lesson entry now says so. The run_log edit stays inside its one test, with a local import, because owner draft #3345 edits that file. The ComfyUI file shares one `_adapter_json()` helper across its three tests.

Evidence (all through run.sh, `-n 4`, load 7-34 on 18 CPUs):
- AC#1 red/green: a temporary probe inserted after each stub install called `json.loads('{"k": 1}', object_pairs_hook=dict)`, as an unrelated guarded read would. On the old stubs that gave 9 failed, 221 passed, every failure `TypeError: ... got an unexpected keyword argument 'object_pairs_hook'`. On the new patches it gave 230 passed. The probe was then removed.
- AC#2: in each control, the source behaviour a changed test pins was removed, and that test went red. Each source file was restored with `git checkout`.
  - crypto: `except Exception` wraps the error, giving `ValueError` where the test expects `RuntimeError`.
  - Prompts_DB count: decoding rows in the count trips the stub's `AssertionError`.
  - Prompts_DB page: LIMIT page+2 gives `assert 4 == 3` on the decode count.
  - trajectory: a double decode gives `assert 2 == 1`. Treating MemoryError as not-structured gives `assert 65 is None`.
  - disk cache: removing the size check trips the stub's `AssertionError`.
  - ComfyUI: a parse before the bound checks makes `loads_calls == []` fail in all 4 nodes.
- Five files: 230 passed on base and on the branch. Preflight passed.
- Wider run (Tests/Sync_Interop, Tests/Prompts_DB, Tests/LLM_Provider_Catalog, Tests/Image_Generation, Tests/Chat/test_trajectory_*.py, Tests/Agents/test_run_log*.py):
  - branch: 25 failed, 2424 passed, 8 errors;
  - base, same command: 24 failed, 2425 passed, 8 errors, with the same node set.
  - The one branch-only node, `test_prompts_db_legacy.py::...::test_concurrent_updates_to_same_prompt`, is an unchanged thread race. It fails 6/6 alone, and 4/6 next to the base retained-history file.

Files: Tests/Sync_Interop/test_sync_crypto.py, Tests/Prompts_DB/test_prompts_db_retained_history.py, Tests/Agents/test_run_log_service_wiring.py, Tests/LLM_Provider_Catalog/test_model_discovery_disk_cache.py, Tests/Image_Generation/test_comfyui_image_adapter.py, backlog/docs/lessons-testing-evidence.md.
