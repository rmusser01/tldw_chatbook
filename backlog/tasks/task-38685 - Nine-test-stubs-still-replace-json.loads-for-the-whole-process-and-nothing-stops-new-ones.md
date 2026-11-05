---
id: TASK-38685
title: "Nine test stubs still replace json.loads for the whole process, and nothing stops new ones"
status: Done
assignee:
- '@claude'
created_date: 2026-11-03 11:40
updated_date: '2026-11-04 02:28'
labels:
- follow-up
- tests
priority: low
references:
- Tests/Backup_Recovery/test_mcp_source_lifetimes.py
- Tests/Backup_Recovery/test_async_file_participant_lifetimes.py
- Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py
- Tests/MCP/test_execution_log.py
- Tests/Character_Chat/test_chat_dictionary_injection_cache.py
- Tests/LLM_Management/test_vllm_profiles.py
dependencies:
- TASK-36533
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36533's review, PR #3378, 2026-11-03). Checked on dev 9bbd5c2515 and at #3378's head.

TASK-36533 scopes five one-argument stubs to the module under test. Nine `monkeypatch.setattr(<module>.json or json, "loads", ...)` sites in six files remain. `<module>.json` is the stdlib module object, so each site replaces `json.loads` for every thread in the process while it is installed:
- `Tests/Backup_Recovery/test_mcp_source_lifetimes.py`: :78, :151, :177 and :544 (parse and pause stubs);
- `Tests/Backup_Recovery/test_async_file_participant_lifetimes.py`: :403 (gated);
- `Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py`: :899 (`parse_then_pause`);
- `Tests/MCP/test_execution_log.py`: :330 (counting);
- `Tests/Character_Chat/test_chat_dictionary_injection_cache.py`: :869 (counting wrapper);
- `Tests/LLM_Management/test_vllm_profiles.py`: :309 (`unexpected_parse`, which raises on any call).

These stubs accept `*args, **kwargs`, so they do not raise the `TypeError` that TASK-36533's stubs did. They still see every other `json.loads` call in the process:
- the counting stubs count unrelated parses;
- the pausing stubs can stall another thread's parse;
- the vllm stub makes any concurrent parse raise.

No failure has been traced to them yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 None of the nine sites replaces json.loads outside the module under test, and each changed test still fails when the behaviour it pins is removed
- [x] #2 A guard test fails on any new patch of json.loads on the stdlib json module under Tests/ (shown red with a planted stub)
<!-- AC:END -->

## Implementation Plan

1. At each of the nine sites, replace `monkeypatch.setattr(<module>.json or json, "loads", stub)` with `monkeypatch.setattr(<module>, "json", SimpleNamespace(**{**vars(json), "loads": stub}))`, so only the module under test sees the stub and it keeps every other `json` name it reads. Targets: `MCP/permission_store.py` (five sites), `Chat/prompt_history.py`, `MCP/execution_log.py`, `Character_Chat/Chat_Dictionary_Lib.py`, `UI/LLM_Management/vllm_profiles.py`.
2. Run the six files with `-n 4`; every changed test must stay green, so each stub still fires where it did.
3. AC#1 controls: for each changed test, temporarily remove the source behaviour it pins, show it red, revert.
4. AC#2: add an AST guard under `Tests/CI/` (the fast PR lane runs that directory whole) that fails on any `setattr`/`patch.object` of `loads` on `json` or `<x>.json`, any `patch("...json.loads")` string target, and any `json.loads = ...` assignment under `Tests/`. Show it red with a planted stub, then remove the plant.
5. Preflight.

## Implementation Notes

All nine sites now patch the module under test's own `json` name with `SimpleNamespace(**{**vars(json), "loads": stub})`. The spread keeps every other `json` name the module reads, so no hand-kept list can miss `JSONDecodeError`. The five permission-store sites share `_scope_loads()` in `test_mcp_source_lifetimes.py`; `test_guarded_raw_evidence_reuse.py` already imported its fixture from there. The other sites scope to `prompt_history`, `execution_log`, `Chat_Dictionary_Lib` and `vllm_profiles`.

Guard: `Tests/CI/test_json_loads_patch_guard.py`. The fast PR contract job runs `Tests/CI` whole, so no workflow or census edit was needed. It parses every `Tests/**/*.py` that passes a text pre-filter and flags `setattr`/`delattr`/`patch.object` of `"loads"` on `json`, on an alias of it or on `<x>.json`; a `"json.loads"` or `"<...>.json.loads"` patch target; and `json.loads = ...`. It takes about 3 s at load 15. Eight parametrized cases keep each form detected, and one case pins that the module-scoped form passes.

Evidence (all through run.sh at load 9-19 on 18 CPUs):
- Found while doing this: the two execution-log "no reparse" tests were red on dev, `assert 115 == 0`. The process-wide counter was counting storage-admission parses from other modules. Scoped, the file gives 26 passed.
- Six files, `-n 4`: base gives 44 failed, 243 passed; the branch gives 42 failed, 245 passed. The failing node sets are identical apart from those two execution-log tests. The other 42 are red on dev and are not touched here; see proposed follow-ups.
- AC#1 controls: each source edit below was temporary and reverted with `git checkout`.
  - `raw._scope` without MCP nested-scope reuse: the rmw-admitted test and the 3 corrupt-move cases go red, `RecoveryRequired: storage_locally_paused`.
  - `raw._scope` and `_mcp_observation` lending a same-path source the admitted operation: the borrow test goes red (`raw_operation_provenance_invalid` where `storage_locally_paused` is expected).
  - `check_destination` and `published` identity checks removed: identity-swap `[destination]` and `[backup]` give `DID NOT RAISE`. `raw_parent_identity_changed` refusals removed: `[parent]` gives `DID NOT RAISE`.
  - Composer recall with `exclusive=False`: `warm.is_cancelled` is False.
  - Execution-log cache lookup disabled: `310 == 0` and `135 == 0`.
  - Chat-dictionary bundle cache get returning None: second send gives `(3, 300, 3, 150)`, so the scoped spy sees the module's three entries parses.
  - vLLM size checks removed: `Failed: oversized profile document reached JSON parsing`.
- `test_actual_permission_parse_pause_matches_original_refusal[True/False]` is red on dev and on the branch at its warm precondition (`the actual source must be warm before the pause stimulus`). With that assert temporarily removed, both stubs (old and scoped) give `DID NOT RAISE`: on dev the admitted RMW completes across the pause, as `test_permission_rmw_admitted_before_pause_finishes` pins. A temporary probe showed the scoped stub fires exactly once (`len(pauses) == 1`). Its pinned refusal cannot be shown green here.
- AC#2: a planted multi-line `monkeypatch.setattr(json, "loads", ...)` in a temporary `Tests/Utils/` file made the guard fail at `test_planted_json_stub_tmp.py:5`; the file was then deleted. Run on dev's blobs, the scan flags exactly the nine task lines (78/151/177/544, 403, 899, 330, 869, 309), and none on the branch.
- `Tests/CI`, `-n 4`: 1187 passed, 1 skipped.

Files: the six test files above, `Tests/CI/test_json_loads_patch_guard.py`, `backlog/docs/lessons-testing-evidence.md` (incident added to the existing json.loads entry).

Review fix (2026-11-04):
- The guard now also recognises:
  - a `json` bound by `from pkg import json [as j]`;
  - `target=`, `name=` and `attribute=` keywords;
  - `patch.multiple(json | "json" | "<pkg>.json", loads=...)`. The text pre-filter now matches `loads=`.
- The 9 new parametrized cases were red on the old scanner and are green now.
- A planted `mock.patch.multiple(json, loads=...)` in a temporary `Tests/Utils/` file made the full scan fail at its line; the file was then deleted.
- The docstring lists what a syntax scan cannot see: rebinding, `sys.modules`, `importlib`, `setitem` on `json.__dict__`/`vars(json)`, and an attribute name held in a variable.

