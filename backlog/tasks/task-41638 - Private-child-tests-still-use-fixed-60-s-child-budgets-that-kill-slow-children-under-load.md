---
id: TASK-41638
title: "Private-child tests still use fixed 60 s child budgets that kill slow children under load"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
- flaky
priority: low
references:
- Tests/Chat/test_workflow_planning_native.py
dependencies:
- TASK-41314
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41314 raised its private-child budgets to 180 s because loaded runners killed children that were still making progress. Other tests that run a private child keep a fixed 60 s budget, so they fail the same way under load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every private-child test either uses a budget that tolerates load or uses the shared CPU-idle-aware wait
- [x] #2 A grep for a fixed 60 s child budget in `Tests/` finds no remaining site, or each remaining one is justified in a comment
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test budgets only).

1. Inventory with an AST scan of `Tests/` for literal 60 s budgets on child processes (`subprocess.run`, `Popen.wait`, `_child`, `asyncio.to_thread(subprocess.run)`) and 60 s child-poll deadlines. The shared `_run` (`Tests/Backup_Recovery/test_home_citation_retirement.py`) already waits through the CPU-idle-aware `_wait_while_busy` (TASK-41117), so its 87 `timeout=60` callers are idle bounds and need no edit.
2. Private children (repo code against a profile the test isolates, or a nested pytest child): children that log to a file move to `_wait_while_busy` (`Tests/Backup_Recovery/restart_observation.py`) or `_child(..., idle=60)`; children read through `capture_output` pipes get a 180 s hang bound (TASK-41314 precedent), because an idle wait on a full pipe would kill a child blocked on its own output.
3. 60 s waits that only reap an already SIGKILLed child stay, with a comment saying why.
4. Run the touched files at `-n 4`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**What a "fixed 60 s child budget" is.** A wall-clock 60 s limit on a child process that runs repository code against a profile the test isolates (a private App, DB or driver child, or a nested pytest child). The inventory was an AST scan of `Tests/`: calls with `timeout=60` or `60.0` (literal, conditional or computed), `monotonic() + 60` child deadlines, and helper defaults of 60.

**Already load-tolerant, unchanged.** 87 `_run(..., timeout=60)` callers (shared `Tests/Backup_Recovery/test_home_citation_retirement.py::_run`). Since TASK-41117 (`7e98d6bc04`), `_run` waits through the shared CPU-idle-aware `_wait_while_busy` (`Tests/Backup_Recovery/restart_observation.py`). Its `timeout` is a no-CPU hang bound with a 4x wall backstop. The same applies to `_run_profile_child` (`test_complete_roundtrip.py`, `idle=60`) and the four `_wait_while_busy(idle=2, timeout=60)` calls in `test_config_sibling_capture.py`.

**Changed.**
- Shared CPU-idle wait (children that log to a file):
  - the workflow pause probe (`test_workflow_pause_partial_native.py`) and the workflow wave probe (`test_workflow_phase_wave_native.py`) use `Popen` + `_wait_while_busy(idle=60, timeout=240)`;
  - both persona negatives (`test_created_persona_subtree_rollback.py`) use `_child(timeout=600, idle=_HANG_SECONDS)`, as the module fixture's own children do. The overlap negative's conditional `180 if win32 else 60` was missed by the first scan and fixed in its own commit.
- 180 s hang bound (children read through `capture_output` pipes, where an idle wait would kill a child blocked on a full pipe; TASK-41314 precedent): `workflow_wave_archive.py` (REOPEN), `test_first_replacement_binding.py`, `test_console_ssh_custody.py` and `test_compound_chat_source_lifetimes.py` (both nested pytest children), `test_project_index_detached_cleanup.py`, `test_state_paths.py`, `test_library_prompt_evidence_driver.py` (x3), and the 60 s reach-the-step deadline in `test_chachanotes_bare_open_self_migration.py`.
- Justified in a comment, not changed: the 60 s `child.wait` calls that only reap an already SIGKILLed child (`test_chachanotes_bare_open_self_migration.py`, `test_profile_reference_storage.py`).

**Not private children, left as they are.** These run no isolated profile. They are import-closure and lazy-import probes (`python -c` checking `sys.modules`), packaging/wheel probes, an SQLite lock probe (`test_admission_path_resolution.py`), and external tools or other interpreters (`git archive`, `ffmpeg`, `nvidia-smi`, `docker`, `python3.10` parse/version probes). The list is in the hand-off.

**Evidence** (scratch-profile wrapper; the touched nodes ran together with the TASK-41636 accounting set at `-n 4`, load 65-80):
- Passed on head: the wave owned-close node, the three nested/cold children, state paths, both SIGKILL tests, and first-replacement binding.
- `test_remaining_original_suppressed_workspace_close_retains_custody` failed in the bulk run (load) and passed in the 12-node rerun.
- Fail on base too, so not this change:
  - `test_native_wave_complete_archive_restore_and_rebackup`;
  - all 4 `test_library_prompt_evidence_driver.py` nodes: the driver child fails at import, `cannot import name 'validate_username' from 'tldw_chatbook.Utils.input_validation'`.
- Persona negatives, both node functions run serially: 19 passed in 465 s at load ~45 (module fixture included).

Files (12): `Tests/Chat/{test_workflow_pause_partial_native,test_workflow_phase_wave_native,workflow_wave_archive,test_console_ssh_custody}.py`, `Tests/Backup_Recovery/{test_created_persona_subtree_rollback,test_first_replacement_binding,test_compound_chat_source_lifetimes}.py`, `Tests/Workspaces/test_project_index_detached_cleanup.py`, `Tests/RAG/test_state_paths.py`, `Tests/Library/test_library_prompt_evidence_driver.py`, `Tests/DB/test_chachanotes_bare_open_self_migration.py`, `Tests/TTS/test_profile_reference_storage.py`.
<!-- SECTION:NOTES:END -->
