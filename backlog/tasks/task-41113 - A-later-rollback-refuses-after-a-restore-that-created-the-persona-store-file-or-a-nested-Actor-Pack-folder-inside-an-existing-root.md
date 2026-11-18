---
id: TASK-41113
title: "A later rollback refuses after a restore that created the persona store file or a nested Actor Pack folder inside an existing root"
status: Done
assignee: ['@claude']
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 02:00
labels:
- follow-up
- backup
- actor-packs
priority: low
references:
- tldw_chatbook/Backup_Recovery/later_rollback.py
- Tests/Backup_Recovery/test_created_persona_subtree_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The rollback copy promises the user can undo the restore later. These two shapes cannot be undone.

**Evidence (TASK-40305's notes, review r14act-rev1).**
- A restore from another profile that creates `tldw_chatbook_personas.json` refuses its later rollback with `local_snapshot_absence_unclassified` (observed; the test gives its target a persona of its own to avoid it).
- A restore that creates a nested pack folder inside an existing root refuses the same way (read from the code, not run).
- `later_rollback.py:634` (`_CREATED_TREE_ROOT_OWNERS`) admits created roots only for whole tree owners; the refusals are at `:1449`, `:1459` and `:1888`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A later rollback of a restore that created the persona store file completes and removes it, with a test red first
- [x] #2 The nested-folder case is run; if it refuses, it is fixed the same way, with a test red first
<!-- AC:END -->

## Implementation Plan

1. Red first, end to end, in `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`'s TASK-40305 scenario script: drop the workaround that gave the created-target profile a persona of its own, and assert the persona store is absent before the restore, created by it, and removed by the later rollback. Add a `nested-target` route and test: the target holds another pack, so the restore creates the incoming pack folder inside both existing roots; the rollback must bring the trees back byte for byte.
2. Run both at base and record each refusal and where it is raised.
3. Fix in `later_rollback.py` along the existing created-root proof (`_created_destination_target`): admit a restore-created persona store file and a restore-created nested Actor Pack folder only with the same evidence a created root needs (committed journal, live activation generation, unchanged inode, rediscovered members), then let `_known_absences` retire them.
4. Green; rerun the file's other TASK-40305 routes and the negatives.

## Implementation Notes

Both shapes were run end to end and both refused at base (cc6006fcf5) with `local_snapshot_absence_unclassified`, raised by the first check in `later_rollback._created_destination_target` (`:1449` there):
- **Persona store file (AC1).** `_ACTOR_TREES_ROLLBACK`'s `created-target` profile no longer gets a persona of its own (the TASK-40305 workaround), and the route now asserts the store is absent before the restore, created by it, and gone after the later rollback. The created artifact is `profile:<source>:personas`, `kind='file'`; its archived item sits under a synthetic archive root (`root_id == parent_id == 'root:<sha>'`), so neither the root rule nor the nested-persona rule admitted it.
- **Nested Actor Pack folder (AC2).** New route `nested-target` and test `test_later_rollback_of_a_restore_that_created_nested_actor_pack_folders`: the target holds its own pack, the restore publishes the source's pack as two new nested folders (`actor_packs/<d>` under the `persona.actor_packs` root and `visual_identities/actor_packs/<d>` under `persona.visual_identity`) and retires the target's. The later rollback refused both folders the same way.

Fix (`tldw_chatbook/Backup_Recovery/later_rollback.py`), along the existing created-root and nested-persona proofs, no new kind of evidence:
- `_CREATED_FILE_OWNERS = {"personas"}`: a restore-created file of that owner is admitted only as an exact single file under its archive root, published as a `file` candidate at the same path, owned by the installed `_Definition`, and then goes through the same proof as a created root (committed journal, config dependency, no safety-scope overlap, the live activation generation and its witnesses, unchanged device/inode, rediscovery of exactly that file, validation and digest). `_known_absences` then retires it; `_stage_read_sources` adds it to the held capture scope, without which the execution-time re-preview could not read it (`local_snapshot_created_member_invalid`). Other one-file data owners (dictionary history, RAG context, ...) have the same gap; they can join the set with a test.
- `_NESTED_CREATED_OWNERS = {"persona.assets", "persona.actor_packs", "persona.visual_identity"}`: the nested-pack path that only `persona.assets` had (`_created_persona_subtree`, its ancestor proof against the restore's recorded directory metadata, `_known_absences`' retained ancestors, `_stage_read_sources`' active roots) now takes the owner from the item. `_stage_read_sources` skips a nested folder whose root is itself a retired created root: it is that root's member, and counting it as an active root refused the created-target rollback (`local_snapshot_created_scope_unverified`) once the two Actor Pack owners joined.

Red/green (each child run through `run.sh`, load 70-110):
- Red: both tests failed `ValueError: local_snapshot_absence_unclassified` (`later_rollback.py:1449`).
- Intermediate reds that pinned each part: without the stage-read source, `local_snapshot_created_member_invalid` at execution; without the created-root exclusion, `local_snapshot_created_scope_unverified` in `_stage_read_sources`.
- Green: `-k later_rollback_of_a_restore -n 3`: 3 passed (changed, created incl. the persona store, nested), 452 s.

Covering runs (`-n 4`): `test_later_created_destinations.py`, `test_later_rollback.py`, `test_config_file_retirement.py`, `test_retirement_only_rollback.py`, `test_rollback_group_scope.py`, `test_later_rollback_handoff.py`, `test_eval_later_snapshot.py`, `test_persona_observation_budget.py`, `test_builtin_later_snapshot.py`: 159 passed, 14 failed at load ~100. Rerun of the 14 at load 12-42: 2 passed (one created-destinations node had hit `Timeout (>300.0s)`), 13 failed, and the same groups fail with base `later_rollback.py` and base test file (same command, load 44): the 12 `test_persona_observation_budget.py` nodes (`TypeError: _observe.<locals>.child() got an unexpected keyword argument 'idle'`; its fake `_child` predates TASK-40306's `idle=`) and `test_builtin_later_snapshot.py::test_shared_builtin_later_preview_and_execution_preserve_both_profile_trees` (pytest-timeout >300 s in SQLite validation; both params at base). Proposed follow-up for the coordinator: the budget test's fake `_child` needs `idle`. The file's 9 `test_native_persona_mapping_refuses_unproved_graph` negatives ran on this final `later_rollback.py` in the last iteration of TASK-41115's capture campaign (two lanes): 18 of 18 passed.

Files: `tldw_chatbook/Backup_Recovery/later_rollback.py`, `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py` (`_ACTOR_TREES_ROLLBACK` routes and the new nested test).

### Review r15ac-rev1, fix round 1

- **Finding 1 (Important): the target's own persona store.** The `nested-target` route now captures the target's `tldw_chatbook_personas.json` after its own activation and asserts the restore replaced it and the later rollback brought it back byte for byte. That is the invariant the new `personas` retirement path depends on: only a restore-created file (`previous is None`) is retired. Green: 1 passed (523 s, load ~60). Control: with `and row.previous is None` removed from `_created_destination_target`'s artifact filter, the later rollback is refused (`ValueError: local_snapshot_absence_unclassified`, `later_rollback.py:1482`). Other replaced artifacts in that restore then fail the created-artifact rule before the store would be retired, so the control goes red by refusal, not at the store assertion. The reviewer's probe (`$FU/r15ac-rev1/probe_own_target.py`) also covers a target whose only own data is its persona store.
- **Finding 3 (Minor):** the route comment now says `created-source` serves both target routes, and "had neither tree nor a persona store".
