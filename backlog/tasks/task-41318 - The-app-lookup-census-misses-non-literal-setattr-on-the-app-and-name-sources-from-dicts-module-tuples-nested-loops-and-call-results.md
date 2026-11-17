---
id: TASK-41318
title: "The app-lookup census misses non-literal setattr on the app, and name sources from dicts, module tuples, nested loops and call results"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- architecture
- tests
priority: medium
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_nonliteral_lookups.tsv
- tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py
- https://github.com/rmusser01/tldw_chatbook/pull/3538
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. These are the known ceilings of TASK-40141 (PR #3538), recorded in its `ponytail:` comment. Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code and running a throwaway AST scan; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** TASK-40141 made the census follow literal names that reach the app object through a variable, a helper parameter or a literal loop. Two gaps remain. A string lookup or app mutation hidden in one of these places is not counted. So an "explicit service" migration can move a literal there and read as a reduction, which is the dodge TASK-40141 set out to close.

**Evidence.**
- **`setattr` is not scanned at all.** The scan matches only `getattr`/`hasattr` (`Tests/Architecture/test_app_string_lookup_census.py:160-171`). An AST scan for `setattr`/`delattr` on the census receivers (`_RECEIVERS`, `:45`) with a non-literal name finds 15 sites. Six are in `Event_Handlers/LLM_Management_Events/server_lifecycle.py` (`:99`, `:255`, `:283`, `:311`, `:382` with `process_attr`; `:586` with `attribute`). Others: `Chat/console_runtime.py:6915`, `Chat/console_fleet_attention.py:69`, `UI/Screens/chat_screen.py:15816` and `:15844`, `UI/Screens/library_screen.py:681`, `UI/Navigation/main_navigation.py:110`, `RAG_Search/semantic_availability.py:243`, `UI/Library_Modules/library_file_export.py:253`, `UI/Console_Modules/reaction_preview.py:59`.
- **Name sources the census does not follow** (their rows read `-`; 31 of the 47 rows in `Tests/Architecture/app_nonliteral_lookups.tsv`):
  - dict lookups: `_SOURCE_DB_ATTR.get(source_type)` (`Scheduling/automation_health.py:73`), `service_attributes[owner]` (`UI/Screens/library_screen.py:22320`);
  - module-level tuples: `LAB_SERVER_SOURCES` (`UI/Lab_Modules/lab_server_status.py:38`, `:92`);
  - nested loops: the `declarations` loop in `Backup_Recovery/runtime_maintenance.py:457`;
  - call results: `process_attr = _validate_provider(provider)` (`server_lifecycle.py:91`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census counts or records non-literal `setattr`/`delattr` on app receivers, with a negative control that plants one and fails
- [x] #2 A literal reached through a dict lookup, a module-level tuple, a nested loop or a function's literal return value is counted, each with a negative control; any source left unfollowed is listed in the test's docstring with its reason
- [x] #3 The pins and the non-literal record are re-measured, and the census passes on dev
<!-- AC:END -->

## Implementation Plan

ADR required: no. This extends an existing architecture test (TASK-36942.4, TASK-40141) without changing product code or its rules; only what the census can see grows.

1. Reproduce: negative controls that pass on the unmodified census -- a planted non-literal `setattr`/`delattr` on the app is not recorded, and a literal moved into a dict (`.get`/`[k]`), a module-level tuple, a nested loop or a function's literal return value drops out of the count and the private-name rule.
2. Record non-literal `setattr`/`delattr` on the census receivers in `app_nonliteral_lookups.tsv` (recorded, not counted: a literal `setattr` is an attribute assignment, not a string lookup).
3. Replace the loop-only follower with one that follows a name through local and module-level assignments, enclosing loops (nested loops and comprehensions), dict lookups and same-module function returns. A name that is itself an UPPER_CASE constant stays the record's business, as today.
4. List the sources left unfollowed, with reasons, in the test docstring.
5. Re-measure the pins and the record; record the numbers with a dated note.

## Implementation Notes

The census (`Tests/Architecture/test_app_string_lookup_census.py`) gains two things and keeps its rules:

- **Writes are recorded.** A non-literal `setattr`/`delattr` on a census receiver is now a row in `app_nonliteral_lookups.tsv`, under the same shrink-only rule as non-literal lookups. Writes are recorded but not counted: a literal `setattr(app, "x", v)` is the assignment `app.x = v`, not a string lookup. Counting them would also have changed what the pins mean. The scan finds the 15 sites the task lists.
- **Names are followed further.** `_loop_items` is replaced by `_Follower`. It follows a lookup's name through a plain `name = ...` (in the function or at module level), an enclosing loop's item (nested loops and comprehension generators, using Python's scoping: a loop binds in its body, not in its own iterable), a dict lookup (`D[k]`, `D.get(k[, default])`), and the `return` values of a function with that name in the same module. These compose, so `_validate_provider(provider)` -> `SERVER_PROCESS_ATTRS[provider]` -> the dict's values is followed. A cycle is cut by a per-path seen set. A name that is itself an UPPER_CASE constant is still not followed: the record's `reaches` column covers it, and the TASK-40141 `constant` control pins that. A function that binds the name in any way (`=`, `for`/`with`/`except`/import targets, tuple unpacking) owns it, and only its `=` values count, in any order. Otherwise the follower continues to enclosing functions and lambdas, then the module. A returned parameter does not make its function a lookup helper. Each lookup counts a literal node once. The docstring lists the sources left unfollowed, each with its reason. The real-tree ones are two conditional expressions, `self._CONSOLE_PENDING_STASH_ATTR` (a constant) and `SERVER_PROCESS_ATTRS.items()`.

Evidence (2026-11-16, base fbf1c9ea74 = dev 5e3c8b2bd4 plus filing commits, load 40-100):

- **Reproduction:** on the unmodified scanner, the 7 new controls failed: `test_a_nonliteral_app_write_is_recorded[setattr|delattr]` and `test_a_literal_moved_into_a_helper_or_loop_is_still_counted[dict-get|dict-subscript|module-tuple|nested-loop|call-result]`. The 8 existing controls passed. All 38 nodes pass at head.
- **Re-measured pins**, itemized per lookup site against base:
  - Backup_Recovery 34 -> 41: the `declarations` nested loop, 7.
  - Event_Handlers 41 -> 89: 8 `process_attr` lookups x the 6 `SERVER_PROCESS_ATTRS` values.
  - Scheduling 2 -> 5: `_SOURCE_DB_ATTR.get`.
  - UI 1290 -> 1312: `LAB_SERVER_SOURCES` 6, `service_attributes[owner]` 6, `HOME_CONTROL_METHODS.get` 10.
  - No other package moved, and `app_private_attrs.txt` is unchanged.
- **Record:** 47 -> 62 rows, the 15 write sites.
- **Mutation check:** each source was planted in a scratch copy of the real tree (`_planted_private`) and the census run there. Each plant failed both the right pin and the private-name rule:
  - `_SOURCE_DB_ATTR`: Scheduling +1.
  - `service_attributes`: UI +1.
  - `LAB_SERVER_SOURCES`: UI +1.
  - the `declarations` loop: Backup_Recovery +1.
  - a literal `return` in `_validate_provider`: Event_Handlers +7, one per caller.
  - A planted `setattr(app, name, None)` failed the record test. An unplanted copy passed all 38 nodes.
- **Scan cost:** CPU went from 20.7 s to 22.5-22.9 s (+~10%), measured twice each.
- `./scripts/preflight.sh` passed.
- **Serial `Tests/Architecture` on head** (load 55-100): 1420 passed, 8 failed.
  - Three fail on base too: `test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only` and `test_timer_path_static_update_inventory` x2.
  - Five were `test_task_15103_review_ledger_*` 300 s pytest-timeouts while the run rebuilt the diagnostic inventory. Run alone on head, all five passed (7.5 min).
- **Lane:** same file, so it already runs in `pr-fast-lane-arch`.
- **Review round (2026-11-17, Opus review of 7366b6641e):**
  - The pin-failure message now explains per-site counting and gives the remedy for a new dict, tuple or return value. It also says return-following matches callees by name.
  - Locality and closures were fixed as described above, returned parameters are no longer recorded as helpers, and each lookup counts a literal once.
  - The docstring's not-followed list was completed, and it now states order-insensitivity.
  - New controls:
    - `test_a_literal_moved_into_a_helper_or_loop_is_still_counted[dict-get-default|call-cycle|lambda-closure|nested-def-closure]`
    - `test_a_literal_the_name_cannot_hold_is_not_counted[9 cases]`
    - `test_a_literal_reached_twice_by_one_lookup_counts_once`
  - Before the fix, 8 of the new nodes failed. The census file now passes 52 nodes.
  - Mutation check: each of 7 mutants was caught by its control. They dropped the `.get` default, the cycle guard (RecursionError), the comprehension scope filter, closure following, binding-based locality, the helper-off-on-return rule, and dedupe.
  - Pins and record re-measured: unchanged.
  - Real-tree plants: re-run with the same results, and the unplanted copy passes 52 nodes.
  - `./scripts/preflight.sh` passed with `PYTHON=` set to the repo's 3.12.11 venv. This worktree has no `.venv`, so preflight otherwise picks `python3.14` (3.14.7), and the Canvas Mermaid check then fails on its 3.12.11 build pin. That failure is environmental.
- **Files:**
  - `Tests/Architecture/test_app_string_lookup_census.py`
  - `Tests/Architecture/app_lookup_census.tsv`
  - `Tests/Architecture/app_nonliteral_lookups.tsv`
