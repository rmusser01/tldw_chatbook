---
id: TASK-41641
title: "The app-lookup census misses names reached through conditional expressions, enclosing loops and global or nonlocal bindings"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- architecture
- tests
priority: low
references:
- Tests/Architecture/test_app_string_lookup_census.py
- Tests/Architecture/app_lookup_census.tsv
dependencies:
- TASK-41318
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41318 taught the census to follow literal names through dicts, module tuples, nested loops and same-module returns. Its docstring lists the sources it still does not follow:
- names chosen by a conditional expression (`settings_screen.py` has one);
- a loop variable bound by an enclosing loop;
- names bound through `global` or `nonlocal`.

A lookup reached that way is not counted, so it can grow without the ratchet noticing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census counts literal names reached through conditional expressions, enclosing loops and global/nonlocal bindings
- [x] #2 Pins are re-measured with dated notes
- [x] #3 A test per new source shape asserts it is counted
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (extends an existing architecture test, TASK-36942.4 / TASK-40141 / TASK-41318; no product code or rule changes, only what the census can see grows).

1. Follow a conditional expression's two branches (`"a" if x else "b"`) wherever the follower meets one: the lookup's own argument, a name's `=` value, a dict value or a return.
2. A `for` statement's variable is its function's (or module's) variable: when an enclosing `for` binds the name, count that scope's `=` to it as well as the loop's items (the loop no longer wins). Comprehension variables stay their own.
3. `global`/`nonlocal`: a function that declares the name `global` reads the module's variable, one that declares it `nonlocal` reads the nearest enclosing function's. A variable's values include every `=` from a function that rebinds it through `global`/`nonlocal`.
4. One negative control per new shape (conditional expression, enclosing-loop `=`, `global` read, `global` rebinding, `nonlocal` read, `nonlocal` rebinding) in `test_a_literal_moved_into_a_helper_or_loop_is_still_counted`, shown red on the unmodified census; plus a scope control that a same-named `nonlocal` in an unrelated function does not leak.
5. Update the docstring's followed / not-followed lists.
6. Re-measure pins, private allowance and record against base, itemized per site; mutation-check each new rule.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The census follower (`Tests/Architecture/test_app_string_lookup_census.py`) follows three more name sources. No rule changes.

- **Conditional expressions.** `values()` follows both branches of `a if c else b`, wherever it meets one: the lookup's argument, a name's `=` value, a dict value or a return.
- **Enclosing `for` loops.** A `for` statement's variable belongs to its function (or the module). Before, an enclosing loop "won": only its items counted, and the function's `=` to the same name did not. Now the loop's items and every `=` to that variable count. A comprehension's variable is still its own.
- **`global`/`nonlocal`.** `_scan` collects `global`/`nonlocal` statements. The follower records which function declares which name. A function that declares the name `global` reads the module's variable, even inside an enclosing loop or function that binds the same name. One that declares it `nonlocal` reads the enclosing function's. A variable's values include every `=` from a function that rebinds it through `global`/`nonlocal`, matched by the variable's owner, so a same-named `global`/`nonlocal` in an unrelated function does not leak in.
- The docstring's followed and not-followed lists are updated. Boolean expressions (`x or "y"`) stay unfollowed: the task names conditional expressions only, and no real lookup reaches one today.

Evidence (2026-11-18, base = dev d0486978dd, load 60-105):

- **Reproduction:** with the unmodified follower and the new test section, 7 nodes failed:
  - `test_a_literal_moved_into_a_helper_or_loop_is_still_counted`, ids `conditional-expression`, `enclosing-loop-assignment`, `global-rebinding`, `global-read`, `nonlocal-rebinding` and `nonlocal-read`;
  - `test_a_literal_the_name_cannot_hold_is_not_counted[global-skips-enclosing-loop]`. The old follower counted the enclosing loop's item for a `global` read.

  The three other new scope controls passed before and after: `another-functions-nonlocal`, `another-functions-global` and `comprehension-variable-not-the-functions`.
- **Head:** the census file passes 65 of 65 serially (55 before).
- **Mutation check:** 8 mutants were each caught by the control named for them:
  - dropping conditional following;
  - dropping the loop's `=`;
  - `global` treated as `nonlocal`;
  - `global` ignored in the read path;
  - `nonlocal` ignored;
  - no rebinders;
  - rebinders not matched to their owner;
  - comprehensions also taking the function's `=`.
- **Re-measured pins (2026-11-18), itemized against base by the counted-literal multiset:**
  - **UI 1312 -> 1316.** This is a measurement of two existing lookups, not new code:
    - `UI/Screens/settings_screen.py:14737`: `"_llamacpp_connection_owner" if llama else "_vllm_connection_owner"`.
    - `UI/Library_Modules/library_artifacts_controller.py:1006`: `"subscriptions_db" if key.source == "live_report" else "chachanotes_db"`.
  - Each site counts 2. These are the two conditional expressions TASK-41318 named.
  - The `global`/`nonlocal` and enclosing-loop `=` rules add nothing in the real tree today. No other package moved, `app_private_attrs.txt` is unchanged (both private names were already allowed), and `app_nonliteral_lookups.tsv` is unchanged.
- **Serial `Tests/Architecture` on head** (load 60-105, 72 min): 1485 passed, 5 skipped, 4 failed. None is in a file this change touches.
  - Three also fail on base d0486978dd, run there serially:
    - `test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only`: `agent_service.py` diagnostics.
    - `test_timer_path_static_update_inventory` x2: the new `ProjectIndexBindingCard._poll` clock root.
  - `test_task_15103_review_ledger_canonical_schema_and_arithmetic` hit its 300 s pytest-timeout, in the suite and alone. With `--timeout=1800` it passed on both trees: base 291 s, head 294 s.
- `./scripts/preflight.sh` passed (`PYTHON` set to the repo venv).
- **Scan cost:** user CPU was 20.78 / 21.20 s on base and 20.64 / 20.69 s on head, in alternating runs. The change is not measurable.
- **Files:**
  - `Tests/Architecture/test_app_string_lookup_census.py`
  - `Tests/Architecture/app_lookup_census.tsv`
<!-- SECTION:NOTES:END -->
