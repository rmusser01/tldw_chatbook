---
id: TASK-1132
title: >-
  Word bench target-id-uniqueness check ran on the read path too, making
  legacy benches unopenable
status: Done
assignee: []
created_date: '2026-07-28 10:00'
labels:
  - evals
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #962 (`b73de3564`) added a check to `BenchConfig.__post_init__`
(`tldw_chatbook/Evals/word_bench/models.py`) rejecting duplicate
`target_ids`: every per-target map downstream (`WordBenchRunner`'s
`clients`, its preflight/canary dicts, `storage.create_run_group`'s
`run_ids`) is keyed by target id, so a duplicate silently collapsed two
grid columns into one shared `eval_run`. That fix is correct on the WRITE
path.

`storage.load_bench` also constructs a `BenchConfig` — from whatever is
already sitting in a stored `eval_tasks.config_data` row — so the same
validation ran on the READ path too. A bench saved before this check
existed, whose stored `target_ids` already carried a duplicate, could no
longer be opened at all: `load_bench` raised `ValueError`, and both
`tldw_chatbook/UI/Evals/bench_editor.py` and `tldw_chatbook/UI/Evals/
inspector.py` swallow that in a bare `except Exception` and render an
error placeholder instead of the bench.

Verified directly: writing an `eval_tasks` row with
`config_data.target_ids = [mid, mid]` via `EvalsDB.create_task` (bypassing
`BenchConfig`, as a pre-validation save would have produced) and then
calling `load_bench` raised
`ValueError: target_ids must be unique, got duplicates: [...]`.

There was also an existing, already-failing test stating the intended
behavior plainly:
`Tests/UI/test_evals_bench_editor.py::test_bench_with_duplicate_target_id_composes_without_raising`
— asserting both the detail pane's target table and the inspector's
readiness list render every row (not just N-1) for a bench carrying a
duplicate target id. It had been failing since #962 because its own
fixture called `BenchConfig(...)` directly with a duplicate, which #962
made raise even at fixture setup.

The fix is not to dedupe on read: silently collapsing two columns into one
on display is the original #962 bug wearing a different hat, and it would
hide the problem from the user who needs to fix it. The correct split is
write-strict, read-lenient:

- WRITE (`BenchConfig` construction from user action, `save_bench`,
  `create_run_group`, `WordBenchRunner.run`): keep rejecting duplicates
  unconditionally. A user must not be able to create or run one.
- READ (`load_bench`, and anything the editor/inspector use to display a
  stored bench): tolerate a duplicate, preserve it exactly as stored, and
  render every row, so the user can see it and remove it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A bench whose stored `config_data.target_ids` contains a duplicate can be opened: `load_bench` returns a `BenchConfig` instead of raising, preserving both duplicate entries (not deduplicated)
- [x] #2 `BenchConfig` constructed the normal way (user-facing creation, no special argument) still rejects a duplicate `target_ids` unconditionally
- [x] #3 `storage.create_run_group` and `WordBenchRunner.run` still reject a duplicate target list before any `eval_runs` row is created
- [x] #4 `storage.save_bench` rejects a duplicate even when handed a leniently-loaded `BenchConfig`, so a legacy duplicate can never round-trip back into storage un-flagged
- [x] #5 `test_bench_with_duplicate_target_id_composes_without_raising` passes: both the bench editor's target table and the inspector's readiness list render every row for a duplicate-carrying bench
- [x] #6 A test covers a legacy stored bench (written directly against `EvalsDB.create_task`, bypassing `BenchConfig`) loading and preserving both ids
- [x] #7 A test covers that user-facing creation and `create_run_group` still reject duplicates
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a `strict: InitVar[bool] = True` to `BenchConfig`, gating only the
   `target_ids`-uniqueness check in `__post_init__`. Default `True` keeps
   every existing call site (including the plain-constructor write-path
   test) unchanged.
2. `storage.load_bench` constructs with `strict=False` so a legacy
   duplicate is preserved rather than rejected.
3. `storage.save_bench` gets its own independent duplicate check (mirroring
   `create_run_group`'s existing one) so a leniently-loaded config can
   never be persisted with a duplicate still in it — defense in depth,
   since `save_bench` cannot assume the `BenchConfig` handed to it was
   built strictly.
4. Update the pre-existing `bench_with_duplicate_target_id` fixture in
   `Tests/UI/test_evals_bench_editor.py` to write directly against
   `EvalsDB.create_task`, since `BenchConfig(...)` (now correctly strict by
   default) can no longer be used to construct that fixture's legacy shape.
5. Add new tests: legacy load-and-preserve (storage-level), and write-path
   rejection for both plain `BenchConfig` construction and `save_bench`
   given a leniently-loaded config.
6. Revert-check: temporarily restore pre-fix `models.py`/`storage.py` and
   confirm exactly the targeted tests fail, with the expected error text.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `BenchConfig` gained a `strict: dataclasses.InitVar[bool] =
True` parameter. `InitVar` is accepted by the generated `__init__` and
passed to `__post_init__`, but is not a real dataclass field — it never
appears in equality, `repr`, or `asdict`, so it adds no persistent surface
to the type. `__post_init__` now only runs the `target_ids`-uniqueness
check `if strict`; the other invariants (`prompt_mode`, `top_k`,
`concurrency`) are unaffected and still always enforced — they predate #962
and are not a read-path concern. Every existing call site is unchanged
(default `True`); only `storage.load_bench` passes `strict=False`.

`storage.save_bench` gained its own explicit duplicate check (same shape as
`create_run_group`'s pre-existing one), because it cannot assume the
`BenchConfig` handed to it was built strictly — a caller could pass through
a leniently-loaded legacy config unmodified. This is defense in depth for a
path nothing currently exercises (no UI edit-and-resave flow exists yet in
this PR slice; `save_bench`'s only production caller,
`sample_bench.create_and_run_sample_bench`, always builds a single-target,
strictly-validated `BenchConfig`), but it is one of the write call sites
the task explicitly named, and closing it now means a future edit form
cannot silently persist a duplicate it inherited from a lenient read.

`create_run_group` and `WordBenchRunner.run` needed no change: the former
already validates its `targets` argument independently of `BenchConfig`
(documented in its own docstring as deliberate, since it is callable
directly, bypassing `BenchConfig`); the latter relies on that same check
before any cell capture begins.

**What the user sees.** Opening a legacy bench with a duplicate target id
now renders normally instead of an error placeholder: the target table and
readiness list both show the target twice (index-derived widget ids, from
the pre-existing Qodo #941 fix, already handle the duplicate without
colliding). Seeing the same target listed twice IS the visibility the user
needs to notice and remove it — no new UI affordance was added. The
`#evals-primary-action` "Run" button is unconditionally disabled in this PR
slice regardless of bench validity (`evals_screen.py`'s
`_primary_action_state`, "wiring lands in a later PR"), so there is no
separate run-blocking state to wire for a duplicate specifically yet; the
write-side guard (`create_run_group`) is what will stop an actual run once
that wiring lands.

**Revert-check.** Temporarily restored pre-fix `models.py`/`storage.py`
(new tests and the updated fixture left in place) and reran the target
scope: exactly 3 failures, nothing else.
- `test_bench_with_duplicate_target_id_composes_without_raising` →
  `AssertionError: both duplicate-id target rows should compose` (the
  `ValueError` from `load_bench` is caught by `bench_editor.py`/
  `inspector.py`'s own `except Exception`, so it surfaces as zero rendered
  target rows, not a raised exception in the test itself).
- `test_load_bench_tolerates_and_preserves_a_legacy_duplicate_target_id` →
  `ValueError: target_ids must be unique, got duplicates: [...]` raised
  directly out of `load_bench`.
- `test_save_bench_rejects_duplicates_even_for_a_leniently_loaded_config` →
  `TypeError: BenchConfig.__init__() got an unexpected keyword argument
  'strict'` (the pre-fix dataclass has no such parameter at all).
Restored the fix afterward and reconfirmed all 182 tests in scope pass.

**Files modified:**
- `tldw_chatbook/Evals/word_bench/models.py` — `BenchConfig.strict`
  `InitVar`, gated duplicate check.
- `tldw_chatbook/Evals/word_bench/storage.py` — `load_bench` constructs
  with `strict=False`; `save_bench` gained its own duplicate check.
- `Tests/UI/test_evals_bench_editor.py` — `bench_with_duplicate_target_id`
  fixture now writes directly via `EvalsDB.create_task`, bypassing
  `BenchConfig`.
- `Tests/Evals/word_bench/test_storage.py` — three new tests: legacy
  load-and-preserve, plain-construction still rejects, `save_bench` still
  rejects a leniently-loaded duplicate.

**Test result:** `pytest Tests/UI/test_evals_bench_editor.py
Tests/Evals/word_bench Tests/UI/test_evals_results_grid.py -q` → 182
passed (178 baseline + 4 new; the previously-erroring fixture now passes
instead of erroring).

**Follow-up (Qodo review, finding 2): `target_ids` element-type
validation.** The review characterized `strict=False` as bypassing "the
only validation that would catch malformed `target_ids`". That causation
is wrong: `strict` in `BenchConfig.__post_init__` has only ever gated the
*uniqueness* check above — `prompt_mode`, `top_k`, and `concurrency`
validate unconditionally, and no check has EVER validated `target_ids`'
element shape, on read or write, before or after this task's original fix.
This was a **pre-existing gap**, not something this PR introduced.

It was still worth closing here, because this PR's read-leniency means
`BenchConfig` now deliberately accepts more from stored data than it used
to: a corrupted `config_data.target_ids` entry (an int, a nested list, an
empty string) loaded without complaint and only failed much later inside
`db.get_model(target_id)` as an opaque sqlite parameter-binding error far
from the cause (`eval_models.id` is `TEXT`).

Added to `BenchConfig.__post_init__`, **ungated** (unlike the uniqueness
check, which stays `strict`-gated exactly as before): `target_ids` must be
a `list`/`tuple`, and every element must be a non-empty `str`, on every
construction path including `load_bench`'s lenient one. Placed before the
uniqueness check's `set(self.target_ids)` call specifically so an
unhashable element (e.g. a nested list) fails with this check's
diagnosable `ValueError` (naming the offending value and its type) rather
than an opaque `TypeError` out of `set()`.

Tests added: `test_bench_config_rejects_a_non_string_target_id`,
`test_bench_config_rejects_a_non_string_target_id_even_when_lenient` (the
key one — proves the check is NOT gated by `strict`), and
`test_bench_config_rejects_target_ids_that_is_not_a_list_or_tuple` in
`Tests/Evals/word_bench/test_models.py`; `test_load_bench_rejects_a_malformed_stored_target_id`
in `Tests/Evals/word_bench/test_storage.py` (storage-level: a corrupted
`eval_tasks.config_data.target_ids` fails at `load_bench` instead of
loading silently). The pre-existing legacy-duplicate test and the plain
round-trip test already covered "a legacy duplicate still loads" and "a
valid bench is unaffected" respectively, so no new test was needed for
those two.

Revert-check: reverted only `models.py` to its pre-follow-up state (new
tests left in place), reran the same scope — exactly 4 failures, all
`Failed: DID NOT RAISE <class 'ValueError'>`, nothing else. Restored the
fix and reconfirmed 150 passed. Fixed-code error text confirmed directly:
`target_ids elements must be non-empty strings, got 123 (type: int)` and
`target_ids must be a list or tuple, got 't1' (type: str)`.

Also added Google-style `Args:`/`Returns:` sections to the
`bench_with_duplicate_target_id` fixture and its consuming test in
`Tests/UI/test_evals_bench_editor.py`, and to the three tests in
`Tests/Evals/word_bench/test_storage.py`'s legacy-duplicate section
(~112-170), per a separate Qodo docstring finding — prose preserved,
structure added around it.

**Files modified (this follow-up):**
- `tldw_chatbook/Evals/word_bench/models.py` — ungated `target_ids`
  element-type check in `__post_init__`; class docstring updated.
- `Tests/Evals/word_bench/test_models.py` — 3 new tests.
- `Tests/Evals/word_bench/test_storage.py` — 1 new test; Google-style
  docstrings added to 3 existing tests.
- `Tests/UI/test_evals_bench_editor.py` — Google-style docstrings added to
  the `bench_with_duplicate_target_id` fixture and its consuming test.

**Test result (this follow-up):** `pytest
Tests/UI/test_evals_bench_editor.py Tests/Evals/word_bench -q` → 150
passed (146 baseline in this narrower scope + 4 new).
<!-- SECTION:NOTES:END -->
