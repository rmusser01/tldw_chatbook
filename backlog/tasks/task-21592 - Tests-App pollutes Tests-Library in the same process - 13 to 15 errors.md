---
id: TASK-21592
title: >-
  Tests App pollutes Tests Library in the same process   13 to 15 errors
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - flaky
  - test-isolation
priority: medium
---
## Description

Running `Tests/App` and `Tests/Library` in the same process produces 13-15 errors whose node ids
vary run to run, and zero errors when either directory runs alone. This is cross-file state
leakage, and it is a real source of CI flakiness that will misattribute failures to whichever
branch happens to be running.

## Acceptance Criteria

- [x] The leaking state is identified by name (module-level singleton, patched global, app instance, or fixture scope) rather than worked around with ordering or `-p no:randomly`
- [x] `Tests/App` and `Tests/Library` run clean together in one process, repeatedly and under random ordering
- [x] A guard prevents the same class of leak from returning — e.g. an autouse fixture asserting the relevant global is unset at teardown
- [x] The error count is confirmed to be zero on several consecutive runs, since the symptom is nondeterministic

## Evidence

Observed by the TASK-21111 implementer while A/B-baselining: 13 errors on its branch and 15 of
the same class on pristine dev `f49956038`, with varying node ids; running its new file alone with
those Library files gave 306 passed and 0 errors.

Pre-existing and unrelated to that task's change.

## Implementation Plan

1. Recount on the working base before assuming the filing still holds.
2. If it does not reproduce, reproduce it on the base it was filed against and
   find what closed it.
3. Read the errors themselves rather than their node ids, and name the state.
4. Bisect the two directories down to the smallest reproduction available.
5. Add the guard that is still missing, and mutation-verify it.
6. Confirm zero errors over several runs and under a shuffled order.

## Implementation Notes

### Recount: the filed symptom is already closed on this base

| Run | `Tests/App` + `Tests/Library` | `Tests/Library` alone |
| --- | --- | --- |
| dev `f49956038` (the filing's base) | 2581 passed, **10 errors** | 2403 passed, **0 errors** |
| dev `a71e62e4b` (this base), App first | 2638 passed, 1 failed, **0 errors** | 2448 passed, 1 failed, 0 errors |
| dev `a71e62e4b`, Library first | 2638 passed, 1 failed, **0 errors** | — |

The cross-directory premise was real on `f49956038` and I reproduced it. The one
failure on this base (`test_outcome_first_recipe_has_stable_blank_markdown_blocks_in_both_lanes`)
also fails with `Tests/Library` alone, so it is not an isolation effect.

It closed because **TASK-21562** (`f9128f9bc`) and **TASK-21562.1**
(`d7e218276`) landed between the two bases. They set `HF_HUB_OFFLINE=1` in the
pre-import bootstrap and patch `huggingface_hub.constants.HF_HUB_OFFLINE` from
an autouse fixture. Every one of the ten errors was a live hub fetch.

### The leaking state, by name

Every error is a **teardown** error raised by the autouse `_no_network_io`
fixture, and every one of them reads
`socket.create_connection -> huggingface.co:443`, with
`huggingface_hub` logging `Retrying in 1s [Retry 1/5]` for
`sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json`.

Two globals, in series:

1. **`huggingface_hub.constants.HF_HUB_OFFLINE`** — frozen at that module's
   import time. Nothing set it on `f49956038`, so a Library test that built the
   default `all-MiniLM-L6-v2` embedding model against an empty cache directory
   reached the real hub. This is the *enabling* condition, and it is the one
   TASK-21562 closed.
2. **`Tests/network_guard._blocked_attempts`** — a module-level, process-global
   list, drained and asserted empty at **every** test's teardown. `huggingface_hub`
   retries a blocked request five times with 1/2/4/8/16s backoff on a worker
   thread that outlives the test that started it, so attempts made by test A are
   drained at the teardown of tests B, C, D… This is the *misattribution*
   mechanism: it is why the count is 10-15 rather than 1, why the node ids vary
   run to run, and why every erroring node had itself passed. It is untouched by
   TASK-21562 and is what this task adds a guard for.

Running either directory alone leaves too short a tail of subsequent tests for
the retries to land on — that, not the absence of the fetch, is why the alone
runs report zero.

### Bisection

On `f49956038`, against four Library files
(`export_roundtrip`, `export_scope`, `export_state`, `rag_scope`, 87 tests):

| Combination | Errors |
| --- | --- |
| the four Library files alone | 0 |
| + each of the other eight `Tests/App` files | 0 |
| + `Tests/App/test_app_shutdown.py` | 6 |
| + only `test_app_shutdown.py::test_mounting_the_real_app_under_test_arms_no_watchdog` | 3, 7, 3 (three runs) |

That test is the only one in `Tests/App` that mounts the real `TldwCli`, and it
does so with no config/HOME isolation fixture of its own.

**Recorded rather than glossed:** I could not name what that mount leaves
behind. The shared sandbox `config.toml` is byte-identical after it (diffed).
No single Library file reproduces with it — the reduced repro needs the whole
group, consistent with a retry tail that has to outlive several tests. And
loading a no-op diagnostic plugin (a `pytest_runtest_protocol` wrapper plus a
passthrough wrapper on `_deny`) suppressed the failure on three consecutive
runs, which is a timing window, not a fix. The enabling condition is closed, so
I stopped there rather than keep guessing at the trigger.

### Guard added

`Tests/network_guard.py` now records the **thread name** behind each blocked
attempt (`_blocked_attempt_threads`, a parallel list so the published
`(call, address)` shape stays unpacking-compatible for the seven files that
already consume it), and `describe_blocked_attempts()` builds the failure
message. When an attempt came from anything other than the main thread the
message says so explicitly, so a bystander failure names its likely origin
instead of reading as "this test did network I/O".

`Tests/conftest.py`'s `_huggingface_hub_is_offline` became a yield fixture that
**re-asserts the offline latch at teardown**. That latch is the single condition
standing between the suite and this entire failure class; a test that turned it
back off would otherwise re-arm it silently for everything after it. The check
runs at teardown regardless of whether the module was loaded at setup — the
first version returned early when `sys.modules` had no hub yet, and a probe test
that imported the hub and set the constant to `False` passed clean. Fixed, then
re-probed: it now errors with the intended message.

### Mutation results

| Mutation | Result |
| --- | --- |
| `_deny` stops recording the thread name | 2 failed |
| `describe_blocked_attempts` never emits the provenance note | 1 failed |
| `drain_blocked_attempts` stops clearing the thread record | 1 failed |
| a test sets `constants.HF_HUB_OFFLINE = False` | 1 error, correct message |

### Counts

`Tests/test_network_guard.py` 19 passed / 1 skipped → 23 passed / 1 skipped
(four new). `Tests/test_huggingface_offline.py` + `Tests/test_network_guard.py`:
26 passed, 1 skipped.

Four consecutive `Tests/App` + `Tests/Library` runs, three orderings, **zero
errors in every one**:

| Run | Tree | Order | Result |
| --- | --- | --- | --- |
| baseline | pristine `a71e62e4b` | App, Library | 2638 passed, 1 failed, 2 skipped, 0 errors |
| baseline | pristine `a71e62e4b` | Library, App | 2638 passed, 1 failed, 2 skipped, 0 errors |
| A | this branch | App, Library | 2638 passed, 1 failed, 2 skipped, 0 errors |
| B | this branch | Library, App | 2638 passed, 1 failed, 2 skipped, 0 errors |
| C | this branch | 80 files shuffled, seed 21592 | 2638 passed, 1 failed, 2 skipped, 0 errors |

The single failure is identical on pristine dev and appears with `Tests/Library`
alone, so it is not an isolation effect.

`pytest-randomly` is **not installed** in this venv (`pytest-mock`, `-asyncio`,
`-shard`, `-xdist`, `-json-report`, `-timeout`, `-metadata` are), so
"random ordering" was done by shuffling the 80 test files with a recorded seed
(21592) rather than by a plugin. Worth knowing: `-p no:randomly` is a no-op here
and proves nothing about ordering.

### Files

`Tests/network_guard.py`, `Tests/conftest.py`, `Tests/test_network_guard.py`.
No production changes.
