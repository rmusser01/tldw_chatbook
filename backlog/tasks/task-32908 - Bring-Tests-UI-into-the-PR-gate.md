---
id: task-32908
title: Bring Tests/UI into the PR gate
status: Done
assignee:
  - '@claude'
labels:
  - ci
  - testing
created_date: '2026-09-22'
---

## Description (the why)

`Tests/UI` is the largest directory in the tree and gates nothing on a pull
request. It is widely believed to be covered — twice — and is covered neither
time, so defects that a UI test already names as broken merge anyway. The goal
is a PR gate that actually runs some of `Tests/UI` and can only grow, rather
than a comprehensive job that never reports.

## Acceptance Criteria (the what)

- [x] The real PR gate is identified and the coverage claim is verified against
      a real pull request, not against workflow files alone
- [x] `Tests/UI` pass / fail / error counts and wall-clock are measured, not
      estimated, with the ADR-126 share separated from genuine failures
- [x] Any test that hangs is named
- [x] A subset of `Tests/UI` runs on every pull request and its result is
      required for merge
- [x] The gated subset cannot silently shrink, and the guard that enforces this
      is demonstrated failing (negative control)
- [x] `Tests/UI/test_console_library_tool_setting.py` no longer contains an
      assertion that cannot pass
- [x] `./scripts/preflight.sh` exits 0

## Implementation Plan (the how)

1. Establish where the PR gate actually is (workflow triggers + a real PR's checks)
2. Measure the full `Tests/UI` suite with a per-test timeout
3. Test the config-load lead for the ADR-126 recovery gate and report what it recovers
4. Classify residual failures; name hangs
5. Choose a gate option on the measured numbers and implement it
6. Add a shrink-only guard for the gated set, with a negative control
7. Fix the stale assertion and whatever it was shadowing

## Implementation Notes

### The premise was wrong in a way that mattered

The investigation was framed around `test.yml:121` (`pytest Tests
--ignore=Tests/UI`). That line is irrelevant — it is just the split between
the core job and the sibling 12-shard `ui-tests` job in the same workflow. The
real reason is four lines higher: `test.yml` is `on: push: branches: ["main"]`
plus `workflow_dispatch`, so **it does not accept `pull_request` events at
all**. `gh pr checks` on PR #2795 confirms it — `PR Fast Lane`, `Derived
artifacts`, GGUF/platform evidence, and no `Tests` check of any kind.

The nightly "safety net" was worse than absent. `nightly-deep.yml` runs
`pytest ./Tests/` with no `--continue-on-collection-errors`, so one bad import
aborts everything: run 35706024071 logged `collected 101851 items / 1 error`
→ `Interrupted` → **zero tests executed**, six consecutive nights. Root cause:
`Tests/Chat/test_google_native_tools.py` imports `_google_tools_payload`,
which 0ea2906e99 deleted from `LLM_API_Calls.py` while leaving its call site at
line 3435 — a live `NameError` on the Google native-tools path, still unfixed
and filed separately.

### Measured, not estimated

Full `Tests/UI`, 6 parallel processes, `--timeout=60 --timeout-method=signal`:

| | |
|---|---|
| collected | 25,917 |
| executed and counted | 18,148 (70.0%) |
| passed | 10,858 (59.8%) |
| failed | 7,072 (39.0%) |
| errored | 214 (1.2%) |
| wall clock | >2 h on 6 parallel processes, 18-core M-series (shared) |

The failed share was sampled repeatedly as the run progressed — 40.3%, 40.4%,
40.9%, 40.6%, 39.9%, 39.5%, 39.2%, 39.0% — i.e. stable inside ±1 point over
70% of the suite, so **~40% of Tests/UI is red**. That alone rules out "full
green", before any budget argument. The run was stopped at 70% rather than
driven to completion: the ratio had been flat for hours, the box was shared
with other agents, and the remaining 30% could not have moved the gate
decision. Counts are read off the progress stream, not extrapolated. The dominant cause is the ADR-126 config-participant
admission, not genuine product failures: `Tests/UI/app_factory.py:112`
(`build_test_app_config` → `load_settings()`) fails
`RecoveryRequired("raw_source_selection_changed")` because the participant
binds to the bootstrap profile at conftest import and the autouse
`isolate_test_environment` fixture then redirects `TLDW_CONFIG_PATH` to a
per-test `tmp_path`. 481 of 1153 files use `_build_test_app`.

### Failure classification and hangs

Classified on a 65-file non-app_factory slice run with `--tb=line` (35 failed
/ 292 passed / 7 errors), exception mentions: **RecoveryRequired 58**,
AssertionError 18, AttributeError 10, TypeError 7, `Failed: Timeout` 4,
NoMatches 3. The ADR-126 admission dominates even in the *safest* slice; in
the app_factory half it is near-total (`test_console_library_tool_setting.py`
measured 14 of 14 failures as `raw_source_selection_changed`).

Four tests hang (>60 s, correlated from the timeout banners' position in the
progress stream against the collected order):

* `Tests/UI/test_pattern_gallery_snapshots.py::test_gallery_snapshot[textual-dark]`
* `Tests/UI/test_actor_pack_export_ownership.py::test_export_controller_is_in_app_owned_shutdown_before_profile_teardown`
* `Tests/UI/test_actor_pack_export_ownership.py::test_export_shutdown_finishes_before_later_app_owners`
* `Tests/UI/test_mcp_workbench_lifetime.py::test_poll_during_descendant_detach_is_safe_and_remount_recovers[during-canvas-workspace_root]`

Two of the four time out *inside `difflib`* — pytest building an assertion
diff over a large rendered snapshot, not the product hanging. `signal` is the
right `--timeout-method` here (the repo's ini leaves it unset for exactly this
reason, TASK-22062): `thread` would destroy the process and name nothing. None
of these files is in the census.

### The config-load lead does NOT generalise to Tests/UI

It is real, and it is a different bug. `Tests/Media/test_local_media_reading_
service.py` went **72 failed → 6 failed (+66 recovered)** by importing
`tldw_chatbook.Chunking` first. The mechanism is not "a migration loads
config": it is that `Chunking/Chunk_Lib.py:301` snapshots
`summarize_system_prompt` at **module import time** (deliberately — pinned by
`Tests/Internal_Prompts/test_summarization_migration.py`), and
`_apply_migration_v6_to_v7` imports that module **lazily**, so the first test
to build a media DB pays the import inside an open participant scope.

Two corrections worth recording:

* Doing the import from a wrapper *before* pytest starts (the form the lead was
  written in) binds the participant to the developer's **real**
  `~/.config/tldw_cli/config.toml` and then fails at collection. It has to live
  inside `Tests/conftest.py`, after the bootstrap env block.
* It recovers **~0 tests in Tests/UI** — measured, same 14 failures before and
  after. Tests/UI's gate failure is the app_factory/`tmp_path` rebinding above,
  a different mechanism with the same exception name.

The fix is kept anyway (one import, ~0.26 s/process, +66 tests in Tests/Media,
+1 in Tests/Internal_Prompts, zero regressions).

### Option chosen: (c) sharded/subset, with a shrink-only census

(a) full green is impossible at ~40% red. (b) a ratchet over the whole suite
is impossible on budget — a ratchet still has to *run* the suite, and the PR
gate is `derived-artifacts.yml`, whose `PR Fast Lane` measured 15m54s against a
30-minute cap. 5.5 CPU-hours does not fit in the remainder.

So: a new `ui-fast-lane` job runs a **verified-green subset** —
`scripts/ui_pr_gate_census.txt`, **120 files / 851 tests, 3m48s serial**. It is
its own job (not another step on `pr-fast-lane`) so it cannot eat that lane's
budget, and `derived-artifacts` gains `needs: [pr-fast-lane, ui-fast-lane]`
plus a second verdict step, keeping ONE branch-protection context.

Deliberately serial, on the same minimal dependency set `pr-fast-lane`
installs, in a pinned file order — because that is the configuration the census
was verified in, and `requirements-test.txt` has no `pytest-randomly`, so order
is reproducible. A censused file run **alone** can still error on the ADR-126
admission; it is green in census order, which is what CI runs.

`--continue-on-collection-errors` was added to `nightly-deep.yml` in the same
change: "keep the rest nightly" is only honest if the nightly runs at all.

### The stale assertion, and what it was shadowing

`test_console_library_tool_setting.py:229` asserted `service._collections is
app.local_library_collections_service`. 5dd1077df6 ("retire generic containers
from current surfaces") removed both `collections_service` from
`LocalLibraryToolService.__init__` and `self._collections`; the only occurrence
left in that module is inside a comment. Assertion removed, along with its now-
dead setup line.

It was shadowing a real defect. There are **two** copies of that factory.
`UI/Console_Modules/library_activity.py:145` (the one the test exercises) was
updated correctly. `Chat/console_runtime.py:681`
(`_library_provider_for_app`, used as the controller's
`library_provider_factory`, no test coverage at all) still passed
`collections_service=` — `TypeError: __init__() got an unexpected keyword
argument`, on every Console-direct Library tool build, since 2026-09-01. Fixed.

### Evidence

* Census green: 851 passed, 0 failed, 228.26 s, exit 0 (serial, census order,
  minimal venv).
* No regression to the existing gate: the current `PR Fast Lane` suites run
  1149 + 123 passed under this branch's `Tests/conftest.py`; the only 3
  failures were the CI shape tests this change intentionally edits, now green
  (`Tests/CI`: 353 passed).
* Negative control 1 (census shrinkage): deleting one line →
  `check_ui_pr_gate_census.py` exits 1, "census has shrunk: 119 files, floor is
  120". Restored → exits 0.
* Negative control 2 (gate catches a product regression), and it took two
  attempts — recorded because the first one is the trap this task is about.
  Mutating `_ARGS_SUMMARY_LIMIT` 80 → 200 in
  `Widgets/Chat_Widgets/chat_approval_card.py` and running the full census gave
  **851 passed, exit 0**, which reads as "the gate does not work". It was not:
  that render is 46 characters, bounded by a *second* constant
  (`_ARGS_VALUE_LIMIT = 34`), so the mutation changed nothing observable and
  there was no regression to catch. With BOTH constants moved
  (`len(rendered)` 46 → 400, verified before running), the census run goes red:
  **1 failed, 850 passed, exit 1**, naming
  `test_raw_shell_dedicated_view_does_not_expand_the_generic_summary_budget`.
  Reverted; `git status` clean for that file. Lesson filed.

### Modified / added

* `.github/workflows/derived-artifacts.yml` — new `ui-fast-lane` job, second
  verdict step, census checker step
* `.github/workflows/nightly-deep.yml` — `--continue-on-collection-errors`
* `scripts/check_ui_pr_gate_census.py`, `scripts/ui_pr_gate_census.txt` (new)
* `scripts/preflight.sh` — census checker
* `Tests/CI/test_derived_artifacts_workflow.py`,
  `Tests/CI/test_ci_queue_pressure_contract.py` — pin the new job's shape
* `Tests/conftest.py` — eager `tldw_chatbook.Chunking` import
* `Tests/UI/test_console_library_tool_setting.py` — stale assertion removed
* `tldw_chatbook/Chat/console_runtime.py` — `collections_service=` TypeError
* `backlog/docs/lessons-testing-evidence.md` — new lesson

### Left open (filed, not fixed here)

* `_google_tools_payload` is called at `LLM_API_Calls.py:3435` and defined
  nowhere — live `NameError`, and the reason the nightly executed nothing.
* The ADR-126 `app_factory` admission failure is ~40% of Tests/UI. Closing it
  is what grows this census past 120 files, and it is a design question about
  `keep_bootstrap_profile` rather than a per-file allowlist edit.
