# CI throughput, sub-project 1: merge-conflict hotspots and wasted runner work

**Date:** 2026-09-27
**Status:** Draft, awaiting owner review
**Scope:** `rmusser01/tldw_chatbook` only (sub-project 1 of 5; see "Program context")

## Why

The owner's per-PR procedure is: rebase on the latest `dev`, address Qodo's review, merge.
With `dev` absorbing 23-50 merges a day, every merge makes the other open PRs stale, and
every rebase is a new head that goes to the back of the CI queue. Two things turn that
treadmill into a stall, and this sub-project removes as much of both as it can:

1. **Rebases that conflict.** A PR in conflict gets **no CI runs at all** (DIRTY), and the
   resolution is manual.
2. **Runner work that produces no signal**, competing for the account's shared runner pool.

It also removes the one flaky test group that fails the required check on unrelated PRs.

## Evidence (measured 2026-09-27)

All numbers come from real history on `origin/dev` and the Actions API for 09-19..09-26.

**Conflicts.** We replayed the 547 real `dev -> PR branch` sync merges found in the last 300
merged PRs with `git merge-tree`. **263 (48%) conflicted.** The files involved:

| File | Syncs conflicting |
|---|---|
| `Docs/security/production-diagnostic-inventory.json` | 102 |
| `Docs/User_Guide/library/notes.md` | 74 |
| `backlog/docs/lessons-testing-evidence.md` | 43 |
| Other `Docs/User_Guide/**/*.md` | 47 |
| `backlog/docs/lessons-live-verification.md` | 12 |

**186 of the 263 (71%)** conflicted only on these three kinds of file. In those syncs, every
other file merged cleanly.

**The inventory.** 127 real syncs had both sides change the inventory. Replaying them with the
top-level `summary` block stripped from base, ours and theirs:

| Outcome | With `summary` (today) | Without `summary` |
|---|---|---|
| Conflict | 102 | 17 |
| Clean, but differs from the committed resolution | n/a | 7 |

The remaining 17 are real overlaps: both sides changed diagnostics in the same source file,
and a human should look at those. The 7 are caught by the required check (below).

The six `summary` totals are the cause: any two PRs that touch any logger call both edit
the same lines. Inventory drift also caused 41 of the required check's 84 failures, across 21
branches (including 4 pushes to `dev`).

**The User Guide.** The conflicts are "Verified against ..." paragraphs. `CLAUDE.md:143-144`
asks UI PRs to update the page "or at least its 'Verified against' stamp", and parallel PRs
append those paragraphs at the same spot on the same page. Many are now multi-paragraph
change logs.

**Runner waste.** Over 8 days, about 22.5k runner-minutes (about 2.8k/day):

- **GGUF evidence workflows (TASK-2062.1/.2):** 16% of all runner-minutes and **81% of all
  Windows minutes**. Each triggering PR starts 3 OS jobs. Both tasks have been Done since
  2026-08-13. The workflows fire on broad paths: `pyproject.toml`, `app.py`, `config.py`,
  `css/**`, `Tests/conftest.py`, `Tests/UI/conftest.py`, `Tests/UI/app_factory.py`.
- **Duplicated guards:** `css-bundle-guard.yml` (612 runs, 0 failures) and `backlog-guard.yml`
  (366 runs) run checks that `derived-artifacts.yml` already runs in the required job, on every
  PR and every push to `dev`/`main` (`check_bundle_sync`, `check_backlog_task_ids`;
  `derived-artifacts.yml:255` says so). Each still takes a runner slot per PR.
- **Evidence-only runs:** 883 skipped-at-job-level runs a week come from `task-19637` (462) and
  `task-32011` (421), both triggered on every `synchronize`.

**Flakes.** Four tests in `Tests/UI/test_mcp_workbench.py` failed across 16 of the 37 failed
`PR Fast Lane` jobs, on unrelated branches, and pass on rerun and locally on `dev`.
TASK-32049 already records the root cause of one of them: a TextArea `COMPONENT_CLASSES`
registration race.

## Goals

- Cut sync-merge conflicts on these hotspots at the source, while keeping the privacy review the
  inventory exists for (ADR-029).
- Remove runner jobs per PR that produce no signal.
- Stop the flaky group from failing the required check on unrelated PRs, without losing it.

## Non-goals

This sub-project does not touch:

- The required check's name, `needs` wiring, dependency boundary or cadence (ADR-103).
- `tldw_server`'s workflows (sub-project 2).
- The nightly (sub-project 3).
- Widening the PR gate (sub-project 4).
- Existing "Verified against" paragraphs already in the User Guide. They stay; only new ones
  stop. Deleting them now would conflict with every open PR that appends one.
- The generated CSS bundle: it did not appear among conflict hotspots.

## Design

### A. Diagnostic inventory: stop committing the `summary` totals

`scripts/check_persistent_diagnostic_inventory.py` changes as follows:

- **`--write`** no longer emits the top-level `summary` object. `schema_version` goes from 3
  to 4.
- **Totals are derived, not stored.** The six totals (`owner_files`,
  `path_privacy_candidate_calls`, `persistent_sink_files`, `task_31551_calls`,
  `task_492_calls`, `task_494_calls`) are computed from the rows whenever they are needed:
  - for the drift report (`_summary_lines`, around line 1241), which now compares totals
    derived from the committed rows against totals derived from the rebuilt rows;
  - for the stdout lines around 1773 and 1788.
- **Old-format files are rejected.** Check mode fails a committed file that still has
  `schema_version` 3 or a `summary` key, and names the fix:
  `python scripts/check_persistent_diagnostic_inventory.py --write`.
- **Everything else is unchanged:** every row, the rules, the ordering, and the requirement
  that a PR's diff shows each added, removed or changed diagnostic for review.

Tests to update:

- `Tests/Architecture/test_derived_artifact_checkers.py:77` edits `rebuilt["summary"]`. It
  becomes a row edit that shifts a derived total.
- `Tests/Architecture/test_diagnostic_path_privacy.py:1004` builds a `summary` fixture. It
  becomes a derived-total assertion.

New test: the drift report still shows total deltas, and a schema-3 file fails with the
regenerate instruction.

**Why the 7 clean-but-different merges stay safe.** The required check runs the inventory
checker on every PR head (the merged ref) and on every push to `dev`. A clean merge whose
rows don't match the merged source fails the check, and `scripts/preflight.sh` catches the
same thing locally. Today those cases surface as a conflict; after this change they surface
as a failing required check, and the fix is the same regeneration.

**Rollout.** Every open PR that touches the inventory will conflict or fail once after this
lands, and needs one `--write` on its merged head. That is the same work a single summary
conflict costs today.

**ADR.** Append a dated amendment to `backlog/decisions/029-local-private-data-boundary.md`:
the inventory stores per-file rows only; totals are derived at check time; this was done
because of merge-conflict churn (the evidence above). The review guarantee is unchanged.

**Alternatives rejected.**

- *One file per source file* (about 600 files). The same 17 real overlaps still conflict, so
  it adds nothing over removing `summary`, at a much larger diff and tooling cost.
- *Don't commit the inventory; derive it in CI.* This removes the reviewed diff ADR-029
  depends on. Lesson TASK-14651 records how a generator-resolved conflict silently blessed 16
  privacy-relevant diagnostics.

### B. Append-only prose

**B1. Lessons files.** Add a root `.gitattributes` entry:

```
backlog/docs/lessons-*.md merge=union
```

With this, local merges and rebases keep both sides' appended entries without a conflict.
Lessons are append-only by convention, which is the case `union` is meant for.

Known limit: GitHub's server-side mergeability check is expected to ignore merge attributes.
A PR may therefore still show "conflicts" on GitHub until it is rebased or merged locally,
which the owner's procedure does anyway. Implementation verifies two things and records the
result as a lesson:

1. A local `git rebase` with two appended entries resolves cleanly.
2. How GitHub reports such a PR, using a throwaway branch, only with the owner's OK because it
   is outward-facing.

**B2. User Guide verification stamps.** Change the `CLAUDE.md` rule at `CLAUDE.md:143-144` to:

> **UI changes:** PRs that change a screen's UI update the matching `Docs/User_Guide/` page's
> content where behaviour changed. Record what was verified, and against which branch and
> date, in the task's Implementation Notes, not in the User Guide page. Do not append
> "Verified against" paragraphs to User Guide pages.

Only `CLAUDE.md` carries this rule on `origin/dev` (checked 2026-09-27), so no other
instruction file changes.

### C. Runner work that produces no signal

| Workflow | Change | Why |
|---|---|---|
| `css-bundle-guard.yml` | Delete | `derived-artifacts.yml` runs `check_bundle_sync` in the required job on every PR and every push to `dev`/`main`. |
| `backlog-guard.yml` | Delete | Same for `check_backlog_task_ids` (`derived-artifacts.yml:255`). |
| `task-2062-1-gguf-import-evidence.yml` | Narrow `paths` to its own file, `tldw_chatbook/Model_Artifacts/**`, `tldw_chatbook/UI/Screens/model_installed_view.py`, `Tests/Model_Artifacts/**`, `Tests/UI/test_model_installed_view.py`. Drop `pyproject.toml`, `app.py`, `css/**`, `Tests/conftest.py`, `Tests/UI/conftest.py`, `Tests/UI/consolidated_css.py`. Keep `workflow_dispatch`. | Keeps the three-OS GGUF guard where GGUF code changes, and stops it firing on unrelated app-wide edits. Not a required check, so a path filter is safe. |
| `task-2062-2-gguf-source-evidence.yml` | Narrow `paths` to its own file, `Model_Artifacts/**`, `Event_Handlers/LLM_Management_Events/**`, `UI/LLM_Management_Window.py`, `UI/Screens/llm_screen.py`, `Tests/LLM_Management/**`, `Tests/Model_Artifacts/**`, `Tests/UI/test_llm_gguf_source_modes.py`. Drop `pyproject.toml`, `app.py`, `config.py`, `Tests/conftest.py`, `Tests/private_profile.py`, `Tests/UI/app_factory.py`, `Tests/UI/conftest.py`. Keep `workflow_dispatch`. | Same reasoning. |
| `task-32011-linux-storage-evidence.yml` | Replace the `pull_request` trigger with `workflow_dispatch` | TASK-32011 is Done. This stops a skipped run on every push. |
| `task-19637-platform-evidence.yml` | No change | TASK-19637 is In Progress. Its label-gated re-run on `synchronize` is still wanted, and its skipped runs cost no runner time. |
| `nightly-deep.yml` | **Pause:** remove the `schedule:` trigger and keep `workflow_dispatch`. Add a header comment naming this spec and the condition for restoring the schedule (phase 3: the run can finish and report). | Owner decision, 2026-09-27. It uses about 22% of all runner-minutes and 64% of macOS minutes, and produced no complete result in 8/8 nights: a serial run reaches about 11% in 240 min. It can still be run by hand. |

Also update the comments that name the deleted guards: `derived-artifacts.yml:26` and `:255`,
and `perf-guard.yml:9` and `:43`. No test pins the guard or evidence workflows (grep of
`origin/dev`, 2026-09-27).

**Pausing the nightly: where it lands and which tests change.**

- **Two branches.** Schedules register from the default branch, and the file's header
  requires it to stay identical on `dev` and `main`. So the change lands on `dev` (this
  sub-project's PR) and on `main`.
- **The `main` change ships `dev`'s entire paused file.** `main`'s copy is 25 lines behind
  `dev`'s, which is what open PR #2819 fixes, so this supersedes #2819. Merging to `main` is
  the owner's call.
- **Contract tests.** Two tests pin the cron:
  - `Tests/CI/test_ci_queue_pressure_contract.py:153-154` (`triggers == {"schedule",
    "workflow_dispatch"}`, cron `30 8 * * *`);
  - `Tests/CI/test_github_actions_test_workflow.py:391`.

  They change to assert the paused shape (dispatch only), with a comment pointing at phase 3,
  which restores both the schedule and these assertions.

### D. Quarantine the flaky MCP workbench tests from the fast lane

In `derived-artifacts.yml`'s "Run fast PR contract" step, add one `--deselect` per test, with
a comment naming TASK-32049:

```
--deselect "Tests/UI/test_mcp_workbench.py::test_render_failure_in_show_tool_test_result_notifies_instead_of_only_logging"
--deselect "Tests/UI/test_mcp_workbench.py::test_test_tool_active_watcher_polling_is_bounded_and_stops_on_unmount"
--deselect "Tests/UI/test_mcp_workbench.py::test_test_tool_active_watcher_never_updates_stale_panel"
--deselect "Tests/UI/test_mcp_workbench.py::test_set_initial_view_state_during_inflight_reload_applies_pending_state_once"
```

`--deselect` matches node-id prefixes, so the parametrized test is covered by its base id.
Implementation confirms each id against `origin/dev`, and confirms that each id deselects the
intended node and nothing else (compare `--collect-only` counts before and after).

- **Contract test.** `Tests/CI/test_ci_queue_pressure_contract.py` pins the fast-lane targets
  (`FAST_LANE_TARGETS`). Extend it to pin the quarantine set too, so an entry can't be added
  silently.
- **Check whether the contract test runs in CI at all.** It does
  `pytest.importorskip("yaml")`, and the fast lane installs only
  `-e . pytest pytest-asyncio pytest-timeout packaging`. Implementation checks whether `yaml`
  is present in that environment. If it isn't, the contract test is silently skipped in CI;
  report that as a finding, not a fix in scope.
- **The tests are not dropped.** They still run in `test.yml` and the nightly. TASK-32049
  (extended to cover all four tests) owns the root-cause fix and removing the `--deselect`
  lines.

## Verification

Each item needs evidence, not assertion:

- **A:**
  - The sync-merge replay, re-run on schema-4 files, reproduces "102 -> about 17". The replay
    is committed as a small stdlib script, `scripts/measure_sync_merge_conflicts.py`, so the
    2-week success review re-runs the same method.
  - The checker round-trips: `--write` then check passes. A deliberate row edit fails with
    total deltas shown.
  - A schema-3 file fails with the regenerate instruction.
  - `./scripts/preflight.sh` rc=0.
- **B1:** a local two-branch rebase test resolves both appended lessons entries without a
  conflict, and a negative control without the attribute conflicts.
- **C:** `gh pr checks` on the implementation PR shows no CSS/Backlog Guard runs. The task
  notes carry a table of sample changed-path sets (`app.py` only; `css/**` only;
  `Tests/conftest.py` only; `Model_Artifacts/**`; `pyproject.toml` only) against each edited
  workflow's new `paths`. It shows GGUF evidence fires only for the GGUF rows.
- **D:** `--collect-only` shows exactly the four ids removed from the fast lane, and the
  contract test fails if an entry is added without updating it.
- **Full:** preflight rc=0; the fast-lane files pass locally on the minimal dependency set.

## Success measures (review about 2 weeks after merge)

| Measure | Baseline | Target |
|---|---|---|
| Share of `dev -> branch` sync merges with any conflict (same replay method) | 48% | ≤ 20% |
| Inventory share of required-check failures | 41 of 84 | ≤ 10% |
| Runner-minutes per day | ~2.8k | Down by ≥ 35% (GGUF about 16% plus the nightly about 22%) |
| macOS minutes per day | ~290 | Down by ≥ 60% |
| Windows minutes | – | Down by ≥ 70% |
| Runs per PR head SHA | 5.47 | ≤ 4 |
| Spurious `PR Fast Lane` failures from the MCP group | 16 of 37 failed jobs | 0 |

## Risks

- **A clean-but-wrong inventory merge lands on `dev`.** This happens only if a PR merges
  without its required check running on the merged ref. Branch protection prevents that, and
  the push-to-`dev` run catches it after the fact.
- **`merge=union` duplicates a line.** This happens if someone edits an existing lessons entry
  on two branches. That is rare for append-only files and visible in review.
- **A quarantine outlives its fix.** Mitigated by pinning the set in the contract test and
  tracking removal in TASK-32049.

## Program context

**Owner priority order (2026-09-27):**

1. CI throughput: sub-projects 1, 2 and 5.
2. Clear the open-PR backlog: 34 open PRs on 2026-09-27; 30 to `dev`, 5 drafts, 4
   conflicting, 10 idle for more than 14 days.
3. Tests and issues: sub-projects 3 and 4, plus TASK-32049.

Sub-project 4 widens the PR gate, which adds CI work per PR, so it deliberately waits for
phase 3.

The nightly's red is mostly one harness cause, not product bugs. A local re-run of its three
worst files on `origin/dev`, 2026-09-27:

- `Tests/Agents/test_local_tool_provider.py`: 248 of 263 fail, all with
  `RecoveryRequired: raw_source_selection_changed` from the backup/recovery gate.
- `Tests/Agents/test_fleet_runtime.py` and `Tests/Agents/test_agent_service.py`: 146 and 79
  fail, matching the nightly's counts. They are dominated by the same gate
  (`raw_source_selection_changed`, `raw_participant_not_installed`,
  `config_source_not_installed`), surfacing directly or as agent runs that end in `error`.
- A small remainder looks like real test drift.

The sub-projects not designed here:

2. **Account-wide runner queue.** `tldw_server` creates 3-6× this repo's runs, mostly from
   `workflow_run` fan-out. At 2026-09-27 04:00Z it had 366 runs queued and 0 in progress; this
   repo's median push-to-verdict was 425-745 min on 09-24..26. Survey in progress.
3. **A nightly that produces a result.** `Nightly Deep` produced no complete result in 8/8
   nights:
   - Serially it reaches about 11% in 240 min.
   - Windows fails with a cp1252 collection error and an `os.write` hang.
   - It shows about 2,050 deterministic failures in Agents, Backup_Recovery and Audio.
   - It burns 22% of runner-minutes and 64% of macOS minutes.
4. **A wider PR gate.** Change-aware selection inside the one required context. This amends
   ADR-103, which rejected path-based selection because `Docs/` and `backlog/` files are test
   inputs, so any selector must treat them conservatively.
5. **Fewer cycles per PR.** Batch the Qodo fixes into the rebase push; optionally skip CI on
   draft PRs.
