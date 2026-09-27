# CI throughput, sub-project 1: merge-conflict hotspots and wasted runner work

**Date:** 2026-09-27
**Status:** Draft, revision 2 (after an independent adversarial review the same day). Awaiting
owner review.
**Scope:** `rmusser01/tldw_chatbook` only. This is sub-project 1 of 5; see "Program context".

## Why

The owner's procedure for each PR is: rebase on the latest `dev`, address Qodo's review, merge.
`dev` absorbs 23-50 merges a day, and since 2026-09-27 it is **strict** again: a PR must be up to
date to merge. So every merge makes each other ready PR re-sync, and every re-sync is a new head
that needs a fresh CI verdict. Two things turn that into a stall:

1. **Re-syncs that conflict.** A conflicting (DIRTY) PR gets **no CI runs at all**, and resolving
   the conflict is manual.
2. **Runner work that produces no signal**, competing for the account's shared pool of 40 jobs
   (5 macOS).

A third problem, a flaky test in the required fast lane, costs whole CI cycles on unrelated PRs.

## Evidence (measured 2026-09-27)

These numbers come from real history on `origin/dev` and from the Actions API for 09-19..26.

**Conflicts.** We replayed the 547 real `dev -> PR branch` sync merges from the last 300 merged
PRs with `git merge-tree`. **263 (48%) conflicted.**

| File | Syncs conflicting |
|---|---|
| `Docs/security/production-diagnostic-inventory.json` | 102 |
| `Docs/User_Guide/library/notes.md` | 74 |
| `backlog/docs/lessons-testing-evidence.md` | 43 |
| other `Docs/User_Guide/**/*.md` | 47 |
| `backlog/docs/lessons-live-verification.md` | 12 |

**The inventory.** 127 real syncs had both sides change the inventory. We replayed them with the
top-level `summary` block stripped from base, ours and theirs:

| | With `summary` (today) | Without `summary` |
|---|---|---|
| Conflicts | 102 | 17 |
| Clean, but different from the committed resolution | – | 7 |

- **The 17** are real overlaps: both sides changed diagnostics in the same source file.
- **The 7** fail the required check and preflight, which run on the merged head.
- **Why `summary` causes it:** its six totals change whenever any logger call changes, so two
  PRs that each touch any logger call both edit the same lines.
- Inventory drift also caused 41 of the required check's 84 failures, across 21 branches.

**The User Guide.** The conflicting hunks are "Verified against ..." paragraphs. `CLAUDE.md`
("UI changes") asks UI PRs to update a page "or at least its 'Verified against' stamp", so
parallel PRs add their paragraphs at the same spot on the same page. Many of these paragraphs
are now multi-paragraph change logs.

**Runner waste.** About 22.5k runner-minutes over 8 days, roughly 2.8k per day.

- **GGUF evidence (TASK-2062.1/.2):** 16% of all minutes and **81% of Windows minutes**. Each
  triggering PR starts 3 OS jobs. Both tasks have been Done since 2026-08-13. The workflows fire
  on `pyproject.toml`, `app.py`, `config.py`, `css/**`, `Tests/conftest.py`,
  `Tests/UI/conftest.py` and `Tests/UI/app_factory.py`.
- **`Nightly Deep`:** 22% of all minutes and 64% of macOS minutes, with **0 complete results in 8
  nights**. Run serially it reaches about 11% of the suite in 240 min.
- **`css-bundle-guard.yml` and `backlog-guard.yml`:** they re-run `check_bundle_sync` and
  `check_backlog_task_ids`. `derived-artifacts.yml` already runs both in the required job, on
  every PR and every push to `dev`/`main`. The guards are path-filtered, but together they
  started 978 runs in 8 days, each taking a runner slot.

**Flake.** The error `KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"` in
`Tests/UI/test_mcp_workbench.py` fails the required fast lane on unrelated PRs:

- 16 of 37 failed fast-lane jobs had it, across 4 or more distinct tests.
- TASK-32049 recorded it first, in `test_test_tool_preview_*`, as a TextArea component-class
  registration race.
- It hit #2816 and #2815 again on 2026-09-26/27.

## Goals

- Stop re-syncs from conflicting where the conflicts are artificial. Keep the privacy review the
  inventory exists for (ADR-029).
- Remove runner work that produces no signal.
- Remove the fast-lane flake at its root cause, without weakening the gate's
  no-selection-suppression contract.

## Non-goals

- The required check's name and `needs` wiring. These are unchanged; cadence changes are handled
  under E.
- `tldw_server` (sub-project 2), widening the PR gate (sub-project 4), and restoring the nightly
  (sub-project 3).
- **Lessons-file conflicts (deferred).** `merge=union` was considered and dropped:
  - GitHub's server-side merge, and so `gh pr update-branch`, is expected to ignore it.
  - A scratch repro showed it losing the blank line and `---` separator between entries.
  - The files are not append-only. DoD #10 says "add or update", and 13 of the last 200 commits
    to the two big lessons files delete lines.
  - It would also flatter the replay metric, because `git merge-tree` honours the local
    attribute.

  Revisit after A and B land, with GitHub's behaviour verified first.
- Existing "Verified against" paragraphs stay in place. Only new ones stop.

## Design

### A. Diagnostic inventory: stop committing the `summary` totals

**Changes to `scripts/check_persistent_diagnostic_inventory.py`:**

- `--write` no longer emits the top-level `summary` object, and `schema_version` goes from 3 to 4.
- The six totals are derived from the rows wherever they are needed:
  - the drift report (`_summary_lines`, around line 1241), which now compares totals derived
    from the committed rows against totals derived from the rebuilt rows;
  - the stdout lines at around 1773 and 1788.
- Everything else is unchanged: every row, the rules, the ordering, and the requirement that a
  PR's diff shows each added, removed or changed diagnostic for review.
- **No separate schema-3 rejection.** The existing byte comparison plus `_metadata_lines`
  already reports `schema_version 3 -> 4` as drift. Implementation confirms this with a
  negative control. It adds a check only if that control fails.

**Every consumer that must change** (verified on `origin/dev`):

| File | What uses `summary` / schema 3 |
|---|---|
| `Tests/Architecture/test_persistent_diagnostic_inventory.py` | `:1147-1149` asserts `schema_version == 3` and reads `summary`; `:3455-3462` asserts the whole `summary` dict |
| `Tests/Architecture/test_derived_artifact_checkers.py:77` | edits `rebuilt["summary"]` |
| `Tests/Architecture/test_diagnostic_path_privacy.py:1004` | builds a `summary` fixture |
| `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py` | `normalized["summary"]` and `_assert_task_492_summary` (`:1067-1101`, `:2291-2293`); two mutant tests edit `summary` (`:2388-2415`) |
| `Tests/fixtures/summarization_diagnostic_review.json:4-5` | pins SHAs of the normalized inventory. **Do not regenerate** (see below) |

**The review fixture is left alone, deliberately.** Its pinned SHAs certify that the inventory
changed only in the two summarization rows. They are already stale on `origin/dev`: 3 tests in
`test_summarization_diagnostic_privacy.py` fail there today. Regenerating them would certify the
whole current inventory as reviewed, which is the trap in lesson TASK-14651. So A preserves the
exact baseline red set of that file:

- it removes the `summary` masking from the projection;
- it drops the now-structural `_assert_task_492_summary` and the forged-summary mutant;
- it re-points the generated-drift mutant at an owner row.

Finding, filed as a follow-up and not fixed here: those mutant tests are **inert negative
controls**. They pass only because their base test is already red.

None of these files run in preflight or the fast lane, so A's verification runs all of them
explicitly.

**Why the 7 clean-but-different merges stay safe.** Strict protection means every merge has a
green required check on a head that contains the current `dev`, and that check runs the
inventory checker. `scripts/preflight.sh` catches the same thing locally.

**Rollout.** After A lands, each open PR that touched the inventory (7 on 2026-09-27: #2842,
#2838, #2834, #2817, #2563, #2427, #2196) conflicts once and needs one `--write` on its merged
head. That is the same cost one summary conflict has today.

**ADR.** Append a dated amendment to `backlog/decisions/029-local-private-data-boundary.md`:
per-file rows only, totals derived at check time, the evidence above, and an unchanged review
guarantee.

**Rejected alternatives.**

- *One file per source file.* The same 17 real overlaps still conflict, at a far larger diff.
- *Don't commit the inventory.* This removes the reviewed diff ADR-029 depends on (see lesson
  TASK-14651).

### B. User Guide verification stamps

Replace the `CLAUDE.md` "UI changes" rule with:

> **UI changes:** PRs that change a screen's UI update the matching `Docs/User_Guide/` page's
> content where behaviour changed. Record what was verified, and against which branch and date,
> in the task's Implementation Notes, not in the User Guide page. Do not add "Verified against"
> paragraphs to User Guide pages.

The stamp practice is also taught at `backlog/docs/lessons-live-verification.md:3124-3180`
("run it before you stamp anything"). That lesson gets a dated note pointing at the new rule.
Historical plan documents that mention stamps (for example
`plan-2026-09-02-schedules-handoff-pr5.md:26`) stay as they are.

### C. Runner work that produces no signal

**C1. Delete `css-bundle-guard.yml` and `backlog-guard.yml`.** These pins and references must be
updated in the same change. Otherwise the required check goes red, because `Tests/CI` is a
fast-lane target and PyYAML is a core dependency.

- **`Tests/CI/test_ci_queue_pressure_contract.py:47-52`:** drop both files from
  `STANDALONE_WORKFLOWS`. The tests at `:343-357` iterate over it.
- **`Tests/CI/test_derived_artifacts_workflow.py:219-226`:** the test that keeps
  `backlog-guard.yml` and `derived-artifacts.yml` from carrying divergent copies of the check
  becomes moot. Delete it and keep the `derived-artifacts.yml` side.
- **`Tests/Packaging/test_python_runtime_floor.py:94`:** it reads `css-bundle-guard.yml`, so
  retarget it to `derived-artifacts.yml`.
- **Comment and docs references:**
  - `Tests/CI/test_backlog_task_id_uniqueness.py:8`;
  - `Tests/README.md:363-368`;
  - `test.yml` comments at 31/56/69;
  - `derived-artifacts.yml:26`, `:128`, `:255`;
  - `perf-guard.yml:9`, `:43`;
  - `scripts/check_backlog_task_ids.py:20`, `:209`;
  - `scripts/check_bundle_sync.py:15`.
- **Conflict warning:** open PR #2026 edits `backlog-guard.yml` and so will hit a
  modify/delete conflict. Tell its owner.

**C2. Narrow the GGUF evidence triggers.**

- **`task-2062-1-gguf-import-evidence.yml` `paths`:** its own file,
  `tldw_chatbook/Model_Artifacts/**`, `tldw_chatbook/UI/Screens/model_installed_view.py`,
  `Tests/Model_Artifacts/**`, `Tests/UI/test_model_installed_view.py`.
- **`task-2062-2-gguf-source-evidence.yml` `paths`:** its own file, `Model_Artifacts/**`,
  `Event_Handlers/LLM_Management_Events/**`, `UI/LLM_Management_Window.py`,
  `UI/Screens/llm_screen.py`, `Tests/LLM_Management/**`, `Tests/Model_Artifacts/**`,
  `Tests/UI/test_llm_gguf_source_modes.py`.
- **Both keep `workflow_dispatch`.**
- **Pinned tests:** update `Tests/CI/test_task2062_1_gguf_import_evidence.py:35-47` and
  `test_task2062_2_gguf_source_evidence.py:42-58`, which pin the exact lists.
- **Keeping cross-cutting edits covered:** add `Tests/UI/test_model_installed_view.py` and
  `Tests/UI/test_llm_gguf_source_modes.py` to `scripts/ui_pr_gate_census.txt`, if both are green
  on the fast lane's minimal dependency set. That keeps Linux coverage of those screens when
  `app.py`, `css/**` or `conftest.py` change. If either file is not green, record it and leave it
  out; don't gate on red.

**C3. Disable `Nightly Deep` (owner action, zero diff).** Run
`gh workflow disable nightly-deep.yml --repo rmusser01/tldw_chatbook`.

- **Reversible** with `gh workflow enable`.
- **Needs no push to `main`,** where the schedule is registered and where any push starts the full
  `test.yml` run.
- **Leaves alone:** contract tests, `nightly-deep.yml`, and open PR #2819.
- **Recorded** in `backlog/docs/branch-protection-baseline.md` and in the ADR-103 amendment (E).
- **Restored** by sub-project 3 once a run can finish and report.

The task-32011 trigger change from revision 1 is dropped: it saved zero runner-minutes.

### D. Fix the fast-lane flake at its root cause

The fast-lane contract deliberately forbids selection-suppressing flags:

- `Tests/CI/test_ci_queue_pressure_contract.py:325-340` rejects `--deselect`.
- The design doc (`2026-08-29-fast-pr-lane-design.md:253-254`) says selection-suppressing flags
  must not be able to turn a subset into a pass.

The four failing tests also share one error with the tests TASK-32049 already records, so
deselecting them would most likely move the flake, not remove it. So D does not quarantine.

**Mechanism (verified 2026-09-27 against Textual 8.2.8).** It is a latent production bug, not
test timing:

- `Widget._message_loop_exit` calls `self._detach()` and then `self._component_styles.clear()`.
- A screen repaint that was already queued then reaches `TextArea.render_lines`. Its first step,
  `theme.apply_css(self)`, looks up `text-area--gutter` and raises `KeyError`.
- `_detach()` runs first, so `is_attached` is already `False` whenever the styles are gone.

A deterministic reproduction exists: remove a stock `TextArea` from the DOM, then call
`render_lines`, and it raises exactly the CI error.

- **Fix:** add a shared `DetachSafeTextArea(TextArea)`
  (`tldw_chatbook/Widgets/detach_safe_text_area.py`) whose `render_lines` returns blank strips
  when `not self.is_attached` and otherwise renders exactly like the stock widget. A detached
  widget is never visible, so blank is correct.
  - Use it for all five MCP-module TextAreas: `mcp_schema_form.py:244`, `mcp_inspector.py:1614`,
    and `mcp_profile_form.py:107`, `:132`, `:432`.
  - Extend TASK-32049 to cover every test showing this error.
- **Evidence:**
  - the deterministic test fails with stock `TextArea` (negative control) and passes with the
    subclass;
  - an attached-render equality test proves there is no visual change;
  - `Tests/UI/test_mcp_workbench.py` run 10 times on the minimal dependency set before and after,
    as supporting evidence only, since the timing-dependent loop may not reproduce.
- **Fallback if the time box runs out:** bring the owner options, for example moving the file
  into a separate non-required job. That changes the fast-lane target list, so it needs an
  ADR-103 amendment. Don't pick one without the owner.

### E. ADR-103 amendment

ADR-103 says full-tree coverage "remains mandatory" through `main` pushes, manual dispatch and the
nightly. It also says any change to "event cadence" must update it. Append a dated amendment
recording:

- **The nightly:** disabled on 2026-09-27 by the owner. It had 0 of 8 complete runs, and its
  budget went to other work.
- **What full-tree coverage remains:** `main` pushes (last push 2026-09-14) and manual dispatch.
  Sub-project 3 restores the nightly.
- **The deleted guards,** with the Consequences line on routine path-scoped guards and runner
  counts updated to match.
- **The strict re-enable** (already noted in revision 1's docs PR).

## Coverage after this change (stated, not implied)

- **Full suite, any OS:** only on `main` pushes and manual dispatch, until sub-project 3.
  Nothing ran to completion before this change either; the nightly produced no complete result
  in 8 nights.
- **macOS/Windows, any automatic cadence:**
  - `test.yml`'s `artifact-lease-spike` (3 OSes) on `main` pushes;
  - the narrowed GGUF workflows;
  - label-gated evidence workflows (19637, 598-603);
  - `voice-aec-wheels` on its paths.
- **Windows collection failures:** the kind that aborted the nightly (the cp1252 errors of
  #2818 / task-32916) would go unnoticed. A cheap weekly Windows `--collect-only` canary is
  listed for sub-project 3.
- **GGUF screens:** covered on Linux in the PR gate through C2's census additions, and
  cross-OS when GGUF code changes.

## Rollout order under strict protection

Each step is its own PR, and each merges before the next is started:

1. **#2848** (this spec plus the merge-gate docs). B edits the same `CLAUDE.md` lines, so this
   lands first.
2. **A alone:** small and fast, in a quiet window. It touches the inventory, which is the file
   everything conflicts on, so it goes first of the implementation PRs.
3. **C1 + C2 + E,** plus the C3 owner action, recorded.
4. **D:** the root-cause fix.
5. **B:** the `CLAUDE.md` rule and the lesson note.

## Verification

- **A:**
  - the consumer test files keep `origin/dev`'s exact baseline red set, compared by node-id
    set: 3 failures in the summarization privacy file and 2 in
    `test_persistent_diagnostic_inventory.py`, all present before A, and no new ones;
  - `--write` then check round-trips;
  - a row edit fails with total deltas shown;
  - a schema-3 file is reported (negative control);
  - preflight rc=0;
  - the sync-merge replay re-run on schema-4 files reproduces roughly 102 → 17, using the
    committed `scripts/measure_sync_merge_conflicts.py`.
- **C:**
  - every pin listed in C1 and C2 updated, and `Tests/CI` passing on the minimal dependency set;
  - the census additions green;
  - `gh workflow view nightly-deep.yml` shows `disabled_manually`;
  - a table of sample changed-path sets (`app.py` only, `css/**` only, `Tests/conftest.py` only,
    `Model_Artifacts/**`, `pyproject.toml` only) showing which workflows fire.
- **D:** the 20-run before/after counts, and the flake's root cause named with file:line.

## Success measures (review about 2 weeks after rollout step 5)

The baseline (09-19..26) was taken under `strict=false`, and with the `tldw_server` duplicate lane
still running. That lane was removed on 2026-09-27. So end-to-end numbers mix this
sub-project's effect with the `tldw_server` fix, and the attributable measures are separated
out below.

**Throughput (the owner's goal; not attributable to this sub-project alone):**

- merges per day;
- median time from ready to merged;
- required-check push-to-verdict p50/p90, split into queue time and run time;
- re-sync cycles per merged PR;
- DIRTY share of open PRs, sampled daily.

**Attributable to this sub-project:**

| Measure | Baseline | Target |
|---|---|---|
| Sync merges with any conflict (replay, no local merge attributes) | 48% | ≤ 25% |
| Runner-minutes per day for the changed workflows (GGUF ×2, guards ×2, nightly) | ~1,100 | ≥ 80% lower |
| Fast-lane failures carrying the `text-area--gutter` error | 16 of 37 failed jobs | 0 |

## Risks

- **A clean-but-wrong inventory merge lands on `dev`.** Strict protection prevents that: the
  required check always runs on a head that contains the current `dev`.
- **D's time box runs out.** This falls back to an owner decision; nothing is quarantined
  silently.
- **Full-suite coverage is effectively zero until sub-project 3.** That is stated in the ADR-103
  amendment rather than hidden. The nightly produced nothing either.

## Program context

**Owner priority order (2026-09-27):**

1. CI throughput: sub-projects 1, 2 and 5.
2. Clear the open-PR backlog. On 2026-09-27 there were 34 open PRs: 30 to `dev`, 5 drafts, 4
   conflicting, and 10 idle for more than 14 days.
3. Tests and issues: sub-projects 3 and 4, plus TASK-32049 if D's fallback applies.

Sub-project 4 widens the gate, which adds CI work per PR, so it waits for phase 3.

**Merge gate (owner decisions, applied 2026-09-27):**

- `dev` is strict again.
- `allow_update_branch` and `allow_auto_merge` are on.
- Required conversation resolution is on.

These are recorded in `backlog/docs/branch-protection-baseline.md`, and ADR-103 has a note. Under
strict, A is a throughput prerequisite. Before A, 80% of re-syncs where both sides touch the
inventory conflict, and a conflicting PR gets no CI.

**The nightly's red is mostly one harness cause, not product bugs.** Its three worst files,
re-run locally on `origin/dev` on 2026-09-27:

- `Tests/Agents/test_local_tool_provider.py` fails 248 of 263, all with
  `RecoveryRequired: raw_source_selection_changed`.
- `test_fleet_runtime.py` fails 146 and `test_agent_service.py` fails 79, matching the nightly's
  counts. Both are dominated by the same backup/recovery gate.

**Sub-project 2, the account queue.** The main cause was removed on 2026-09-27: `tldw_server`'s
`LICENSE_FIRST_CI_ENABLED` duplicate lane of 750-job runs, which filled the account's 40-job cap.
What remains is `tldw_server`-side:

- each audit still creates 28 no-op `workflow_run` runs and 1 ungated `ci.yml` preflight job;
- finish its TASK-12986 cutover.

**Sub-projects 3-5:** the nightly (starting with the `RecoveryRequired` harness cause, plus a
Windows collection canary); a wider PR gate (amends ADR-103, which rejected path-based selection
because `Docs/` and `backlog/` are test inputs); fewer cycles per PR.
