---
id: TASK-19572
title: >-
  CI has not completed successfully since 2026-06-26 and dev has no branch
  protection — ship one install-free required check
status: In Progress
assignee: ['@claude']
created_date: '2026-08-21 20:25'
labels:
  - ci
  - process
  - infrastructure
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 7 (process, tooling & repo health) —
its **F1** and the recommendation from **F2**. The lane called F1 **the root
cause under most of its other findings**, and this filing's verification found
it is **worse than reported**.

**The Tests workflow has not completed successfully since 2026-06-26** — 56
days. Last success anywhere: `2026-06-26T18:00:01Z` (branch
`codex/personas-rail-collapse`); last success on `dev`: `2026-06-26T00:17:48Z`.

Measured now, over the last **500** runs (2026-08-17 → 2026-08-21):
**423 cancelled / 62 failure / 15 in-flight / ZERO success.**

Mechanism — `.github/workflows/test.yml:20-22`:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

**Correction that makes this materially worse than the review stated:** the
required quiet window is **not ~60 minutes**. The last two *successful* runs
took **200.1 and 226.8 minutes**. Recent cancelled runs die at 9–63 minutes.
Measured merge rate to `origin/dev`: **39 / 23 / 12 / 45 / 10 merges** on
Aug 17–21 (peak 62 on Aug 11). A 3.5-hour uninterrupted window on `dev` is
unattainable. `main` is exempt from cancellation but never receives pushes.

**And even a completed red run could not stop anything.**
`gh api repos/rmusser01/tldw_chatbook/branches/dev/protection` → **404 Branch
not protected**. Same 404 for `main`. No required status checks. No git hooks
(`core.hooksPath` → `.git/hooks`, **0 non-`.sample` files**). No
`.pre-commit-config.yaml`.

This is the reason the repo's standing "CI is intentionally cancelled, verify
locally" workaround exists — and the reason TASK-19568's schema pins were
**merged red**. The suite that would have caught them exists, is in the CI job,
and has produced **no verdict in eight weeks**.

**Cost is not the obstacle.** All three derived-artifact checkers are
**stdlib-only** (verified two ways: static import analysis, and a runtime
`sys.modules` audit finding nothing from site-packages) and cheap:

| checker | measured |
|---|---|
| `tldw_chatbook/css/check_bundle_sync.py` | **0.80 s** |
| `scripts/check_profile_owned_path_inventory.py` | **10.81 s** |
| `scripts/check_persistent_diagnostic_inventory.py` | **20.50 s** |
| **total** | **32.1 s** (~90 s with checkout) |

Only `check_bundle_sync.py` runs in CI today
(`.github/workflows/css-bundle-guard.yml:63`). The other two are reachable
**only** through `Tests/Architecture/…` — i.e. through the suite that never
completes.

**The checker is also unactionable when it does fire.**
`scripts/check_persistent_diagnostic_inventory.py:326-354` has exactly one
flag, `--write` — no `--diff`, no `--verbose`. On mismatch it prints one
sentence naming no file, no owner and no call site:
*"production diagnostic owners or persistent-sink topology changed; review the
diff before running --write"*. The only route to detail is to run `--write`
(overwriting the checked artifact) and then `git diff` by hand. Every past
burn-down task hand-rebuilt a ~30-line diff probe. This is not new:
**task-2768 (2026-08-07) is the same incident**, and 15 commits since 08-08
exist whose only change is this JSON.

**The lane's recommendation, with alternatives already rejected on evidence:**
make the checker **teach the fix** (`--diff` on every non-zero exit:
rows-only-committed / only-rebuild / changed, with old→new counts and digests,
plus the next command), carry all three in a **new install-free
`derived-artifacts` job**, and make **that** the one required check on `dev`.

Alternatives the lane rejected, with its reasons — do not relitigate without
new evidence:
- **pre-commit hook** — 22.6 s per commit, bypassable with `-n`, does not
  survive a clone.
- **auto-sync-with-review** — forbidden by the artifact's own design rationale
  (it trains regeneration without reading).
- **classification-only pin** — measurably worse: catches PR #1869 only,
  misses #1877 and #1880, which are the majority of the risk surface.
- **adding it to `test.yml`** — already there; that is precisely why it caught
  nothing.

Per the owner's standing ruling, this is the durable/pragmatic choice: one
small check that actually runs beats a comprehensive suite that never reports.

## Acceptance Criteria

- [x] A `derived-artifacts` CI job exists that runs the three stdlib-only
      checkers with no dependency installation, completing in roughly 90 s
      including checkout
- [ ] That job is a **required status check** on `dev`, so a red guard blocks a
      merge — **owner action, see "Owner gate" below**
- [ ] Branch protection exists on `dev` (and a decision is recorded for `main`)
      — **owner action, see "Owner gate" below**
- [x] `check_persistent_diagnostic_inventory.py` gains a `--diff` mode that
      runs automatically on every non-zero exit and names what changed —
      rows-only-committed / only-rebuild / changed, with old→new counts and
      digests, and the exact next command
- [x] The other two checkers are held to the same standard: when they fail they
      name the offending file
- [ ] The Tests workflow's cancel-in-progress behaviour is addressed so that
      *some* full-suite verdict is produced on a regular cadence — e.g. a
      scheduled run on a quiet branch, sharding, or exempting a nightly ref.
      The 200–227 minute runtime is the constraint to design around (see also
      TASK-19425, which covers the runtime growth itself) — **root-caused here
      and NOT fixable from `dev`; see "Owner gate"**
- [ ] The first completed full-suite run since 2026-06-26 is recorded, with its
      pass/fail counts, so the repo knows its actual baseline — **blocked on the
      two ACs above**

## Implementation Plan

1. Reproduce the measurements on this machine: run all three checkers, record
   real timings, confirm the diagnostic-inventory pin is red at the current
   `origin/dev` tip.
2. Promote the throwaway diff probe into the checker: a report emitted on
   **every** non-zero exit naming only-in-committed / only-in-rebuild / changed
   rows with `old_count/old_digest -> new_count/new_digest`, sink-topology
   deltas, metadata deltas, and the exact next command.
3. Use that report to do the per-row review the artifact's design demands, then
   regenerate the inventory so the new gate is green on arrival (a required
   check that is red on day one cannot be turned on). Surface, do not absorb,
   anything the review flags.
4. Hold the other two checkers to the same standard: name the file, say what to
   do next, and print a positive line on success.
5. Ship `.github/workflows/derived-artifacts.yml` — install-free, one job, all
   four checks, modelled on `css-bundle-guard.yml` but **unfiltered**, because a
   path-filtered check cannot be required.
6. De-duplicate the backlog duplicate-id logic into `scripts/check_backlog_task_ids.py`
   so the required job and `backlog-guard.yml` cannot drift apart.
7. Add `scripts/preflight.sh` so the local burn-down is one command.
8. Shape tests for the workflow and unit tests for the new report; run the
   pre-existing guard suites for regressions.
9. Write up what is and is not in an implementer's control.

## Implementation Notes

Shipped the guard that can actually report, and the tooling that makes its
failures self-explaining. Branch protection itself is untouched — that is a
repo-admin action and is written up under **Owner gate** below.

**`--diff` on every non-zero exit.** `check_persistent_diagnostic_inventory.py`
previously failed with one sentence naming nothing, so four separate burn-down
tasks each hand-built a ~30-line probe. `render_diff()` now reports, on every
failing exit and with no flag required: summary deltas, owner rows
only-in-committed / only-in-rebuild / changed (`old_count/old_digest ->
new_count/new_digest`, and a count-preserving digest change is labelled
*reworded / re-levelled / new args* — the case that matters for privacy), per-
entry persistent-sink deltas, **metadata deltas** (the check compares the whole
encoded file, so a changed `classification_rules` would otherwise fail with zero
rows and read as a false alarm), a distinct message for serialization-only
drift, and the exact next command. `--diff` also exists as an explicit flag, and
a `::error::` annotation is emitted only under `GITHUB_ACTIONS`.

**The inventory was red at `origin/dev` 3193816e7 and is now green.** The report
named 16 files; the delta was reviewed statement-by-statement (each added and
removed logger call was recovered by scanning both revisions with the checker's
own AST scanner, not by reading a line diff) before `--write`. Almost all of it
is metadata-only — `type(exc).__name__`, ids, counts — or pure code movement
(the three character-picker diagnostics moved from `chat_screen.py` into
`UI/Console_Modules/character.py` unchanged). **One row is worth a follow-up and
is deliberately not absorbed silently:** task-19576 added three
`logger.error(f"Failed to extract ... from {file_path}: {e}")` calls in
`Utils/file_handlers.py`, which put a full user file path in a persistent sink —
the same class as the open TASK-19321/TASK-19322 path-logging repairs.

**`derived-artifacts.yml`.** One job, four stdlib-only checks, `setup-python`
and nothing else installed. Two deliberate departures from `css-bundle-guard.yml`:
it is **not path-filtered** (a skipped required check never reports, so GitHub
parks the PR on "Expected — waiting for status to be reported" forever; at ~90 s
it is cheaper to just always run), and every checker step carries
`if: ${{ !cancelled() }}` so one red checker cannot hide the other three — a
burn-down must see all the drift in one pass. The job name is the string the
owner types into branch protection, so the shape test pins it.

**De-duplication.** The duplicate-task-id check moved out of inline shell into
`scripts/check_backlog_task_ids.py`; `backlog-guard.yml` now calls it, and so
does the new job — two hand-maintained copies of a guard is the failure mode
this whole task is about. Behaviour is unchanged (both namespaces: filename
prefix and frontmatter `id:`), verified against the old shell pipeline on the
live tree and against a synthetic collision.

**Also:** `check_profile_owned_path_inventory.py` gained a next-step trailer on
failure and a positive line on success (it used to print nothing at all when it
passed, which is indistinguishable from a step that did not run).
`scripts/preflight.sh` runs all four locally in one command and reports every
failure, not the first.

### Verified

- Timings on this machine (M-series, `.venv` python 3.12): bundle-sync
  **0.86 s**, profile-owned-path **10.98 s**, diagnostic inventory **20.17 s**,
  backlog ids **0.10 s**; `scripts/preflight.sh` end-to-end **31.9 s**.
- The report against the real dev-tip drift named 16 files and every count/digest
  delta.
- Mutation proof, diagnostic checker: adding one `logger.warning` in
  `Workspaces/git_workspace.py` and re-levelling one `logger.info` ->
  `logger.warning` in `Notes/sync_engine.py` produced
  `~ changed: tldw_chatbook/Workspaces/git_workspace.py 1/3b62237e439a1aafa3a8 -> 2/1177de3ed8987e4d8c5f  (+1 diagnostic call(s))`
  and `~ changed: tldw_chatbook/Notes/sync_engine.py 20/b597ae206b3e2b1ddfd3 -> 20/fd7db158e7e83f7706ec  (same count, content changed ...)`.
  Both mutations restored via Edit.
- Mutation proof, profile checker: a planted `"~/.config/tldw_cli/git.toml"`
  literal produced
  `tldw_chatbook/Workspaces/git_workspace.py:39: module:TEMPORARY_MUTATION_PROBE: literal:~/.config/tldw_cli/git.toml: unapproved occurrence`.
  Restored via Edit.
- `Tests/Architecture/test_derived_artifact_checkers.py` +
  `Tests/CI/test_derived_artifacts_workflow.py`: **18 passed**.
- Pre-existing guard suites (`test_persistent_diagnostic_inventory.py`,
  `test_profile_owned_path_inventory.py`, `test_css_bundle_sync_guard.py`,
  `test_github_actions_test_workflow.py`): **98 passed** — including
  `test_production_diagnostic_inventory_and_sink_topology_are_unchanged`, which
  was red at this base before the reviewed regeneration.

### NOT verified

The workflow has never executed. It parses as YAML and its shape is pinned by
tests, but nothing here proves it is green on a GitHub runner, and the ~90 s
figure is local work plus an estimate for checkout/setup — see the queue-time
finding below, which makes *wall* time a different question entirely.

## Owner gate — what an implementer cannot ship in a PR

**1. Branch protection + the required check (ACs 2, 3).** Requires repo-admin
API writes, which this task deliberately did not perform. After merge:
`Settings -> Branches -> add rule for dev -> Require status checks to pass ->`
select **`Derived artifacts reproduce from their sources`**. Record the `main`
decision at the same time (`main` is currently unprotected too). Note the
workflow must have run at least once for GitHub's picker to offer the name.

**2. The nightly full-suite cadence (AC 6) cannot be fixed from `dev` at all.**
The premise needs correcting: `cancel-in-progress` is *not* what stops the
nightly. `github.ref` for a scheduled run is the default branch, and the rule is
`cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` — so scheduled runs
are already exempt. The real reason is simpler and worse:

> **GitHub schedules cron only from the default branch's copy of the workflow.**
> The default branch is `main`, last updated **2026-07-11** and **10,933 commits
> behind `dev`**, and `main`'s `test.yml` has **no `schedule:` block at all**.
> `gh run list --workflow=test.yml --event=schedule` returns `[]` for the whole
> retained history. The `nightly-deep` job has never run, and adding any
> `schedule:` trigger on `dev` will remain inert until `main` is updated.

Owner options: (a) fast-forward/merge `dev` into `main` so the crons the repo
already wrote start firing, or (b) accept that no cron works and drive the full
suite by `workflow_dispatch` on a cadence. Either is an owner call; (a) has
consequences well beyond CI.

**3. Runner starvation is the other half, and it bounds even this job.**
Measured over the last 100 `css-bundle-guard` runs — one stdlib script, no
install: **61 of 83 `pull_request` runs cancelled, 19 succeeded**, and the
successes are bimodal — ten finished in **16–398 s**, nine took **1.9–5.6
hours**. The job did not get slower; it waited for a runner. `Tests` fans out
~20 jobs (2 core legs + 12 UI shards + 3 lease legs + …) on every push and PR at
23–50 merges/day and starves the pool. Sixty `Tests` runs were created in one
74-minute window while this task was in progress; every one was cancelled or
still queued. Until that fan-out is cut (TASK-19425), a *required* check will be
correct but sometimes slow to report. Worth the owner's consideration:
restricting `Tests` to `workflow_dispatch` + schedule until its runtime is
fixed, since it has produced no verdict since 2026-06-26 and is currently
costing the guards that do work.

**4. A one-line CLAUDE.md addition, left to the owner** (agents must not edit
project instructions on another agent's say-so). Suggested, under the Definition
of Done:

> Run `./scripts/preflight.sh` before opening a PR — it runs the same four
> derived-artifact checks as the `Derived artifacts` CI job.

**5. Follow-up to file:** `Utils/file_handlers.py` logs full user file paths in
three new `logger.error` calls (task-19576), same class as TASK-19321/19322.
