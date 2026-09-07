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
named 17 files; the delta was reviewed statement-by-statement (each added and
removed logger call was recovered by scanning both revisions with the checker's
own AST scanner, not by reading a line diff) before `--write`. 65 statements
were added and 7 removed. Almost all of it
is metadata-only — `type(exc).__name__`, ids, counts — or pure code movement
(the three character-picker diagnostics moved from `chat_screen.py` into
`UI/Console_Modules/character.py` unchanged; `app.py` is the one count-preserving
digest change and it is a local rename, `_log_buffer` -> `_log_records`).
**Rows worth a follow-up, deliberately not absorbed silently:** see Owner gate
item 5 — the regeneration pinned 21 newly-added diagnostics across five files
that interpolate a filesystem path into a log line, the class the open
TASK-19321/TASK-19322 repairs cover.

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
- The report against the real dev-tip drift named 17 files and every count/digest
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

## Pre-merge re-review at the final base (2026-08-22)

Rebased onto `origin/dev` `d4f3f9776`; the one conflict
(`Docs/security/production-diagnostic-inventory.json`) was resolved by taking
**dev's** copy wholesale, discarding this branch's own regeneration.

**The gate arrives green, and no `--write` was run.** Dev's copy reproduces from
dev's tree byte-for-byte — verified twice, once by the checker itself
(`--diff` → *"no drift: the committed inventory matches the rebuild exactly"*,
521 owners / 1221 TASK-492 / 7208 TASK-494 / 7 sink files) and once
independently by rebuilding every owner row from `HEAD`'s git blobs with the
checker's own scanner. This branch changes no file under `tldw_chatbook/`, so it
contributes no drift of its own. Regenerating here would have been the exact
"regenerate without reading" failure the artifact exists to prevent.

**The three drifted rows were still reviewed statement-by-statement**, because
they are unreviewed *by this branch*: the reviewed baseline is the pre-rebase
tip `ee7464d54` (whose pin also reproduces from its own tree — checked, unlike
the `0b112ab1e` pin in Owner-gate item 6). Delta `ee7464d54` → `d4f3f9776`,
recovered with the AST scanner across both revisions, never a line diff:

| row | delta | statements | verdict |
|---|---|---|---|
| `Chat/console_fleet_wake.py` | 11→11, digest changed | 2 calls **re-indented only**; whitespace-normalized text identical on both sides | benign — pure layout |
| `Persona_Buddy/preferences.py` | new row, 0→1 | `logger.bind(exception_type=exception_type).error("persona_buddy_preferences_save_failed")`, where `exception_type` is `type(error).__name__` *regex-validated* before binding | benign — metadata only |
| `UI/Screens/personas_screen.py` | 180→186 | 6 × `logger.warning` with fully static messages (`"Persona Buddy … failed (category=…)"`), no interpolation | benign |

Multiset-checked: raw counts match the pin exactly (180→186, 11→11, 0→1) with
zero multiplicity-only changes, so those 9 statements are the *complete* delta.
**No new exposure found** — nothing interpolates user content, a secret, a path,
or a URL. Nothing was absorbed.

### The report was not sufficient, and has been fixed

Judged in anger, the `--diff` report failed on the one row that mattered. It
labelled `console_fleet_wake.py` *"reworded / re-levelled / new args"* — all
three wrong. The calls had only been **re-indented** when the surrounding code
was refenced. The per-call digest is taken over `ast.get_source_segment`, which
keeps continuation-line indentation, so shifting a call's nesting level moves the
file's digest even though the module docstring says movement is explicitly *not*
a review event. That is task-3750's failure mode reappearing one level down.

Worse, the trailer's recovery recipe was `git diff <base> -- <path>`. Following
it on that row yields a **328-line diff containing zero statement changes** — the
reviewer's job becomes finding two re-indented calls inside an unrelated
248-insertion refactor, which is exactly how "it was probably fine" gets written.

Fixed in this commit:

- **`--statements <path> ... [--since REV]`** — a real recovery mode. It scans
  both revisions with the *same* scanner and the *same* per-call digest the pin
  uses (pinned by a test), pairs off statements whose only difference is layout,
  and prints the full source text of everything genuinely added or removed under
  a `|` gutter at its original shape. On the incident row it answers in one
  line: `moved/re-indented only: 2   removed: 0   added: 0`.
  A path absent at the base revision (an "only in rebuild" row) is reported as
  *"did not exist at REV"*, not as a broken revision argument.
- The changed-row note now reads *"reworded / re-levelled / new args /
  re-indented — use --statements to see which"*.
- `NEXT_STEPS` now hands over the `--statements` command and says plainly not to
  reach for `git diff` here, with the measured reason.
- 8 tests added (`Tests/Architecture/test_derived_artifact_checkers.py`),
  including one that reproduces the re-indentation incident and one asserting the
  recovery mode's digests are the same keys the pin moves.

### Verified at the final base

- `scripts/preflight.sh` — **all four checkers pass in 33 s**: CSS bundle +
  4 generated stylesheets; profile-owned path census (48 occurrences / 18 files /
  46 approved exceptions); diagnostic inventory; backlog ids (2375 task files, no
  duplicates). No other checker is red at this base.
- Mutation proof of the whole loop under `GITHUB_ACTIONS=true`: a planted
  `logger.warning(f"TEMPORARY_MUTATION_PROBE wrote {preferences!r}")` produced
  exit 1, the `::error::` annotation on stdout, the report on stderr naming
  `Persona_Buddy/preferences.py 1/3053… -> 2/b733… (+1 diagnostic call(s))`, and
  the report's own recommended command then printed the offending statement
  verbatim. Restored via Edit; `git status` on `tldw_chatbook/` clean afterwards.
- Suites: `test_derived_artifact_checkers.py` **18**, `test_derived_artifacts_workflow.py`
  **7**, `test_persistent_diagnostic_inventory.py` **65**,
  `test_profile_owned_path_inventory.py` **15**, `test_css_bundle_sync_guard.py`
  **4**, `test_github_actions_test_workflow.py` **15** — **124 passed**
  (the prior 116 plus the 8 new tests).
- Repo-wide `pytest --collect-only -q`: **54,651 tests collected, no collection
  errors**.

## Qodo review fixes, PR #1947 (2026-08-22)

Four findings on the PR: two bugs, two "Unvalidated" rule-violation flags. Fixed
both bugs; the two rule-violation flags are dispositioned below rather than
complied with reflexively, with evidence for each call.

**Finding 1 (High, Bug) — sink diff dropped duplicates. Fixed.**
`_sink_lines()` compared each file's persistent-sink entries by building
`{_sink_key(entry): entry for entry in ...}` -- a plain dict -- so two
byte-identical sink calls (the same handler installed twice) collapsed to one
key and a pure **multiplicity** drift, the highest-consequence class this
artifact exists to catch, could report as no change, or worse. Reproduced
against the real tree, not synthetic data only: planted an exact duplicate of
`Logging_Config.py`'s `loguru_logger.add(...)` call inside
`configure_application_logging`. Against the **pre-fix** code the top-level
check still failed (the raw JSON differed), but `render_diff()`'s own sink
section produced nothing, so the report read: *"the committed inventory's
CONTENT matches the rebuild; only its serialization differs ... Run --write to
re-normalize it"* -- an actively misleading verdict that would have trained a
reviewer to blindly `--write` past a real second sink. Fixed by comparing
`collections.Counter(_sink_key(entry) for entry in ...)` (multiset semantics:
`Counter.__eq__` compares counts) and reporting the delta explicitly per key
(`before_n -> after_n (+delta)`, or `+`/`-` with a `(new, x2)` / `(removed, was
x2)` suffix at the edges). Against the **post-fix** code the same mutation now
reports:
```
persistent sink topology:
  ~ changed sinks: tldw_chatbook/Logging_Config.py (7 -> 8 entries)
      ~ configure_application_logging: loguru_sink.add (9fce73a232622dc7): 1 -> 2  (+1)
```
Mutation restored via Edit; `git diff -- tldw_chatbook/Logging_Config.py` is
empty and `--diff` is back to "no drift" against the unmodified tree. Two new
synthetic tests pin both directions (`test_sink_multiplicity_increase_is_
named_with_counts`, `test_sink_multiplicity_decrease_is_named_too`), asserting
the exact count string is present and that "only its serialization differs"
does **not** appear when there is real drift.

**Finding 4 (Bug, Reliability) — `--statements` crashed on an out-of-repo
absolute path. Fixed.** `_run_statements()` called
`path.relative_to(REPO_ROOT)` unconditionally for an absolute `PATH`, so
`--statements /etc/hosts` raised an unhandled `ValueError` -- a traceback from
exactly the recovery tool the failure report tells a developer to run.

**Finding 3 (Rule violation, "Unvalidated `--statements` paths read") — fixed
on the merits, not with the literally suggested implementation.** Qodo's
suggested fix (`path_validation.py`) is inappropriate for this file:
`Utils/path_validation.py` imports `Metrics.metrics_logger`, which imports
`psutil` (third-party) -- pulling it into a script this task's own AC requires
to stay stdlib-only and install-free would break that contract for every
future run of `derived-artifacts.yml`. The underlying concern (a relative `..`
path silently reading outside the repo) is real and is fixed directly: new
helper `_repo_relative()` resolves the candidate against `REPO_ROOT`, and
returns `None` -- never raises -- for anything that resolves outside it,
covering both Finding 3 (relative traversal) and Finding 4 (absolute
out-of-repo path) with one code path. `_run_statements()` now prints a clear
one-line stderr message and returns 1 instead of crashing or reading outside
the repo. Verified: `--statements /etc/hosts` and
`--statements ../../../../../../etc/hosts` both now print `cannot use ...: it
does not resolve inside the repository ...` and exit 1 with **no traceback**;
ordinary relative and absolute in-repo paths are unaffected. Checked the file
for the same unguarded pattern elsewhere (per the task instructions): the two
other `relative_to(REPO_ROOT)` call sites (`build_inventory()`'s file walk and
the `--write`/missing-file messages against the hardcoded `INVENTORY_PATH`)
never take CLI input, so they were not at risk. Six new tests cover
`_repo_relative()` (accept relative-in-repo, accept absolute-in-repo, reject
absolute-outside, reject relative-traversal) and `_run_statements()` (clean
error with no `Traceback` string for both outside-repo shapes; ordinary in-repo
path still returns 0).

**Finding 2 (Rule violation, "Unvalidated `--tasks-dir` used") — declined, with
evidence; not applied.** Same `path_validation.py` objection as Finding 3
applies (stdlib-only contract), but there is a second, independent reason this
one specific "fix" would be actively wrong here: `Tests/Architecture/
test_derived_artifact_checkers.py:305,312` (pre-existing, this branch did not
add them) deliberately pass a pytest `tmp_path` fixture -- outside `REPO_ROOT`
by construction -- as `--tasks-dir` to exercise the checker in isolation.
Confining `--tasks-dir` to the repo root, as the suggested fix would do, breaks
that test and the flag's own stated purpose (`--tasks-dir` exists precisely so
callers, including tests, can point it elsewhere). Separately: neither
workflow that runs this script (`backlog-guard.yml`, `derived-artifacts.yml`)
ever passes `--tasks-dir` -- both invoke the script bare -- so there is no
CI-reachable attacker-controlled input reaching this flag at all; the only
caller is a developer's own CLI argument in their own shell, already reading
files at their own OS-level permission, which is not a privilege boundary a
"traversal" can cross. No code-behavior change; added a code comment at the
argument definition recording this reasoning so a future reviewer (human or
Qodo) does not treat the omission as an oversight.

**Verified after all four dispositions:**
- `scripts/preflight.sh` (`GITHUB_ACTIONS=true`) — all four checkers still
  green: CSS bundle + 4 sheets; profile-owned path census (48/18/46, unchanged);
  diagnostic inventory (`no drift: the committed inventory matches the rebuild
  exactly` -- **the pin still reproduces byte-for-byte; nothing was
  regenerated**); backlog ids (2375 files, no duplicates).
- `Tests/Architecture/test_derived_artifact_checkers.py`: **27 passed** (the
  prior 18 plus 9 new: 2 sink-multiplicity, 4 `_repo_relative`, 3
  `_run_statements`).
- The five guard suites named in the review brief together: `test_derived_
  artifact_checkers.py` **27**, `test_derived_artifacts_workflow.py` **7**,
  `test_persistent_diagnostic_inventory.py` **65**, `test_profile_owned_path_
  inventory.py` **15**, `test_css_bundle_sync_guard.py` **4** — **118 passed**
  (109 pre-existing + 9 new).
- Repo-wide `pytest --collect-only -q`: **54,660 tests collected** (54,651 +
  9 new), **no collection errors**.
- `git status` after all restores: only `Tests/Architecture/test_derived_
  artifact_checkers.py`, `scripts/check_persistent_diagnostic_inventory.py`,
  and `scripts/check_backlog_task_ids.py` modified -- nothing under
  `tldw_chatbook/` or `Docs/security/production-diagnostic-inventory.json`.

Commit: `fix(guard): address Qodo review findings on PR #1947 (task-19572)`.
Status stays **In Progress**: the owner-gated ACs (branch protection, required
check, main's cron) are unaffected by this fix-up and remain open per "Owner
gate" below.

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

**5. Follow-ups to file — path-interpolating diagnostics absorbed by this
regeneration.** Independent review (2026-08-22) re-derived every added/removed
statement from the two revisions and found the class is **wider than one file**.
21 newly-pinned diagnostics across five files interpolate a filesystem path:

| file | new path-logging calls | what is interpolated |
|---|---|---|
| `UI/Screens/change_review_screen.py` | 8 | `{root!r}` ×6, `{roots!r}`, `{remote!r} at {root!r}` — workspace root paths |
| `Widgets/Console/console_conversation_inspector.py` | 5 (3 net new; 2 carried from the retired `console_context_modal.py`) | `{path}` — user-chosen export/snapshot destination |
| `DB/ChaChaNotes_DB.py` | 4 | `{self.db_path_str}` in the V42→V43 / V43→V44 migration lines (replicates the pre-existing style of the ~330 calls already in that file) |
| `Utils/file_handlers.py` | 3 | `{file_path}` — lines **380, 410, 441** (task-19576) |
| `Workspaces/git_workspace.py` | 1 | `{root}`, at `debug` level |

**Severity is lower than "into a persistent sink" implies, and the wording
elsewhere in this file should be read with that correction.** The only
general-purpose on-disk log is `PrivateRotatingFileHandler`, and it carries
`PersistentDiagnosticFilter`, which admits **only** records marked by
`log_persistent_metadata` / `persist_event`. A plain
`logger.error(f"... {file_path} ...")` is a Chatbook record without that marker,
so `filter()` returns `False` and it never reaches the rotating file. Verified
directly:

```
PersistentDiagnosticFilter().filter(<ERROR record from tldw_chatbook.Utils.file_handlers>)
-> False
```

These lines therefore reach the terminal and the in-app Logs screen, not the
private log file. They remain in scope for TASK-19321/19322, but as a UI/console
exposure, not an on-disk one — and any follow-up should cover all five files,
not only `file_handlers.py`.

**6. Two rows in this regeneration predate this branch's base and were never
reviewed when they landed.** The previous pin (blob `b07d9e10f`, committed at
`0b112ab1e`) did **not** reproduce from the tree of its own commit: rebuilding
the inventory from `0b112ab1e`'s git blobs yields `Client_Media_DB_v2.py`
339 (pinned 338) and `library_screen.py` 111 (pinned 109). Because the drift
predates the base, a statement-recovery diff between the pin's commit and the
base — the method used here — shows *nothing* for those two rows, and they were
carried into the new pin unexamined. Traced independently and both are benign:
`Client_Media_DB_v2.py` gained one `logger.info("Media search completed
(mode={}, limit={}, offset={}, sort={}, summary=true).", ...)` at `e351a9c99`;
`library_screen.py` gained two static-message warnings at `a85681ba0` and
`f72cc8c2b`. No action needed beyond knowing the gap exists — which is itself
the argument for the gate this task ships.

**7. The pin must be regenerated against the FINAL base immediately before
merge.** The inventory is a whole-tree artifact and `dev` moves 23–50 times a
day. Measured 2026-08-22: this branch's freshly regenerated pin was *already* red
against `origin/dev` `cbb11633f` — three rows drift
(`Chat/console_fleet_wake.py` digest, `Persona_Buddy/preferences.py` new row,
`UI/Screens/personas_screen.py` 180→186). Since a `pull_request` check runs on
the merge ref, a stale pin turns the required check red on PRs that did nothing
wrong. Two consequences the owner should price in before flipping the switch:
any PR that adds a logger call must regenerate, and two such PRs racing will
both conflict textually on the same sorted JSON. Neither is a defect in this
branch; both are costs of making *this* artifact the required gate.

**Resolved at the final base — see "Pre-merge re-review" below.** The rebase onto
`origin/dev` `d4f3f9776` took *dev's* copy of the inventory, and dev's copy
reproduces from dev's tree exactly, so the gate arrives **green** with no
regeneration by this branch at all. The three predicted rows were reviewed
statement-by-statement anyway, because they are unreviewed *by this branch*.
