---
id: TASK-19425
title: Core CI jobs now exceed their 120-minute ceiling and get canceled mid-suite
status: In Progress
assignee:
  - '@rmusser01'
created_date: '2026-08-21'
updated_date: '2026-10-01 01:03'
labels:
  - ci
  - testing
  - triage
dependencies: []
priority: high
---

## Description (the why)

With TASK-19160's fix in (PR #1858), the Core jobs no longer die on the
xdist INTERNALERROR — the workers survive the full run. What that exposed:
**both Core jobs now hit `timeout-minutes: 120`** (`test.yml` line 50) and
are canceled while still making progress, so the job reports no test
summary at all.

The growth trend predates 19160 and is visible in merged PRs' Core
durations (ubuntu): **#1840 79 min → #1835 87 min → #1823 121 min
(canceled) → #1858 120 min (canceled, both platforms)**. Dev gained ~17
PRs on 2026-08-19/20 (realtime voice, research runs, focus mode, latency
guardrails, …), and the all-but-UI suite was ~20.6k tests at the last
completed count.

Confounder to rule out at measurement time: the #1858 runs executed on
runners after a ~7-hour queue backlog (03:40–06:00 UTC), which may inflate
wall-time; #1835's runs at similar hours took 61–87 min, so backlog alone
does not explain the full jump.

## Acceptance Criteria (the what)

- [x] The slowest test files/modules in a Core run are measured (pytest
      `--durations` or the json-report artifact), not guessed — naming
      whether the growth is a few pathological tests, a hang, or broad
      accretion
- [x] A deliberate remedy ships: split/shard the Core job, raise the
      ceiling with the reason recorded, or fix the named slow tests —
      chosen from the measurement, with the owner consulted if the ceiling
      moves
- [ ] Both Core jobs complete (pass or fail) within their ceiling on a PR
      run, reporting a real test summary — PENDING LIVE CI PROOF: the
      matrix is six core shards now (TASK-21411/22250 territory), and no
      core shard of the Tests workflow has completed since 2026-09-14
      (every later dispatch was backup-only), so within-ceiling completion
      cannot be demonstrated from existing runs. See Implementation Notes
      for what still blocks it.
- [x] Any per-test timeout interaction is checked: `--timeout=300` with
      `timeout_method = "thread"` kills the whole worker process on a
      single hung test, which under `--max-worker-restart=3` can silently
      re-run large scopes and multiply wall-time

## Implementation Plan

1. Measure from real artifacts, not guesses: locate recent completed
   core-shard runs of `.github/workflows/test.yml` via `gh run list`,
   download their `core-test-results-<shard>` json-report artifacts, and
   rank slowest tests/files plus per-stage (setup/call/teardown) totals.
2. Classify the growth: pathological hangs (tests pinned at the
   `--timeout=300` cap), a few heavy soak/qualification tests, or broad
   accretion; verify each named pathology still reproduces at the current
   base before touching code.
3. Fix the pathologies the measurement names (test-code changes only;
   production changes only if a measured test exposes a real product
   hang), verifying each fix with a targeted local before/after run.
4. Check the `--timeout=300` / `timeout_method` / `--max-worker-restart=3`
   interaction analytically against the flags as they exist today.
5. Record the measurement table, the remedy chosen, and any ceiling or
   sharding decision that only the owner can make, in Implementation
   Notes; leave AC #3 (live CI proof within ceiling) explicitly pending
   unless a real run demonstrates it.

## Implementation Notes

Measured 2026-09-30, base `90597ade77` (origin/dev tip), from real
json-report artifacts pulled with `gh run download`. The last core shards
that actually ran: runs 34185952279 and 34156968562 (2026-09-08 push),
34795954804 and 34859834173 (2026-09-14 push). Everything later is
backup-only dispatch (core skipped).

**Measurement caveat that shapes all of this: shards killed by
`timeout-minutes` upload NO artifact** — the `if: always()` upload step
never runs after a timeout cancellation. In run 34795954804 shards 0 and 5
were cancelled at exactly 120.27/120.25 min; in 34859834173 shards 2 and 3
at 120.7 min. Their four `core-test-results-*` artifacts do not exist.
Measurement therefore comes from the sibling shards that finished.

### Current timeout behavior (answers the task's original complaint)

Even after six-way sharding (TASK-21411), **2 of 6 core shards still hit
the 120-minute ceiling per run as of 2026-09-14** and die with no summary;
the other four complete in 89-115 min (09-08 era walls: 5353/5811/6820/6880
s; the one complete 09-14 shard: 5090 s for 11245 tests).

### Measured slowest tests (run 34185952279, complete shards 0/1/3/4; stage = setup+call+teardown)

| seconds | test | classification then |
|--------|------|---------------------|
| 12 x ~301 | `Tests/Chat/test_console_local_citation_boundary.py::test_citation_repair_*` (12 variants) | HANGS: each pinned at exactly call=300, the `--timeout` cap; all failed |
| 252 | `.../test_console_durable_turn_fix_round1.py::test_success_cleanup_drops_content_and_bounds_minimal_tombstones` | soak: 1000 sequential durable submits |
| 203/186/176/175 | `.../test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_rejects_*` / `..._and_sink_topology_are_unchanged` (111-123) | git-history walk, once per worker (lru_cache'd since 2026-08-11); siblings in-file are cheap |
| 126 | `.../test_summarization_diagnostic_privacy.py::test_manifest_boundary_rejects_new_generated_origin_dev_drift` | git-history drift gate |
| 109 | `.../test_terminal_resource_qualification.py::test_four_maximum_sessions_...` | by-design RSS qualification (incl. one deliberate `sleep(5)`) |
| 105/101 | ProductionApp root-state tests | real-app mounts; one had a 94 s teardown |

Slowest files by cumulative seconds: test_console_local_citation_boundary
3725 s (88 tests), test_installed_distribution 1352 s (150 subprocess
wheel-install tests), test_persistent_diagnostic_inventory 1134 s, then a
long tail of 200-600 s files.

**Classification: dominated by pathological hangs, not broad accretion.**
Across the four complete 09-08 shards the SUM of all call stages is
11,223 s; the 12 citation hangs alone burn 3,600 s — **32% of the entire
suite's test time** — and under `--dist loadscope` they strand one worker
for an hour while its siblings idle. Per-stage totals for the same four
shards: setup 1,303 s, call 11,223 s, teardown 35,296 s — teardown is the
largest bucket but is spread thin (only ONE test in the whole run had
>30 s teardown) and is runner-amplified: the same files show ~0.04 s/test
teardown locally under identical xdist flags, so it is 4-vCPU contention
on shared runners, not a fixable per-test pathology (TASK-19570's timed
pauses overlap here). The suite is also simply bigger than at filing:
~10-11k tests per shard x 6 shards vs the ~20.6k all-but-UI count in the
description.

### What was fixed, with before/after

1. The 09-08 citation hangs were already fixed upstream between 09-08 and
   09-14 (the 09-14 shard-1 report shows this file's 17 collected tests
   all passing quickly). Not re-fixed here; verified from artifacts.
2. **Regression found at current base and fixed (commit 5b0445226b):**
   the Console hook-consent send gate (`aed1b13501`, 2026-09-27 — three
   days before this session, and AFTER the last completed core run, so CI
   has never executed it) reads the saved `[hooks]` section through the
   guarded config loader on every `submit_draft`/`queue_prompt`. Under
   the per-test sandbox that read raises
   `RecoveryRequired("raw_source_selection_changed")`; the gate's
   fail-closed `except` denies every send ("Hooks unavailable..."), and
   the citation suite's controlled-gateway tests wait forever for
   provider events that never come — reintroducing exactly the 300 s
   hang class this task measures. Fix: added
   `test_console_local_citation_boundary.py` to the established
   `keep_bootstrap_profile` set in `Tests/conftest.py` (default
   `[hooks]` template has no rows, so `requires_authority` is False and
   sends are admitted). Full file, `--timeout=60`, same machine:
   before **87 failed, 8 passed, 1312.5 s** (17 tests pinned at the cap)
   plus a post-session interpreter-exit wedge on non-daemon persistence
   workers (~13 more minutes, had to be SIGKILLed — on CI this variant
   alone can eat a shard to the ceiling); after **2 failed, 93 passed,
   80.0 s** with a clean exit. The 2 remaining reds
   (`test_citation_repair_stop_while_checking_cancels_before_dispatch`
   [direct]/[agent]) are a separate fast-failing regression — a second
   submit during citation checking is accepted when the test expects it
   blocked — not duration-related, and left for its owner.

### AC #4 analysis (flags as they exist today)

`timeout_method` is deliberately unset (TASK-22062, pyproject.toml
comment): pytest-timeout selects `signal` wherever SIGALRM exists — both
CI runner platforms for the core job — falling back to `thread` only on
Windows, and no config anywhere sets `thread` (repo grep: comments only).
With `signal`, a hung test fails at 300 s and the xdist worker survives,
so `--max-worker-restart=3` no longer multiplies wall-time off per-test
hangs; restarts remain possible only for crash/OOM-class deaths, which
re-dispatch only the dead worker's unreported queue. Empirical
confirmation from the artifacts: the 12 capped hangs in run 34185952279
are recorded as ordinary failures while their workers completed ~10k
tests each, and all six complete shard reports contain **zero duplicate
nodeids** — no silent scope re-runs. The workflow's `PYTEST_TIMEOUT=300`
env var duplicates the `--timeout=300` flag (harmless).

### Remedy decision and what is left to the owner

Chosen remedy: fix the measured pathologies (the two waves above).
NOT chosen, and why:

- Ceiling/shard changes are outside this task's lane (TASK-22250 owns the
  matrix), and the test.yml comments already treat 120 min as 4x margin
  over the ~32 min/shard estimate. The 09-14 overruns on 2/6 shards are
  driven by runner teardown amplification + by-design gates, which test
  fixes cannot remove; whether to accept occasional 120-min shard kills,
  resize the matrix, or chase teardown amplification is an owner call.
- The 252 s soak (`test_success_cleanup...`) loops 1000 durable turns
  against `DURABLE_TOMBSTONE_CAP = 128` — ~8x overshoot beyond what its
  cap-saturation assertions need; 160 turns would cut ~210 s/run.
  Documented rather than changed: that file is currently red at base from
  the same hook-gate regression (it shares `_controller` with
  `test_console_first_send_atomicity`, imported by ~10 Chat suites, none
  of which CI has run since the gate landed), so a loop-count change
  could not be honestly before/after-verified today.

**Owner follow-ups out of this task's scope:** (1) the hook-consent gate
denies every sandboxed controller send suite it was not adapted for —
besides the citation file fixed here, at minimum the
`test_console_first_send_atomicity._controller` importers
(durable_turn_fix_round1/3, dispatch_recovery_fix_round1/2/3,
close_during_durable_postcommit, trace_first_send/trace_current_turn,
send_gate_queue_race (that one also has an unrelated `close_session`
API-drift red), durable_commit_offload, UI dispatch_recovery_fix_round2)
fail fast on "Hooks unavailable" at base; the gate's blanket
`except Exception -> deny` deserves its own decision. (2) The two
send-during-checking citation reds above. (3) The exit-wedge class:
timeout-killed tests leave non-daemon persistence workers that can hang
interpreter shutdown.

ADR required: no — test-suite admission-profile fix and CI measurement;
no storage/sync/interface/security decision. (The hook-gate follow-up may
warrant one when taken.)

Files changed: `Tests/conftest.py` (+13), this task file.
Branch: `fix/task-19425-core-suite-durations`, commits 2c3bfae166,
5b0445226b. Not pushed (owner policy).
