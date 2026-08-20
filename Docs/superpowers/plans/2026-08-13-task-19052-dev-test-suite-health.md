# TASK-19052 Latest-dev Test-Suite Health Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every test failure and error on a pinned latest `dev`, repair each actionable discovered root cause without hiding coverage, and merge one focused follow-up PR into `dev`; leave the existing Console size ratchet to TASK-3070's approved extraction series.

**Architecture:** A dependency-free checkpoint harness freezes pytest collection, executes disjoint bounded chunks, and writes an authoritative terminal-outcome ledger outside disposable worktrees. Repairs are derived only from that ledger, grouped into small RED-GREEN commits, and verified first at the triggering node or sequence and then against the exact discovered-failure set. The inherited Console size ratchet remains collected and delegated to TASK-3070.

**Tech Stack:** Python 3.11+, pytest 8, standard-library `subprocess`/`json`/`hashlib`/`unittest`, existing Ruff/mypy/compile/generated checks, Git/GitHub CLI, Backlog.md CLI.

---

## Scope and file map

In this plan, derive `TASK16073_EVIDENCE_ROOT` at runtime from the shared Git directory:

The `TASK16073_*` harness protocol and ignored evidence namespace are retained as
immutable internal identifiers from before latest `dev` claimed TASK-16073. The
human-facing work item is TASK-19052; changing the sealed protocol identifiers would
invalidate the reviewed harness evidence without improving runtime isolation.

```bash
TASK16073_GIT_COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"
TASK16073_MAIN_ROOT="$(dirname "$TASK16073_GIT_COMMON")"
TASK16073_EVIDENCE_ROOT="$TASK16073_MAIN_ROOT/.superpowers/sdd/2026-08-13-task-16073-dev-test-health"
```

Resolve and record that absolute value only as the value in each generation's
permission-restricted `provenance/evidence_root.json`, require it to be outside the
disposable task worktree, and never commit the resolved machine path. The privacy scan
allowlists that exact field/value and rejects the path in every other evidence or
repository file.

- Create, outside the worktree and under `TASK16073_EVIDENCE_ROOT`:
  - `TASK16073_EVIDENCE_ROOT/harness/checkpointed_pytest.py` — collection, chunk execution, resume validation, recursive coverage split, sequence minimization, and completeness verification.
  - `TASK16073_EVIDENCE_ROOT/harness/pytest_outcome_plugin.py` — exact pytest node/phase/session JSONL records.
  - `TASK16073_EVIDENCE_ROOT/harness/test_checkpointed_pytest.py` — standard-library harness contract and mutation tests.
  - Immutable `generations/<generation-id>/...` directories — fingerprints, collection manifest, chunk slices, logs, JUnit, ledgers, classifications, and hashes.
- Create after baseline classification:
  - `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md` — sanitized red-outcome inventory only; no raw traces, passing-node rows, secrets, or private paths.
- Modify only as baseline outcomes justify:
  - Exact production/test/config files named by each classified repair cluster.
- Modify at closeout:
  - `backlog/tasks/task-19052 - Restore-latest-dev-test-suite-health.md`
  - This plan document.
- Do not add a runtime or test dependency. Do not commit the raw harness/evidence tree.

## Required working rules

- Use `@superpowers:test-driven-development` for every repair.
- Use `@superpowers:systematic-debugging` before classifying or changing any failing cluster.
- Use `@ponytail` to keep each repair at the smallest shared root cause.
- Use `@superpowers:verification-before-completion` before every success claim, commit, PR update, and merge.
- Use `@superpowers:receiving-code-review` for Qodo, CI, and human review comments.
- Source and tests are frozen while any complete pipeline generation is running.
- No baseline red outcome may become skipped, xfailed, deselected, uncollected, removed, or renamed without explicit user approval and an AC amendment.

### Approved verification amendment (2026-08-20)

The user requested the minimal path and tests limited to touched functionality. Final
verification therefore runs the exact 107 discovered nodes, mapping two upstream-renamed
nodes to their current equivalents, plus directly affected checks. The only accepted red
is the unchanged `chat_screen.py` size ratchet owned by TASK-3070.5 through TASK-3070.11;
the budget must not be raised and this PR must not absorb that seven-PR decomposition.

## ADR check

ADR required: yes.

ADR path: `backlog/decisions/072-checkpoint-harness-process-ownership.md`.

Reason: Task 2 negative testing exposed a Darwin process-ownership boundary. ADR-072 records the approved cooperative-subprocess limitation, fail-closed capability gate, and PID-version-safe cleanup; application runtime boundaries remain unchanged.

### Task 1: Pin the newest `dev` and prepare an isolated generation

**Files:**
- Modify: `backlog/tasks/task-19052 - Restore-latest-dev-test-suite-health.md`
- Evidence: `TASK16073_EVIDENCE_ROOT/generations/<id>/provenance/`

- [ ] **Step 1: Verify ownership and cleanliness**

Run:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

Expected: branch is `codex/task-18912-dev-test-health`; tracked worktree is clean before rebase.

- [ ] **Step 2: Fetch and rebase onto the newest `origin/dev`**

Run:

```bash
git fetch origin
git rebase origin/dev
```

Expected: rebase succeeds without dropping the approved design/task history. If conflicts occur, resolve only after comparing both contracts, then rerun this task from Step 1.

- [ ] **Step 3: Record exact provenance**

Record SHA, Python executable/version, pytest version, platform, locale, explicit PATH, dependency declaration hashes, sorted `pip freeze --all` hash, normalized pytest arguments, and sanitized environment keys in immutable JSON/text files.

Run:

```bash
../../.venv/bin/python -V
../../.venv/bin/python -m pytest --version
../../.venv/bin/python -m pip freeze --all
git rev-parse HEAD
git status --short
```

Expected: all commands succeed; no credential value or real profile/config content enters evidence.

- [ ] **Step 4: Build the scratch profile boundary**

Create one permission-restricted generation-owned scratch root and set explicit `HOME`, `USERPROFILE`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `TLDW_CONFIG_PATH`, `TLDW_TEST_CONFIG_ROOT`, `TMPDIR`, `TLDW_TEST_MODE=1`, and scratch `[paths].data_dir`. Remove ambient `PYTEST_ADDOPTS`, plugin/xdist selectors, live-test gates, and credential/key/token variables.

Expected: real profile/config fingerprints are byte-identical before and after every generation.

### Task 2: Build and prove the checkpointed pytest evidence harness

**Files:**
- Create: `TASK16073_EVIDENCE_ROOT/harness/checkpointed_pytest.py`
- Create: `TASK16073_EVIDENCE_ROOT/harness/pytest_outcome_plugin.py`
- Create: `TASK16073_EVIDENCE_ROOT/harness/test_checkpointed_pytest.py`

- [ ] **Step 1: Write RED harness contract tests**

Use `unittest` with tiny synthetic pytest projects. Cover:

```python
CASES = (
    "missing_node",
    "duplicate_node",
    "corrupt_report",
    "collection_error",
    "mid_run_interrupt",
    "post_summary_hang",
    "sequence_dependent_timeout",
    "false_complete_marker",
    "process_containment_unavailable",
    "retained_sentinel_late_exec",
    "stale_audit_token",
    "forged_or_truncated_census",
)
```

Each test must assert either `generation_complete is False` or one exact owned red collector/process outcome. The sequence test must prove that removing a required predecessor makes the trigger disappear.

- [ ] **Step 2: Run the harness tests and capture RED**

Run:

```bash
../../.venv/bin/python -B \
  "$TASK16073_EVIDENCE_ROOT/harness/test_checkpointed_pytest.py" -v
```

Expected: failures identify missing harness/plugin behavior; production repository files remain unchanged.

- [ ] **Step 3: Implement the minimum outcome plugin and collector records**

Implement pytest hooks that append JSONL records containing schema version, record type, `nodeid` or collector ID, phase, pytest outcome, `wasxfail`, longrepr category only, worker/process identity, deselected node IDs, collection finish counts, and session exit status. `pytest_collectreport` owns collector pass/fail/skip records; `pytest_deselected` owns exact deselected IDs; `pytest_runtest_logreport` owns setup/call/teardown records; `pytest_sessionfinish` owns the pytest exit code. Flush and `fsync` after every record. Never serialize traceback bodies, fixture bodies, env values, or user data.

Reduce phase records deterministically to exactly one node terminal status:

1. setup or teardown failure → `error` (retain all failing phase categories);
2. otherwise failed call → `failed`;
3. otherwise skipped call with `wasxfail` → `xfailed`;
4. otherwise passed call with `wasxfail` → `xpassed`;
5. otherwise any setup/call skip → `skipped`;
6. otherwise passed call with successful setup/teardown → `passed`;
7. any impossible/missing phase combination → corrupt/incomplete, never inferred green.

The baseline non-executed set is the exact `skipped` + `xfailed` + deselected ledger. On a later manifest, any missing baseline node is an explicit `not_collected` manifest-diff outcome unless the user approved its removal/rename.

- [ ] **Step 4: Implement the concrete harness CLI, collection, and bounded execution**

Implement these exact modes:

```text
checkpointed_pytest.py collect --repo <worktree> --python <venv-python> --generation <dir>
checkpointed_pytest.py run --generation <dir> --max-nodes 250 --deadline-seconds 900
checkpointed_pytest.py resume --generation <dir>
checkpointed_pytest.py verify --generation <dir> [--baseline-generation <dir>]
checkpointed_pytest.py minimize --generation <dir> --process-outcome <id>
```

The functional pytest argument vector is exactly empty (`[]`), matching `python -m pytest` with no selectors. The harness records that vector and adds only evidence/control arguments: collection uses `--collect-only -q -p pytest_outcome_plugin`; execution uses each frozen node slice plus `-vv --tb=short -p pytest_outcome_plugin --junitxml=<immutable-path>`. `PYTHONPATH` contains only `TASK16073_EVIDENCE_ROOT/harness` plus the recorded repository environment. Per-generation `generation.json` is the sole input to `run`, `resume`, `verify`, and `minimize` and contains repo, interpreter, functional args, sanitized environment, manifest hash, harness hashes, and immutable output root.

The harness must also:

- collect once with explicit normalized args and parse exact node IDs;
- hash the ordered manifest;
- create disjoint chunks capped at 250 nodes and a 15-minute default deadline;
- run chunks serially with `-vv --tb=short`, JUnit, immutable stdout/stderr, and the outcome plugin;
- validate exact expected-versus-observed nodes before writing a complete marker;
- resume only when command, source/environment generation, expected hash, outcome hash, exit status, and marker all verify.

- [ ] **Step 5: Implement abnormal-process and pytest-exit ownership**

For each chunk, record exactly one process outcome keyed by chunk ID. Pytest exit 0 or 1 may be structurally complete only when every expected node reduced exactly once, no unexpected node/collector exists, and session/process records agree. Exit 2 (interrupted), 3 (internal error), 4 (usage error), 5 (no tests for a nonempty slice), signal termination, deadline termination, missing session record, post-summary hang, extra/missing node, corrupt JSONL, or conflicting duplicate phase is an owned red process outcome. Exit 0 with red node reports is corrupt; exit 1 without at least one red node/collector report is corrupt.

Implement ADR-072's process-ownership boundary. Supported test subprocesses retain at
least one observable ownership signal; deliberate removal of all signals is out of
scope. On Darwin create one private attempt-scoped regular-file sentinel, pass its
descriptor only to the pytest root, census surviving holders with hard-gated `libproc`,
bind every signal to PID-version identity through audit-token signaling, and require
ownership verification between two identical identity reads plus two successful,
non-truncated empty full censuses after root exit. The real preflight must exercise the
exact census calls, flavor 17's pinned 56-byte result, valid audit-token signaling,
mutated-pidversion `ESRCH` rejection, and probe disappearance. Missing symbols, wrong
sizes, permission failure, truncation, stale-token acceptance, or ownership uncertainty
is the exact red
`process_containment_unavailable` outcome and must occur before pytest launch when the
preflight itself is unsupported. Never fall back to bare PID or PGID signaling.

RED/GREEN tests must cover unsupported preflight, a retained-sentinel late `atexit`
fork with `setsid` and minimal-environment exec, stale-token rejection, and forged or
truncated census. A mutation replacing audit-token signaling with bare PID/PGID
signaling must fail the exact contract test.

On interruption, timeout, or post-summary hang, retain the harness-owned process outcome; recursively split for remaining coverage and delta-minimize the original ordered triggering slice. A single-node or minimized-sequence timeout is red but completely attributed, never silently green. Deselect records are owned by the collection ledger; final manifest absences are owned `not_collected` outcomes; neither can vanish through chunk execution.

- [ ] **Step 6: Run harness tests to GREEN**

Run the Step 2 command.

Expected: all contract cases pass and false completeness is rejected.

- [ ] **Step 7: Mutation-check sequence minimization and completeness**

Temporarily bypass one predecessor in the sequence minimizer and permit one missing-node marker.

Expected: the corresponding contract tests fail for the exact reason. Restore both mutations and rerun GREEN.

- [ ] **Step 8: Hash and review the uncommitted harness**

Record SHA-256 for all three harness files and their test output in the generation manifest. Confirm the resolved absolute `TASK16073_EVIDENCE_ROOT` is ignored, outside `.worktrees/`, and permission-restricted.

### Task 3: Capture the complete latest-dev baseline

**Files:**
- Evidence: `TASK16073_EVIDENCE_ROOT/generations/baseline-<sha>/`
- Create after completion: `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md`

- [ ] **Step 1: Freeze source and collect with the concrete interface**

Run:

```bash
../../.venv/bin/python \
  "$TASK16073_EVIDENCE_ROOT/harness/checkpointed_pytest.py" \
  collect --repo "$PWD" --python "$PWD/../../.venv/bin/python" \
  --generation "$TASK16073_EVIDENCE_ROOT/generations/baseline-<sha>"
```

Expected: collection has no unowned errors and produces a hashed ordered manifest. If collection is red, record it in the cumulative inventory, execute Tasks 4 and 5 for that collector cluster, start a new immutable baseline generation, and repeat this step until collection is structurally complete; never continue execution from the failed collection generation.

- [ ] **Step 2: Execute every manifest chunk**

Run:

```bash
../../.venv/bin/python \
  "$TASK16073_EVIDENCE_ROOT/harness/checkpointed_pytest.py" \
  run --generation "$TASK16073_EVIDENCE_ROOT/generations/baseline-<sha>" \
  --max-nodes 250 --deadline-seconds 900
../../.venv/bin/python \
  "$TASK16073_EVIDENCE_ROOT/harness/checkpointed_pytest.py" \
  verify --generation "$TASK16073_EVIDENCE_ROOT/generations/baseline-<sha>"
```

If interrupted, use the exact `resume` mode; if a process red needs attribution, use the exact `minimize` mode before final verification. Continue until every collected node has exactly one terminal outcome and every collector/process red outcome has exactly one owner.

Expected: no missing, duplicate, corrupt, or unowned outcome. Preserve exit-1 chunks as complete when all expected node outcomes were recorded.

- [ ] **Step 3: Verify privacy and profile isolation**

Scan raw evidence for credential canaries and real home/profile paths. Allow the
resolved evidence-root path only in the exact
`provenance/evidence_root.json` field/value; reject it in logs, JUnit, ledgers,
manifests, reports, and committed files. Compare pre/post real-profile fingerprints.

Expected: no real secret or private path; fingerprints identical. Synthetic canaries may exist only in ignored raw evidence.

- [ ] **Step 4: Create the sanitized red inventory**

For each red outcome record only: stable outcome ID, exact repository-relative node/collector/sequence, category pending, minimal sanitized symptom, baseline SHA, evidence hashes, and later repair commit. Do not copy traceback bodies or passing-node rows.

- [ ] **Step 5: Commit the baseline inventory**

```bash
git add Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md
git commit -m "test: inventory latest-dev suite failures"
```

Expected: one documentation-only commit; raw evidence remains ignored.

### Task 4: Classify every baseline red outcome

**Files:**
- Modify: `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md`
- Read as authority where applicable: `backlog/tasks/`, `backlog/decisions/`, maintained `Docs/`, tests and production seams named by each outcome.

- [ ] **Step 1: Diagnose one cluster at a time**

For each node/collector/process sequence, run the exact node alone, its smallest state-sharing file/suite, and the original order/minimized sequence. For alternating outcomes, retain enough identical repetitions to reproduce the baseline.

- [ ] **Step 2: Assign one supported category**

Use only: product defect, stale test contract, order/isolation defect, flake/race, or environment harness defect. A stale-contract classification must cite a task/ADR/maintained document or explicit owner decision predating baseline.

- [ ] **Step 3: Pause ambiguous contracts**

If evidence does not determine the intended behavior, stop only that cluster and ask the user. Do not rewrite the test to match current production behavior.

- [ ] **Step 4: Group shared root causes**

Map every red outcome to exactly one repair cluster. Prove every changed hunk planned for the PR maps to a baseline red outcome; exclude opportunistic cleanup.

- [ ] **Step 5: Commit the classified inventory**

```bash
git add Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md
git commit -m "docs(test): classify latest-dev suite failures"
```

### Task 5: Repair each classified cluster with RED-GREEN commits

**Files:**
- Modify/Test: exact files named by the cluster inventory; do not pre-authorize other files.
- Modify: `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md`

Repeat these steps independently for every repair cluster:

- [ ] **Step 1: Preserve the discriminating RED**

Run the exact baseline node, collector failure, or minimized ordered sequence and retain its failing evidence. If coverage is insufficient, add the smallest regression first and show it fails against the pinned baseline/root-cause mutant.

- [ ] **Step 2: Implement the smallest shared fix**

Prefer an existing contract/helper, production-shaped fixture, deterministic event/owner observation, and standard library. Do not add retries, sleeps, dependencies, or generalized frameworks unless the recorded root cause makes them unavoidable.

- [ ] **Step 3: Run focused GREEN**

Run the original trigger and directly affected file/subsystem suite. Expected: exact red outcome is gone without changing the baseline non-executed set.

- [ ] **Step 4: Prove non-vacuity**

Temporarily remove/invert the guard, state reset, ordering boundary, or assertion that embodies the fix. Expected: the discriminating regression fails. Restore and rerun GREEN.

- [ ] **Step 5: Repetition-check flakes and races**

Run the original triggering node/sequence at least the baseline reproduction count and at least 20 times for a low-rate flake. Expected: all attempts pass with identical source/environment.

- [ ] **Step 6: Run scoped static/generated checks**

Run Ruff format/check, py_compile, targeted mypy, diagnostic inventory, CSS/generated sync, or other family-specific checks only where the cluster touched those owners. Classify inherited failures against the pinned clean baseline; never silently ignore a new error.

- [ ] **Step 7: Update inventory and commit the cluster**

Record category, authority/evidence, RED-GREEN/repetition results, files, commit SHA placeholder, and affected tests.

```bash
git add <exact-cluster-files> Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md
git commit -m "fix(<scope>): <root-cause summary>"
```

### Task 6: Verify the frozen candidate with the complete pipeline

**Files:**
- Evidence: `TASK16073_EVIDENCE_ROOT/generations/candidate-<sha>/`
- Modify: `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md`

- [ ] **Step 1: Freeze the candidate and refresh provenance**

Expected: clean worktree; installed-package hash equals the generation's clean-dev baseline. Dependency drift starts a new clean environment/generation.

- [ ] **Step 2: Recollect and compare manifests**

Expected: every baseline node remains present; only explicit added regression nodes are new. No removed/renamed/non-collected baseline node without user-approved AC amendment.

- [ ] **Step 3: Run the full checkpointed pipeline**

Run the same concrete `collect`, `run`, and `verify --baseline-generation <baseline-dir>` interface from Task 3 with a new candidate generation. Expected: every collected node has exactly one terminal outcome; zero failures, errors, collector errors, timeouts, abnormal exits, or post-summary hangs; skip/xfail/deselect/not-collected set matches baseline.

- [ ] **Step 4: Run cumulative static and privacy checks**

Run all applicable touched-family checks, `git diff --check origin/dev...HEAD`, evidence privacy scan, and real-profile fingerprint comparison.

- [ ] **Step 5: Update the report with final counts and hashes**

Record exact SHA, collection/outcome hashes, counts, duration, commands, package hash, and profile-fingerprint equality.

### Task 7: Independent branch review

**Files:**
- Review: `origin/dev...HEAD`
- Modify only if findings require it: exact affected files and report.

- [ ] **Step 1: Request independent correctness/security/test-quality/YAGNI review**

Reviewers must trace each changed hunk to a baseline red outcome, inspect assertions for weakening, inspect skip/xfail/collection deltas, and verify privacy/evidence claims.

- [ ] **Step 2: Resolve every supported finding RED-first**

Use `@superpowers:receiving-code-review`; verify the finding locally before changing code. Commit focused corrections separately.

- [ ] **Step 3: Invalidate and rerun evidence after source/test edits**

Any source/test correction requires affected checks and a new complete candidate generation. Documentation-only accuracy corrections require their direct checks and report hash refresh only.

### Task 8: Rebase onto the newest `dev` and prove the ready-PR head

**Files:**
- Evidence: immutable `clean-dev-<sha>/` and `ready-pr-<sha>/` generations.
- Modify as newly discovered dev outcomes require: report plus exact cluster files.

- [ ] **Step 1: Fetch and rebase onto latest `origin/dev`**

```bash
git fetch origin
git rebase origin/dev
```

- [ ] **Step 2: Establish the new clean-dev generation**

If rebase changes the base, run the complete checkpointed pipeline on exact clean `origin/dev` in a disposable worktree/environment with the same package hash. Any newly introduced collector/node/process red outcome is appended to the cumulative inventory, classified with Task 4, repaired with Task 5 on the rebased branch, and verified with Task 6. Recollect after each collection repair. Repeat this explicit Task 4 → Task 5 → Task 6 loop until both the clean-dev delta and rebased branch are fully accounted for.

- [ ] **Step 3: Run affected checks and the exact discovered-node set on rebased HEAD**

Expected: 106 passes and only the unchanged TASK-3070 size-ratchet red across the exact
107-node set, with directly affected checks green.

- [ ] **Step 4: Rerun independent review if the rebase required source/test changes**

Expected: no unresolved Critical/Important/Minor finding.

### Task 9: Open the review PR while TASK-19052 remains In Progress

**Files:**
- Modify: PR title/body and remote branch only.

- [ ] **Step 1: Push and create one ready PR against `dev`**

Keep TASK-19052 In Progress and ACs unchecked until review/rebase/final-head verification is complete. The PR body must include the failure-cluster table, baseline/current-candidate generation hashes and counts, commands, static checks, privacy statement, and any explicit limitations.

- [ ] **Step 2: Verify the posted PR state**

Run GitHub CLI queries for base/head SHAs, mergeability, required checks, reviews, Qodo comments, and unresolved inline threads.

Expected: base is `dev`, head is this task branch, and no feedback is assumed resolved merely because a check is pending.

### Task 10: Address Qodo/CI/reviewer feedback and perform the final rebase

**Files:**
- Modify only as supported review/CI findings require.

- [ ] **Step 1: Inspect every check and review thread**

Use GitHub CLI to read PR checks, Qodo inline/general comments, and human review threads. Treat external CI providers as links unless their logs are available through the approved interface.

- [ ] **Step 2: Verify each finding before acting**

For each supported issue, reproduce it, add/preserve a focused RED, apply the smallest fix, rerun affected checks, and commit separately. Explain and resolve unsupported suggestions with evidence rather than changing code blindly.

- [ ] **Step 3: Invalidate stale evidence after any source/test change**

Every supported source/test review or CI fix runs its affected checks and then a new Task 6 complete candidate generation. Documentation-only accuracy fixes rerun their direct checks and refresh only affected report hashes.

- [ ] **Step 4: Rebase once more after review fixes**

Fetch/rebase latest `dev`, then run affected checks and the exact discovered-node set. Do
not proceed until required checks and Qodo/human findings are resolved and no red exists
beyond the unchanged TASK-3070 size ratchet.

### Task 11: Complete task hygiene on the reviewed, rebased source/test tree

**Files:**
- Modify: `backlog/tasks/task-19052 - Restore-latest-dev-test-suite-health.md`
- Modify: `Docs/superpowers/plans/2026-08-13-task-19052-dev-test-suite-health.md`
- Modify: `Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md`
- Modify if genuinely new knowledge exists: applicable `backlog/docs/lessons-*.md`

- [ ] **Step 1: Finish task notes truthfully**

Check ACs only after the reviewed/rebased executable-input candidate is green. Add concise Implementation Notes with baseline/candidate SHAs, the hash of every executable/test/collection/config/dependency/harness input, stable final generation name `ready-pr-final`, outcome inventory, authority citations, RED-GREEN/mutation/repetition evidence, candidate suite/static counts, review, privacy, lessons, and ADR decisions. State that `ready-pr-final` is run after this documentation-only commit, its actual exact-head metrics live in the immutable manifest and PR evidence, and any mismatch reopens the task.

- [ ] **Step 2: Mark Done safely and verify readback**

Use file-safe editing for the five-digit task ID, then run:

```bash
backlog task 19052 --plain
git diff --check origin/dev...HEAD
git status --short
```

- [ ] **Step 3: Commit and push closeout documentation**

```bash
git add "backlog/tasks/task-19052 - Restore-latest-dev-test-suite-health.md" \
  Docs/superpowers/plans/2026-08-13-task-19052-dev-test-suite-health.md \
  Docs/superpowers/reports/2026-08-13-task-19052-dev-test-suite-health.md
git commit -m "docs(test): complete latest-dev suite restoration"
git push
```

### Task 12: Verify the exact final PR head and merge

**Files:**
- Evidence: `TASK16073_EVIDENCE_ROOT/generations/ready-pr-final/`
- Modify repository files only if this gate fails; a failure reopens TASK-19052.

- [ ] **Step 1: Freeze exact post-closeout HEAD**

Record HEAD/base/package/environment/harness hashes. Prove the executable-input hash is byte-identical to Task 11's reviewed candidate. No repository change is allowed after this point unless the task returns to In Progress.

- [ ] **Step 2: Run the exact discovered-node set on executable HEAD**

Expected: all 106 actionable nodes pass, the unchanged TASK-3070 size ratchet is the
sole red, both upstream rename mappings pass, and directly affected/static checks are
green.

- [ ] **Step 3: Handle any mismatch without papering it over**

If the exact-head pipeline, required check, or review state differs from the documented expectation, set TASK-19052 back to In Progress, classify and repair via Tasks 4–6, repeat Tasks 10–12, and do not merge.

- [ ] **Step 4: Update only external PR evidence and merge**

Post exact `ready-pr-final` counts/hashes/duration, command/environment fingerprint, manifest location/hash, and HEAD SHA to the PR without changing repository files. Re-query checks, mergeability, Qodo/human threads, and head SHA. Merge with the repository-preferred strategy only when all are clear; verify the resulting merge commit is reachable from `origin/dev`, then retain evidence through merge verification and clean up only task-owned disposable worktrees/processes.
