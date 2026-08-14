# TASK-16073 Latest-dev Test-Suite Health Design

## Status

Proposed for implementation after review.

## Context

TASK-2703 exposed a large number of failure and error markers during a repository-wide
run, but that run was stopped before quiet-mode pytest emitted its terminal node list.
Only one exact node survived in pytest's cache. A marker count is not an actionable
failure inventory, and comparing it with a different command would be misleading.

The follow-up therefore starts from the latest `origin/dev`, captures one complete,
durable baseline, and fixes the exact failure set produced by that identical command.
The task is test-health work, not an invitation to add product features or refactor
unrelated code.

## Goals

- Produce an exact, durable failure inventory for one pinned latest-dev commit.
- Repair every reproducible failure and error from that inventory at its smallest
  shared root cause.
- Make order-dependent, flaky, and environment-sensitive tests deterministic without
  weakening valid product assertions.
- Finish with the identical complete-suite command reporting zero unexpected failures
  or errors.
- Open one follow-up pull request against `dev`, as explicitly requested.

## Non-goals

- No feature work, broad cleanup, dependency upgrades, or speculative abstractions.
- No deletion, skip, `xfail`, retry wrapper, or assertion weakening merely to turn a
  red node green.
- No attempt to infer node identities from TASK-2703's anonymous `F`/`E` markers.
- No committed raw pytest logs, profile data, credentials, or machine-specific paths.
- No promise to fix a failure that appears only after `dev` advances beyond the pinned
  baseline; such a delta must first be reproduced after rebasing and explicitly added
  to the recorded inventory.

## Pinned baseline and isolation

Immediately before the baseline:

1. Fetch `origin` and rebase the task branch onto the newest `origin/dev`.
2. Record the exact commit SHA, Python version, pytest version, platform, dependency
   lock metadata, and test command.
3. Verify the worktree is clean.
4. Create one task-owned scratch root containing `HOME`, `XDG_CONFIG_HOME`,
   `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `TLDW_CONFIG_PATH`, temporary files, and reports.
5. Fingerprint relevant real profile/config paths before and after the run and require
   byte-identical results. Existing repository network-blocking fixtures remain active.

The complete baseline command uses verbose node streaming plus JUnit output. Verbose
streaming means an interrupted run still names every node reached; JUnit provides the
complete machine-readable failure set when the run finishes. Shell pipe status must be
preserved, and stdout/stderr must be written under the ignored task evidence root.

Conceptually:

```text
pytest -vv --tb=short --junitxml=<evidence>/baseline/junit.xml
```

The precise environment and command are recorded verbatim in a manifest. Source and
tests remain frozen for the entire baseline run.

## Failure inventory and classification

The JUnit failure/error set is authoritative. Each node receives one of these labels:

- **Product defect:** the assertion describes the intended contract and production
  behavior violates it.
- **Stale test contract:** production behavior is intentionally established elsewhere
  and the test or fixture still models a retired contract.
- **Order/isolation defect:** the node passes alone but fails in its original module or
  suite order because state, workers, files, environment, or monkeypatching leak.
- **Flake/race:** identical isolated repetitions alternate outcomes without source
  changes.
- **Environment harness defect:** the test assumes unavailable network, port, process,
  clock, filesystem, or optional-runtime behavior that its own harness should control.
- **Expected optional absence:** only when an existing product contract explicitly
  makes the capability optional. This may use the repository's established skip
  mechanism, but TASK-16073 must not invent a new skip to hide a failure.

Classification evidence is bounded but substantive:

- Run the exact node alone.
- Run its containing file or smallest state-sharing suite.
- For an intermittent node, loop the identical node enough times to reproduce the
  failure rather than accepting one green rerun.
- For order-dependent failures, preserve the smallest preceding sequence that triggers
  the failure and prove the leak is gone.
- Compare identical commands and exact failure sets; never compare counts from different
  invocations.

If several nodes share a root cause, they form one repair cluster. Unrelated clusters
remain separate commits inside the single requested PR.

## Repair rules

Every production or test change follows RED-GREEN:

1. Preserve or add the smallest discriminating regression.
2. Run it against the pinned baseline and observe the intended failure.
3. Fix the shared root cause using existing helpers and contracts.
4. Run the regression, its affected suite, and a mutation/non-vacuity probe where the
   fix adds a guard or concurrency boundary.
5. Commit one coherent repair cluster.

Product code changes are allowed only when the test exposes a real product defect.
Fixture repairs must use real signatures and production-shaped payloads. Wait-based UI
repairs must observe the rendered widget or settled owner, not merely an upstream state
flag. Environment repairs must isolate the test; they must not weaken production
security, privacy, or failure behavior.

## Scope control

The PR may touch multiple subsystems because the user explicitly requested one
follow-up PR for the complete captured failure set. That does not authorize opportunistic
cleanup. Every changed hunk must map to at least one baseline node and its root-cause
record. The PR description groups commits and files by failure cluster.

If a repair would require a schema migration, a new dependency, a service contract,
security/privacy policy, or another architectural decision, implementation pauses for
an ADR/spec amendment before that cluster is changed.

## Verification

During repair:

- Every baseline failure/error node passes.
- Each directly affected file or subsystem suite passes.
- Static analysis and generated-artifact checks run for every touched file family.
- Persistent diagnostic inventory is refreshed only for owners actually changed.
- Privacy scans assert that reports and diagnostics contain no credentials, synthetic
  secret sentinels, real profile paths, or private user data.

Final verification uses the same pinned environment and the byte-identical complete
pytest command used for the baseline. The candidate is frozen during the run. Success
requires zero unexpected failures or errors; expected skips must be unchanged or
individually justified by a pre-existing optional-capability contract.

After final suite success, run cumulative diff review, Ruff/formatter/type/compile
checks for touched files, applicable CSS/generated checks, and independent correctness,
security, test-quality, and YAGNI review. Any review-driven source or test change
invalidates affected and final-suite evidence and requires rerunning it.

## Pull-request workflow

- The branch is based on the newest `origin/dev` before the baseline and rebased again
  immediately before opening the PR.
- Open one ready PR against `dev` with the pinned baseline and final JUnit summaries,
  failure-cluster table, exact test commands, static results, and known limitations.
- Address CI and reviewer/Qodo comments with focused regressions and separate commits.
- Rebase onto the latest `dev` again after review fixes, rerun affected checks and the
  final complete suite, then merge only when all required checks and threads are clear.

## Privacy and evidence retention

Raw logs and JUnit reports stay under ignored `.superpowers/sdd/` evidence. They may
contain synthetic fixture content and repository-relative node paths but must not
contain credentials, real user bodies, real profile/config contents, or private home
paths. The committed task notes record hashes, counts, commands, classifications, and
durations rather than copying raw traces.

## ADR decision

ADR required: no.

ADR path: N/A.

Reason: this task restores existing product and test contracts and does not introduce a
new storage, service, security, dependency, or long-lived UX boundary. If classification
reveals that such a decision is necessary, that cluster stops until this design and an
ADR are amended before implementation.

