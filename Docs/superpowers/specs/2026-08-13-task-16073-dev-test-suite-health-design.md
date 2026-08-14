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
2. Record the exact commit SHA, Python executable and version, pytest version,
   platform, hashes of dependency declarations/locks, and a sorted `pip freeze --all`
   hash.
3. Verify the worktree is clean.
4. Create one permission-restricted task-owned scratch root containing `HOME`,
   `USERPROFILE`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`,
   `TLDW_CONFIG_PATH`, `TLDW_TEST_CONFIG_ROOT`, temporary files, and a scratch
   `[paths].data_dir`. Set `TLDW_TEST_MODE=1`.
5. Fingerprint relevant real profile/config paths before and after the run and require
   byte-identical results. Existing repository network-blocking fixtures remain active.
6. Start from a sanitized environment: unset ambient `PYTEST_ADDOPTS`, pytest plugin
   injection/xdist variables, live-test gates, and credential/key/token variables. Use
   `python -m pytest` from the recorded virtual environment and an explicit, recorded
   locale and `PATH`; never inherit arbitrary pytest selectors from the caller.

### Checkpointed complete-suite pipeline

The complete suite is not one pytest process. A task-owned standard-library harness:

1. Runs frozen collection with `python -m pytest --collect-only`, records every exact
   node ID, and hashes the ordered manifest.
2. Slices that manifest into disjoint bounded chunks whose command lines stay below
   platform argument limits and whose expected duration is below 20 minutes.
3. Runs chunks serially with `-vv --tb=short`, a distinct immutable JUnit/log/outcome
   path, a bounded process deadline, and no fail-fast option.
4. Uses a small task-owned pytest evidence plugin to write exact `report.nodeid`, phase,
   and outcome records plus the session exit status. The harness/plugin source and
   self-test results are hashed into the evidence manifest.
5. Records collection reports and process/session outcomes alongside test reports. A
   collection error is a first-class red outcome keyed to its collector path; the
   frozen generation stops before execution, the collection defect is repaired
   RED-first, and a new generation recollects. The cumulative classified inventory
   retains the original collection outcome and its repair evidence.
6. Writes a complete marker only after the chunk's actual node/outcome set matches its
   expected slice exactly. Exit 0 or 1 may be complete. An interruption with uncovered
   nodes is incomplete and reruns only those uncovered nodes.
7. Treats an abnormal exit, timeout, or post-summary shutdown hang as a first-class red
   *process outcome* even when every node in the slice already reported. The harness
   reruns and delta-minimizes the entire ordered triggering slice (not merely uncovered
   nodes), preserving order, until it finds a one-minimal triggering sequence. A
   sequence-dependent result is retained as that bounded sequence rather than being
   forced into a false single-node attribution.
8. On an incomplete or timed-out multi-node chunk whose trigger is not yet known,
   recursively splits for coverage and separately minimizes the original ordered slice
   for its process outcome. A single-node or minimized-sequence child is terminated,
   recorded with one task-owned terminal outcome plus command/deadline/process evidence,
   and becomes an explicit baseline failure category; it is never silently dropped.
9. Resumes by skipping only chunks whose command, source/environment generation,
   expected-node hash, outcome hash, exit status, and complete marker all verify.

The terminal-outcome ledger is authoritative. It combines exact test-node reports,
collector reports, harness-owned timeout/interruption records, and process/session
outcomes. JUnit supplies failure/error detail but is not authoritative for terminated
tests or session failures. The coverage verifier must prove that every collected node
has exactly one terminal outcome and every collection/process red outcome is owned by
one classified inventory entry. All final-generation pytest processes must exit
normally. In the red baseline only, an exact single-node or minimized ordered-sequence
timeout/hang is a complete *failure outcome* when the harness records bounded
termination/minimization evidence; it is not a green or normal child exit. Missing,
duplicate, corrupt, unowned abnormal exit, or uncollected nodes keep the pipeline
incomplete.

Before the real sweep, negative harness self-tests inject a missing node, duplicate
node, corrupt report, collection error, mid-run interruption, post-summary hang,
sequence-dependent timeout, and false complete marker. Each must either be rejected as
incomplete or appear as the exact owned red outcome described above. The sequence
minimizer is mutation-checked by removing one required predecessor and requiring the
trigger to disappear.

The normalized pytest arguments and sanitized environment are recorded verbatim.
Per-phase output directories and per-chunk node slices differ by design; equality means
all other normalized pytest arguments and environment values are identical. Source and
tests remain frozen for an entire pipeline generation.

## Failure inventory and classification

The verified terminal-outcome ledger is authoritative. Every failure, error, timeout,
collection failure, and process/session outcome receives one of these labels:

- **Product defect:** the assertion describes the intended contract and production
  behavior violates it.
- **Stale test contract:** production behavior is intentionally established elsewhere
  and the test or fixture still models a retired contract.
- **Order/isolation defect:** the node passes alone but fails in its original module or
  suite order because state, workers, files, environment, or monkeypatching leak.
- **Flake/race:** the identical isolated node or original triggering sequence alternates
  outcomes without source changes.
- **Environment harness defect:** the test assumes unavailable network, port, process,
  clock, filesystem, or optional-runtime behavior that its own harness should control.

Classification evidence is bounded but substantive:

- Run the exact node alone.
- Run its containing file or smallest state-sharing suite.
- For an intermittent node, loop the identical node enough times to reproduce the
  failure rather than accepting one green rerun; retain the smallest original sequence
  and repetition count that reproduced it.
- For order-dependent failures, preserve the smallest preceding sequence that triggers
  the failure and prove the leak is gone.
- Compare identical commands and exact failure sets; never compare counts from different
  invocations.

“Stale test contract” is valid only with a cited authority that predates the baseline:
an accepted task, canonical ADR, maintained user/developer documentation, or explicit
owner decision. Current production behavior is not its own authority. An ambiguous
contract pauses that cluster for an owner decision instead of rewriting the test.

If several red outcomes share a root cause, they form one repair cluster. Unrelated
clusters remain separate commits inside the single requested PR. A sanitized committed
inventory maps only baseline failures, errors, timeouts, and collection/process/session
outcomes to category, authority/evidence, repair commit, and final verification; it
contains no passing-node rows, raw traceback bodies, or private paths. Full passing-node
coverage remains proven by the durable manifest/outcome hashes and aggregate counts.

## Repair rules

Every production or test change follows RED-GREEN:

1. Preserve or add the smallest discriminating regression.
2. Run it against the pinned baseline and observe the intended failure.
3. Fix the shared root cause using existing helpers and contracts.
4. Run the regression, its affected suite, and a mutation/non-vacuity probe where the
   fix adds a guard or concurrency boundary.
5. Commit one coherent repair cluster.

An intermittent repair must repeatedly pass the original triggering node/sequence for
at least the number of attempts that reproduced the baseline failure, with a minimum of
20 attempts for a low-rate flake. One isolated pass and one green complete sweep are not
sufficient evidence.

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
  real credentials, real profile paths, or private user data. Synthetic redaction
  canaries may intentionally appear in ignored raw failure evidence; their values must
  not enter persistent production diagnostics or committed inventories.

Final verification uses the same checkpointed pipeline, normalized pytest arguments,
and environment contract used for the baseline. It re-collects on the frozen candidate:
every baseline node must remain present, every added regression is included, and every
collected node receives exactly one terminal outcome. Removing or renaming a baseline
node requires explicit user scope amendment; a mapping note is not enough. Success
requires zero failures or errors and an unchanged skip/xfail/deselect/not-collected set.
No baseline failure/error may transition to a non-executed outcome without an explicit
user-approved spec/AC amendment.

After final suite success, run cumulative diff review, Ruff/formatter/type/compile
checks for touched files, applicable CSS/generated checks, and independent correctness,
security, test-quality, and YAGNI review. Any review-driven source or test change
invalidates affected and final-suite evidence and requires rerunning it.

## Pull-request workflow

- The branch is based on the newest `origin/dev` before the baseline and rebased again
  immediately before opening the PR. Every rebase starts a new pipeline generation:
  refresh the environment/package fingerprint, re-collect, add any new red nodes to the
  inventory, and run the complete checkpointed pipeline on the exact ready-PR head SHA.
  The installed-package hash must remain identical from a generation's clean-dev
  baseline through its final verification. Dependency drift requires a fresh isolated
  environment and a new complete clean-dev baseline generation (or an explicit
  user-approved scope amendment); it cannot make a prior red outcome disappear.
- Open one ready PR against `dev` with the pinned baseline and final JUnit summaries,
  failure-cluster table, exact test commands, static results, and known limitations.
- Address CI and reviewer/Qodo comments with focused regressions and separate commits.
  Every source/test-changing review or CI fix invalidates final evidence and requires
  affected checks plus the complete pipeline on the new head.
- Rebase onto the latest `dev` again after review fixes, rerun affected checks and the
  final complete suite, then merge only when all required checks and threads are clear.

## Privacy and evidence retention

Permission-restricted raw logs, JUnit reports, harness files, and manifests stay above
all disposable worktrees under the main repository's ignored `.superpowers/sdd/`
evidence root through PR merge and worktree cleanup. Baseline, repair, and final phases
use distinct immutable directories; no command overwrites earlier evidence. Artifacts
may contain synthetic fixture content and repository-relative node paths but must not
contain credentials, real user bodies, real profile/config contents, or private home
paths. The committed sanitized inventory and task notes record hashes, counts,
commands, classifications, and durations rather than raw traces.

## ADR decision

ADR required: no.

ADR path: N/A.

Reason: this task restores existing product and test contracts and does not introduce a
new storage, service, security, dependency, or long-lived UX boundary. If classification
reveals that such a decision is necessary, that cluster stops until this design and an
ADR are amended before implementation.
