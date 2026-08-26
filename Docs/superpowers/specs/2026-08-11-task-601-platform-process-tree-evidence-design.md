# TASK-601 Native Process-Tree Evidence Design

## Purpose

Close TASK-601 acceptance criterion 6 with reproducible native evidence that the
local STT worker and its preparation descendants are contained and terminated
before generation scratch cleanup on Linux, Windows, and macOS.

The process-tree implementation already exists. This work adds the missing native
Windows proof, makes the native cleanup contract portable across the three target
operating systems, and records results from one exact tested commit. It does not
change provider behavior, add a general subprocess framework, or broaden ordinary
CI.

## Scope

### In scope

- A real spawned worker and real sleeping descendant on each target operating
  system.
- Parent admission before the worker may launch descendants.
- Native termination of the whole contained tree.
- Proof that the worker and descendant are gone before generation scratch is
  removed.
- The crashed-leader case, where the parent still owns and terminates the surviving
  descendant.
- A label-gated and manually dispatchable GitHub Actions matrix for Linux,
  Windows, and macOS.
- One bounded, path-private JSON artifact per platform and one checked-in aggregate
  after all lanes pass the same executable commit.

### Out of scope

- General CI, the full test suite, model downloads, ONNX inference, FFmpeg media
  correctness, performance qualification, or architecture expansion.
- Production changes made preemptively. Production changes are allowed only if a
  genuine native test exposes a defect.
- TASK-602 model-platform evidence, TASK-603 dictation evidence, or TASK-605
  provider removal.

## Governing Decision

**ADR required:** no new ADR.

**ADR path:**
`backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` and
`backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`.

**Reason:** ADR-025 already requires decoder descendants to be terminated as a
platform process tree before temporary cleanup. ADR-041 does not alter that
ownership boundary. This design supplies the previously deferred native evidence;
it does not introduce a new architecture decision.

## Approaches Considered

### 1. Label-gated native contract matrix — selected

Run only the exact process-tree tests on one current GitHub-hosted runner for each
required operating system. Normalize the JUnit result into bounded JSON and upload
it even when a test fails.

This is the smallest approach that proves the real platform primitives while
keeping failures attributable to one narrow contract.

### 2. Add the tests to general CI — rejected

General CI is broader, slower, and can hide a containment failure among unrelated
suite failures. It would also make platform release evidence run on every change
when TASK-601 only needs an explicit release gate.

### 3. Write a separate native probe beside the tests — rejected

A second probe would duplicate process creation, admission, descendant tracking,
termination, and cleanup logic. The test must exercise the production containment
class directly; evidence infrastructure should only run and summarize that test.

## Native Test Contract

`Tests/STT/test_executor_process_tree.py` remains the single behavioral owner. The
existing POSIX-only native tests will be generalized without weakening their
assertions:

1. A spawned worker reports its native containment identity and cannot advance
   beyond its admission wait until the parent constructs `ExecutorProcessTree` and
   calls `admit()`.
2. After admission, the worker launches one real long-lived Python descendant and
   reports its PID.
3. `terminate_tree()` must return true only after both worker and descendant are
   absent.
4. Only after that proof may the test remove the generation scratch directory.
5. A second native case exits the worker leader while its descendant remains. The
   parent must still terminate the descendant and prove the tree empty before
   scratch cleanup.

On POSIX, the production path is the worker-owned session/process group and
`killpg`. On Windows, the production path is a real kill-on-close Job Object, real
worker assignment before admission, and inherited descendant membership. Existing
fake Win32 API tests remain as precise ordering/error tests; they do not count as
native Windows evidence.

All waits and cleanup are bounded. Test finalizers may kill only PIDs or process
groups created by that test. Failure to prove death leaves scratch in place and
fails the lane; the test must never report cleanup success while a descendant could
still be alive.

The portable liveness helper must not use `os.kill(pid, 0)` on Windows because
Python sends every value other than `CTRL_C_EVENT` and `CTRL_BREAK_EVENT`, including
zero, through unconditional `TerminateProcess`. POSIX may continue using signal
zero. Windows must open the exact PID with
`SYNCHRONIZE`, call `WaitForSingleObject(handle, 0)`, close the handle, and treat
only a signaled handle or `ERROR_INVALID_PARAMETER` from `OpenProcess` as proof of
exit. Any other Win32 error fails closed. This follows the existing repository
idiom in `Tests/Notes/test_git_process_containment.py`.

The required cross-platform evidence nodes will have stable IDs:

- `Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup`
- `Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup`
- `Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch`

The first two exercise native descendants through `ExecutorProcessTree`. The third
exercises the production `LocalSTTExecutor` force-stop and scratch-removal sequence
on every platform. Strengthen that existing controller test with a gated wrapper
around the real `terminate_tree()` call: while termination is deliberately blocked,
the test must observe that generation scratch still exists; after the wrapper calls
the real platform termination and retirement completes, scratch must be absent.
This proves the production ordering rather than merely observing the final state.
Together the three nodes prevent a compositional false positive where tree death
and controller cleanup are each proven, but never on the same target OS.
Platform-specific unit/contract nodes may also run, but cannot substitute for a
required node.

## Evidence Workflow

Create `.github/workflows/task-601-platform-evidence.yml` with:

- `workflow_dispatch` and a `pull_request` `labeled` trigger only;
- required label `task-601-platform-evidence`;
- read-only repository permissions;
- checkout of the exact PR head or the ref selected by the person invoking
  `workflow_dispatch`;
- Python 3.12;
- `fail-fast: false`;
- one lane each for `ubuntu-24.04`, `windows-2022`, and `macos-15-intel`;
- a bounded job timeout;
- the repository test dependencies and no STT model/runtime extra;
- one explicit pytest command covering the process-tree contract file, the exact
  existing controller scratch-ordering node, and the evidence normalizer tests; and
- unconditional JSON validation and artifact upload when the job reaches cleanup.

The workflow is not a substitute for general CI and is not triggered by push or
schedule. A red or timed-out lane is an open release gate, not a flaky result to
waive.

`tested_commit` is always `git rev-parse HEAD` after checkout, validated as a
40-character lowercase hexadecimal object ID. The workflow must not use
`GITHUB_SHA` as the evidence identity: for pull requests it may identify GitHub's
synthetic merge commit rather than the checked-out PR head. The workflow has no
separate free-form commit input; manual runs use GitHub Actions' built-in ref
selector, and checkout plus `git rev-parse HEAD` remains authoritative.

The dependency-install and pytest steps use `continue-on-error: true` only so the
workflow can normalize and upload failure evidence. The normalizer receives the
allowlisted GitHub step outcome (`success`, `failure`, `cancelled`, or `skipped`) in
addition to JUnit. It may write `status: passed` only when the pytest step outcome is
`success`, the JUnit document is well formed, no selected testcase failed, and all
three required nodes passed. The final validation step exits nonzero for every
other outcome, restoring the job's red status after artifact creation.

## Evidence Format

Add a small standard-library normalizer under `.github/scripts/`. It initializes a
failure document, parses the exact JUnit report after pytest, validates the bounded
schema, and writes JSON atomically. It does not implement containment or run a
second native probe.

Each matrix entry supplies an evidence name. The normalizer accepts only these
name/host pairs and derives the host side itself:

- `linux-x86_64` -> `platform.system() == "Linux"` and normalized machine
  `x86_64`;
- `windows-x86_64` -> `platform.system() == "Windows"` and normalized machine
  `x86_64` (including native `AMD64` spelling); and
- `macos-x86_64` -> `platform.system() == "Darwin"` and normalized machine
  `x86_64`.

A mismatched or unknown evidence name fails before a passing document can be
written. This binds the uploaded artifact name to the host that actually executed
the tests.

Each per-platform document contains only:

- schema version and passed/failed status;
- tested commit, workflow run ID, run attempt, and canonical GitHub Actions run URL;
- operating system, architecture, and Python version;
- the allowlisted required test node IDs and their passed/failed/skipped outcomes;
- bounded duration;
- stable failure stage/code when setup or test execution fails.

It excludes commands, exception text, tracebacks, environment variables, local
paths, PIDs, process-group IDs, job handles, usernames, and temporary-directory
names. Validation rejects unknown keys, missing required native nodes, skipped
required nodes, non-allowlisted status values, unbounded strings, and host-local or
absolute path content. Repository-relative pytest node IDs are permitted only when
they exactly equal the three compile-time allowlisted strings above; arbitrary
relative paths are not. The run URL is the other bounded exception and must exactly
match
`https://github.com/rmusser01/tldw_chatbook/actions/runs/<numeric-run-id>`.
The normalizer constructs it from the allowlisted repository identity and the
numeric workflow run ID; it does not accept a caller-supplied URL.

JUnit parsing does not trust an arbitrary `file` value. The normalizer matches the
allowlisted testcase module/class name and test name, then emits the corresponding
compile-time node-ID constant with `/` separators. Duplicate matches, unexpected
parameters, or a required name under another module fail validation. Synthetic
JUnit fixtures cover POSIX and Windows separator spellings.

An initialized or setup-failure document may omit test outcomes, but it is never a
passing document. `--validate` first validates either the bounded passed schema or
the bounded failure schema, then exits nonzero unless status is `passed`, the
recorded pytest step outcome is `success`, no selected testcase failed, and every
required node passed. This preserves useful failure evidence without allowing a
structurally valid dependency failure—or a failure in a selected non-required
contract/normalizer test—to make the lane green.

The workflow uploads `task-601-platform-<platform>` for each lane. After all three
lanes pass one executable commit, their normalized results are aggregated into
`Docs/STT_Evaluation/task-601/platform-evidence.json` with a short README containing
the workflow URL, tested commit, matrix outcome, and scope statement.

The same normalizer owns an explicit `--aggregate` mode rather than relying on
manual JSON editing. Its inputs are exactly the three downloaded per-platform
documents. It validates each input first, then requires:

- schema version 1 and aggregate evidence label
  `task601_native_process_tree_matrix`;
- exactly the platform keys `linux-x86_64`, `windows-x86_64`, and
  `macos-x86_64`, with no extras;
- `status: passed` for every platform;
- the same `tested_commit` in all three inputs and the aggregate;
- one shared workflow run ID and run URL, plus a bounded attempt for each platform;
  and
- all three required node IDs present and passed on each platform.

It sorts keys and writes the aggregate atomically. A `--validate-aggregate` mode
must reject a changed commit, mismatched run, platform/host mismatch, missing/extra
platform, missing/skipped/failed required node, unknown key, non-allowlisted
path-like value, or malformed run identity. Focused unit tests cover both
aggregation and each rejection. The checked-in README summarizes values read from
the validated aggregate; it is not an independent source of truth.

## Failure Handling

- Dependency setup failure records `dependency_install` and fails validation.
- Missing or malformed JUnit records `test_execution` and fails validation.
- A failed, skipped, or absent required native node keeps the platform red.
- A job timeout remains a failed release gate even if the runner cannot upload its
  initialized artifact.
- A native platform defect is fixed with a focused RED/GREEN test on that platform;
  all three lanes then rerun on the repaired executable commit.
- Evidence JSON never converts infrastructure failure into a passing platform
  claim.

## Completion and Task Hygiene

TASK-601 acceptance criterion 6 may be checked only when Linux, Windows, and macOS
all pass the required native nodes for the same executable commit. A later commit
may add only evidence documentation and Backlog metadata. Any change to production,
tests, the workflow, or the evidence normalizer invalidates the matrix and requires
a rerun.

Once the matrix passes:

1. Check in the aggregate evidence and README.
2. Update TASK-601 through Backlog CLI with the workflow run, tested commit, exact
   outcomes, and platform scope.
3. Check acceptance criterion 6 and mark TASK-601 Done only after all other
   Definition-of-Done requirements remain satisfied.
4. Record no new lesson unless the native run exposes a reusable platform trap.

## Expected File Boundary

- Modify `Tests/STT/test_executor_process_tree.py` for portable native assertions.
- Modify `Tests/STT/executor_test_support.py` only for spawn-importable helpers.
- Modify `Tests/STT/test_local_stt_executor.py` only to make the existing
  force-stop/scratch-ordering assertion non-vacuous.
- Create `Tests/CI/test_task601_process_tree_evidence.py` for workflow/schema
  ratchets.
- Create `.github/scripts/task601_process_tree_evidence.py` for normalization.
- Create `.github/workflows/task-601-platform-evidence.yml` for the explicit matrix.
- After a green matrix, create `Docs/STT_Evaluation/task-601/README.md` and
  `Docs/STT_Evaluation/task-601/platform-evidence.json`.
- Update the TASK-601 Backlog file only through Backlog CLI.

No production file is in the planned boundary. If native evidence requires a
production fix, stop, document the defect, extend the plan deliberately, and prove
the fix with the failing platform case before changing production.
