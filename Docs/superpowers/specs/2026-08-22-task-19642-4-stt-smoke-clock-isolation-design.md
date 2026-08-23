# TASK-19642.4 STT Smoke Clock Isolation Design

**Status:** approved by the user on 2026-08-22

**Related task:** TASK-19642.4

## Purpose

Repair the order-sensitive task-602 smoke unit test without changing the native
STT runtime, evidence runner contract, or five-platform release evidence. The
test must keep its exact bounded and allowlisted result assertions while its
fake clock remains private to the dynamically loaded smoke module.

## Root cause

`test_run_smoke_returns_only_bounded_allowlisted_observations` currently patches
`smoke.time.monotonic`. The smoke module's `time` name refers to Python's shared
standard-library `time` module, so this patch replaces `time.monotonic`
process-wide with a three-value iterator.

The node passes alone, and the complete task-602 smoke test file passes alone.
A focused diagnostic uses a `pytest.MonkeyPatch` subclass to make one unrelated
process-clock call immediately after the existing patch. That call
deterministically exhausts the iterator and raises `StopIteration` from
`run_smoke()` at `.github/scripts/task602_platform_smoke.py:714`. This explains
why the node can error only during a broad, loaded run: any concurrent process
clock consumer can steal one of the three values.

## Selected approach

Rebind the loaded smoke module's `time` name to a private fake clock object
instead of mutating an attribute on the shared module. Add a direct regression
assertion that the process-wide `time.monotonic` identity remains unchanged.
Keep the existing exact result, duration, setup-order, offline-environment, and
path-privacy assertions.

This follows existing repository tests that replace a module's `time` binding
with a small clock object. It requires no production seam, new dependency,
serialization marker, or broad-suite exception.

## Alternatives rejected

- Add a clock parameter to `.github/scripts/task602_platform_smoke.py`: this
  expands an executable evidence script API only to serve one unit test.
- Serialize or isolate the test: this hides the global mutation rather than
  removing it.
- Make the iterator unbounded: this reduces exhaustion risk but still replaces
  the process clock and allows unrelated consumers to corrupt asserted timing.

## Verification

1. Preserve the focused diagnostic command that consumes one process-clock
   value and records `StopIteration` at the third `run_smoke()` clock read.
2. Add a process-clock identity assertion while retaining the current patch,
   and run the exact node to prove the permanent regression check RED with an
   `AssertionError` distinct from the diagnostic traceback.
3. Rebind only `smoke.time` to the private clock and rerun the exact node for
   GREEN.
4. Run `Tests/STT/test_task602_platform_smoke.py` and the task-owned platform
   evidence validator tests; do not run unrelated local suites.
5. Provision or locate Python 3.11, 3.12, 3.13, and 3.14 and run the exact node
   under every interpreter. Equivalent exact-node CI evidence may substitute
   for a local interpreter, but absent coverage blocks completion. Require the
   pull request's standard Python 3.12 Ubuntu and macOS core-test lanes to
   remain green.
6. Because the approved TASK-602 evidence design invalidates native evidence
   after any task-owned test change, trigger the label-gated five-lane native
   Python 3.12 workflow on the reviewed commit and refresh its checked-in
   aggregate and README from that single green run.
7. Run Ruff check/format and `git diff --check` for the modified files.
8. Record both the focused traceback and the supported-platform results in the
   task and TASK-19520 failure inventory.

## Platform scope

The repair is interpreter-level test isolation and does not depend on an STT
native wheel. The exact node's supported test matrix is Python 3.11-3.14 plus
the repository's standard Python 3.12 Ubuntu and macOS core-test lanes. It does
not claim direct Windows execution of pytest.

TASK-602's separate native runtime matrix remains Linux x86_64/aarch64, Windows
x86_64, and macOS arm64/x86_64 on Python 3.12. Although the executable smoke
script does not change, the approved TASK-602 evidence design says any
task-owned test change invalidates that evidence. The five native lanes must
therefore be rerun on the reviewed commit before closeout; prior evidence is a
control, not completion evidence for this repair.

## ADR decision

**ADR required:** no

**ADR path:** N/A

**Reason:** this is a test-harness isolation repair that preserves existing
runtime, security, platform, evidence, and module boundaries. ADR-025 continues
to govern the STT artifact and runtime contract.
