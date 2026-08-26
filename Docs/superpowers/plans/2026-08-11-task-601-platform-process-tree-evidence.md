# TASK-601 Native Process-Tree Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce same-commit native evidence that the local STT worker and its preparation descendants terminate before scratch cleanup on Linux, Windows, and macOS, then close TASK-601 acceptance criterion 6.

**Architecture:** Keep `ExecutorProcessTree` as the only containment implementation. Generalize its existing real-descendant tests, make the controller cleanup-order test non-vacuous, and run those contracts in a label-gated three-OS GitHub Actions matrix. A small standard-library script converts JUnit into strict path-private JSON and aggregates only three passing results from one tested commit and workflow run.

**Tech Stack:** Python 3.12, `multiprocessing` spawn, POSIX process groups, Windows Job Objects through existing `ctypes`, pytest/JUnit XML, GitHub Actions, standard-library `argparse`/`json`/`platform`/`xml.etree.ElementTree`, Ruff, Backlog CLI.

---

## Preconditions and scope

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-601-local-stt-executor` on `codex/task-601-platform-evidence`.
- This approved plan and its TASK-601 Backlog plan link must be committed before Task 1 begins; execution starts from a clean worktree.
- Governing spec: `Docs/superpowers/specs/2026-08-11-task-601-platform-process-tree-evidence-design.md`.
- Existing design: `Docs/superpowers/specs/2026-08-02-task-601-local-stt-executor-design.md`.
- ADR required: no new ADR.
- ADR paths: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` and `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`.
- Reason: ADR-025 already fixes process-tree ownership and cleanup order; this plan supplies the deferred native evidence without changing that boundary.
- Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before each commit or passing claim.
- Use the repository evidence lesson: record genuine RED or mutation evidence before implementation, compare exact failure sets, and never treat a skip, timeout, setup failure, or valid failure document as a passing platform.
- Do not run the full repository suite or rely on general CI. Run only the explicit TASK-601 nodes and changed-file static checks.
- Do not install an STT model/runtime extra, download a model, run inference, or add an FFmpeg correctness probe. A real descendant created after worker admission proves the OS ownership boundary for any ordinary preparation child.
- Do not modify production preemptively. If a native lane exposes a production defect, stop after preserving the RED, update this plan and TASK-601 notes, then make the smallest production fix under TDD and rerun all three lanes on the repaired commit.
- Never use `importlib.reload()` on the IPC/containment module; TASK-601 already proved that this splits exact dataclass identity across spawned processes.
- Before the evidence run, rebase onto current `origin/dev`, complete all executable changes, and freeze production/tests/workflow/normalizer. After evidence, only evidence documentation and Backlog metadata may change without rerunning the matrix.

## File map

- Modify `Tests/STT/test_executor_process_tree.py` — portable non-destructive liveness/finalization plus two real native descendant contracts.
- Modify `Tests/STT/executor_test_support.py` — only the spawn-importable descendant helpers required by those contracts.
- Modify `Tests/STT/test_local_stt_executor.py` — strengthen the existing force-stop scratch-ordering test with a blocked real termination call.
- Create `Tests/CI/test_task601_process_tree_evidence.py` — strict unit ratchets for result schemas, JUnit normalization, aggregation, and workflow shape.
- Create `.github/scripts/task601_process_tree_evidence.py` — one standard-library per-platform normalizer/validator/aggregator; no containment implementation.
- Create `.github/workflows/task-601-platform-evidence.yml` — explicit label/manual three-OS matrix.
- Create after a green matrix: `Docs/STT_Evaluation/task-601/README.md` and `Docs/STT_Evaluation/task-601/platform-evidence.json`.
- Modify through Backlog CLI only: `backlog/tasks/task-601 - Add-generation-fenced-local-STT-executor.md`.

### Task 1: Make the native containment evidence portable and non-vacuous

**Files:**
- Modify: `Tests/STT/test_executor_process_tree.py`
- Modify: `Tests/STT/executor_test_support.py`
- Modify: `Tests/STT/test_local_stt_executor.py`
- Reference only: `tldw_chatbook/STT/executor_process_tree.py`
- Reference only: `tldw_chatbook/STT/executor.py:1039-1059`

- [ ] **Step 1: Record the current focused baseline**

Run:

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  -q
```

Expected on the current macOS host: the existing POSIX contracts and controller node pass. Record the exact count and skips; this is baseline only, not cross-platform evidence.

- [ ] **Step 2: Add a Windows-native non-destructive PID probe regression before changing the helper**

In `Tests/STT/test_executor_process_tree.py`, add a Windows-only real-child test:

```python
@pytest.mark.skipif(os.name != "nt", reason="Windows process-handle contract")
def test_windows_pid_probe_does_not_terminate_a_live_process() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _pid_has_exited(child.pid) is False
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(10.0)
```

Do not fake this test. On Windows, the old `os.kill(pid, 0)` implementation would kill the child and the second assertion discriminates that behavior. On POSIX it is explicitly skipped and makes no Windows claim.

- [ ] **Step 3: Record the local platform limitation honestly**

Locally run the node to verify it collects and skips cleanly on macOS:

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_executor_process_tree.py::test_windows_pid_probe_does_not_terminate_a_live_process \
  -q
```

Expected locally: one skip. Record this as collection/skip evidence only, not a RED or Windows result. The first real execution of this behavior is the explicit Windows matrix lane in Task 4; do not simulate a Windows PASS or RED on POSIX.

- [ ] **Step 4: Replace the test helper with the existing repository Win32 wait idiom**

Replace `_pid_exists()` with `_pid_has_exited()` and a small Windows branch equivalent to the existing helper in `Tests/Notes/test_git_process_containment.py`:

```python
def _pid_has_exited(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    error_invalid_parameter = 87
    wait_object_0 = 0
    wait_timeout = 258
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Declare OpenProcess, WaitForSingleObject, and CloseHandle arg/restypes
    # exactly as Tests/Notes/test_git_process_containment.py does.
    ctypes.set_last_error(0)
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == error_invalid_parameter:
            return True
        raise OSError(error, "OpenProcess could not prove PID exit")
    try:
        result = kernel32.WaitForSingleObject(handle, 0)
        if result == wait_object_0:
            return True
        if result == wait_timeout:
            return False
        raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
    finally:
        kernel32.CloseHandle(handle)
```

Keep finalizers exact: retry the owned `ExecutorProcessTree` first; if an individually captured Windows fixture PID remains, open only that PID with `PROCESS_TERMINATE | SYNCHRONIZE`, call `TerminateProcess`, require `WaitForSingleObject` to report the handle signaled within the bounded timeout, then close it. On POSIX, signal only the captured process group. Never use a broad process-name search or `taskkill` without an exact PID.

- [ ] **Step 5: Generalize the two real POSIX descendant contracts**

Rename and remove the POSIX skip from these nodes:

```text
Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup
Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup
```

Retain the behavior in this order:

1. Spawn with `multiprocessing.get_context("spawn")`.
2. Receive the exact `WorkerContainmentIdentity`.
3. Construct the real `ExecutorProcessTree` and call `admit()`.
4. Receive the real descendant PID and prove it is still alive.
5. Call the real `terminate_tree()` and require `True`.
6. Require worker exit and `_pid_has_exited(descendant_pid) is True`.
7. Only then remove scratch and assert it is absent.

Keep `containment_descendant` and `containment_crashed_leader_with_term_ignoring_descendant` as normal importable module-level spawn targets in `Tests/STT/executor_test_support.py`. Add only the minimal OS conditional needed for a child that ignores cooperative POSIX TERM; Windows `TerminateJobObject` is unhandleable and needs no emulated signal path.

- [ ] **Step 6: Strengthen the controller cleanup-order test before touching production**

Change the existing controller test signature to accept `monkeypatch`. Gate its real tree method:

```python
tree = executor._tree
assert tree is not None
original_terminate = tree.terminate_tree
termination_entered = threading.Event()
allow_termination = threading.Event()

def gated_terminate(**kwargs: float) -> bool:
    termination_entered.set()
    assert allow_termination.wait(10.0)
    return original_terminate(**kwargs)

monkeypatch.setattr(tree, "terminate_tree", gated_terminate)
assert executor.force_stop("held") is True
assert termination_entered.wait(10.0)
try:
    assert scratch.exists() is True
finally:
    allow_termination.set()
assert executor.wait_for_retirement(10.0) is True
assert scratch.exists() is False
```

The `finally` is mandatory so a failed assertion cannot leave the retirement thread or worker alive.

- [ ] **Step 7: Prove the new ordering assertion is discriminating**

Temporarily mutate `_terminate_detached()` in `tldw_chatbook/STT/executor.py` so it removes `detached.scratch_path` immediately before calling `terminate_tree()`. Run only:

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  -q
```

Expected: FAIL at `scratch.exists() is True`. Restore the production file exactly, rerun, and require PASS. Do not commit the mutation.

- [ ] **Step 8: Run the Task 1 focused gate and static checks**

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  -q
../../.venv/bin/python -m ruff check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py
../../.venv/bin/python -m ruff format --check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py
git diff --check
```

Expected locally: zero failures; the Windows-only liveness node remains an honest skip.

- [ ] **Step 9: Commit the portable evidence tests**

```bash
git add \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py
git commit -m "test(stt): make process-tree evidence native"
```

### Task 2: Build the bounded path-private evidence normalizer

**Files:**
- Create: `Tests/CI/test_task601_process_tree_evidence.py`
- Create: `.github/scripts/task601_process_tree_evidence.py`
- Reference only: `.github/scripts/task598_external_parakeet_evidence.py`

- [ ] **Step 1: Write normalizer tests before the script exists**

Create `Tests/CI/test_task601_process_tree_evidence.py` with `importlib.util.spec_from_file_location()` loading the planned script and constants for the three exact required node IDs.

Add focused tests for:

- missing script is a collection/assertion RED;
- checked-out `git rev-parse HEAD` is the authoritative 40-character lowercase SHA, not `GITHUB_SHA`;
- each exact evidence name accepts only its matching derived system/architecture;
- `AMD64` normalizes to `x86_64` only for Windows;
- a synthetic passing JUnit document plus pytest outcome `success` produces a passing result;
- step outcome `failure`, any selected `<failure>`/`<error>`, a missing/skipped/duplicated/parameterized required node, or a required test name under the wrong module produces `test_execution` failure;
- JUnit `file` fields using `/` or `\` are ignored as authority and never copied to JSON;
- initialized/dependency failure documents are structurally bounded but CLI validation exits nonzero;
- absolute POSIX paths, Windows drive/UNC paths, PIDs, handles, usernames, commands, and an off-repository run URL are rejected;
- exactly three matching passing platform documents aggregate successfully;
- changed commit, changed workflow run, host mismatch, missing/extra platform, failed/skipped node, unknown key, or path-like content fails aggregation.

Use small synthetic XML strings and parameterization. Do not invoke pytest recursively from these unit tests.

- [ ] **Step 2: Run the script-focused tests to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/CI/test_task601_process_tree_evidence.py \
  -k "not workflow" \
  -q
```

Expected: FAIL because `.github/scripts/task601_process_tree_evidence.py` is missing.

- [ ] **Step 3: Implement one standard-library script and no framework**

Create `.github/scripts/task601_process_tree_evidence.py` with these fixed constants:

```python
SCHEMA_VERSION = 1
EVIDENCE_LABEL = "task601_native_process_tree"
AGGREGATE_LABEL = "task601_native_process_tree_matrix"
EXPECTED_REPOSITORY = "rmusser01/tldw_chatbook"
REQUIRED_NODES = (
    "Tests/STT/test_executor_process_tree.py::test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup",
    "Tests/STT/test_executor_process_tree.py::test_native_crashed_leader_reaps_descendant_before_scratch_cleanup",
    "Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch",
)
EXPECTED_PLATFORMS = {
    "linux-x86_64": ("Linux", "x86_64"),
    "windows-x86_64": ("Windows", "x86_64"),
    "macos-x86_64": ("Darwin", "x86_64"),
}
```

Keep the result shapes exact:

```json
{
  "schema_version": 1,
  "evidence_label": "task601_native_process_tree",
  "evidence_name": "linux-x86_64",
  "status": "passed",
  "failure_code": null,
  "failure_stage": null,
  "run": {
    "tested_commit": "<40 lowercase hex>",
    "workflow_run_id": "<digits>",
    "workflow_run_attempt": "<digits>",
    "workflow_run_url": "https://github.com/rmusser01/tldw_chatbook/actions/runs/<digits>"
  },
  "host": {"system": "Linux", "architecture": "x86_64", "python": "3.12.x"},
  "pytest": {
    "outcome": "success",
    "duration_seconds": 1.0,
    "required_nodes": {"<exact node id>": "passed"}
  }
}
```

Implement only these responsibilities:

- `current_run_identity()` calls `git rev-parse HEAD`, validates the SHA, validates numeric run ID/attempt, and constructs the fixed-repository run URL itself.
- `_host_result(evidence_name)` normalizes `AMD64` to `x86_64`, rejects every other unexpected name/system/architecture pair, and records only Python version.
- `failure_result(...)` creates a bounded failed document without exception text or paths.
- `result_from_junit(...)` uses `xml.etree.ElementTree`, matches testcase `classname` plus exact `name`, emits only compile-time node IDs, records no raw JUnit text, and passes only when the GitHub step outcome is `success`, no selected testcase failed/errored, and all required nodes passed exactly once.
- `validate_result(..., require_pass=True)` rejects unknown keys/values and returns failure for any non-passing release result.
- `_write_result()` writes sorted JSON to a sibling temporary file and publishes with `os.replace()`.
- `aggregate_results()` accepts exactly three validated passing inputs, requires one tested commit/run ID/run URL, permits bounded per-platform attempts, and writes the exact platform map.
- `validate_aggregate()` applies the same strict/path-private validation.
- Recursive string validation permits only the three exact node IDs and the one canonical run URL as slash-bearing values; all other absolute/UNC/drive paths and unbounded strings fail.

The CLI surface is only:

```text
--initialize --evidence-name NAME --output PATH
--record-failure CODE --failure-stage STAGE --evidence-name NAME --output PATH
--from-junit PATH --pytest-outcome OUTCOME --evidence-name NAME --output PATH
--validate PATH
--aggregate INPUT INPUT INPUT --output PATH
--validate-aggregate PATH
```

Do not add classes, dependencies, a generic evidence package, subprocess supervision, or containment code.

- [ ] **Step 4: Run the normalizer tests to verify GREEN**

```bash
../../.venv/bin/python -m pytest \
  Tests/CI/test_task601_process_tree_evidence.py \
  -k "not workflow" \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Mutation-check the three decisive guards**

One at a time, temporarily mutate and restore:

1. Ignore `pytest_outcome != "success"` — the step-outcome failure test must fail.
2. Trust JUnit `file` or test `name` without the exact module — the wrong-module/path test must fail.
3. Allow aggregate commit/run mismatch — the aggregation mismatch tests must fail.

After each mutation, run only its exact node, record the RED, restore, and rerun GREEN. Do not commit mutations.

- [ ] **Step 6: Run static checks and commit the normalizer**

```bash
../../.venv/bin/python -m ruff check \
  .github/scripts/task601_process_tree_evidence.py \
  Tests/CI/test_task601_process_tree_evidence.py
../../.venv/bin/python -m ruff format --check \
  .github/scripts/task601_process_tree_evidence.py \
  Tests/CI/test_task601_process_tree_evidence.py
../../.venv/bin/python -m py_compile \
  .github/scripts/task601_process_tree_evidence.py
git diff --check
git add \
  .github/scripts/task601_process_tree_evidence.py \
  Tests/CI/test_task601_process_tree_evidence.py
git commit -m "test(stt): normalize task 601 platform evidence"
```

### Task 3: Add the explicit three-OS evidence workflow

**Files:**
- Modify: `Tests/CI/test_task601_process_tree_evidence.py`
- Create: `.github/workflows/task-601-platform-evidence.yml`
- Reference only: `.github/workflows/task-598-platform-evidence.yml`

- [ ] **Step 1: Add workflow ratchets before creating the workflow**

Require the workflow text to prove:

- only `pull_request: types: [labeled]` and `workflow_dispatch`; no push/schedule;
- job condition is manual dispatch or label `task-601-platform-evidence`;
- `permissions: contents: read`;
- exact checkout ref `${{ github.event.pull_request.head.sha || github.sha }}`;
- exact runners `ubuntu-24.04`, `windows-2022`, `macos-15-intel` and evidence names matching the script constants;
- Python 3.12, `fail-fast: false`, and a bounded job timeout;
- `defaults.run.shell: bash` so the same quoted multiline commands execute under Git for Windows Bash rather than PowerShell;
- installation of `.[dev]` only, with no transcription extra or runtime package;
- initialization occurs before dependency installation;
- dependency and pytest steps use `continue-on-error: true`;
- the exact pytest command includes the process-tree file, controller node, and normalizer test file plus JUnit output;
- normalization consumes `${{ steps.platform_tests.outcome }}`;
- validation and artifact upload both use `if: always()` and upload one JSON named `task-601-platform-${{ matrix.evidence_name }}`.

- [ ] **Step 2: Run the workflow tests to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/CI/test_task601_process_tree_evidence.py \
  -k workflow \
  -q
```

Expected: FAIL because `.github/workflows/task-601-platform-evidence.yml` is missing.

- [ ] **Step 3: Create the minimal workflow**

Create `.github/workflows/task-601-platform-evidence.yml` with this step order:

Set `defaults.run.shell: bash` once for the job. GitHub-hosted Windows includes Git Bash, and the explicit default keeps line continuations, quoting, and environment expansion identical across all three lanes.

1. Checkout exact PR head/manual ref.
2. Set up Python 3.12 with pip cache.
3. Initialize `$RUNNER_TEMP/task-601-platform-evidence.json`.
4. Install `pip install -e ".[dev]"` in a `continue-on-error` step.
5. Record bounded `dependency_install` failure if install failed.
6. Run this one `continue-on-error` pytest step only if installation passed:

```bash
python -m pytest \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  Tests/CI/test_task601_process_tree_evidence.py \
  --timeout=60 \
  --junitxml="$RUNNER_TEMP/task-601-junit.xml" \
  -q
```

7. Under `if: always() && steps.dependencies.outcome == 'success'`, normalize JUnit with `--pytest-outcome "${{ steps.platform_tests.outcome }}"`.
8. Under `if: always()`, run `--validate`; its nonzero failure result makes the lane red.
9. Under `if: always()`, upload only the JSON with `if-no-files-found: error`.

Set `timeout-minutes: 20`. Do not add matrix architectures, model caching, concurrency infrastructure, secrets, write permissions, or general-CI triggers.

- [ ] **Step 4: Run workflow/normalizer tests and mutation-check outcome plumbing**

```bash
../../.venv/bin/python -m pytest \
  Tests/CI/test_task601_process_tree_evidence.py \
  -q
```

Expected: all tests pass.

Temporarily remove `--pytest-outcome "${{ steps.platform_tests.outcome }}"` or replace it with a literal `success`. Run the exact workflow ratchet; expected FAIL. Restore and rerun GREEN.

- [ ] **Step 5: Run static/diff checks and commit the workflow**

```bash
../../.venv/bin/python -m ruff check \
  .github/scripts/task601_process_tree_evidence.py \
  Tests/CI/test_task601_process_tree_evidence.py
../../.venv/bin/python -m ruff format --check \
  .github/scripts/task601_process_tree_evidence.py \
  Tests/CI/test_task601_process_tree_evidence.py
git diff --check
git add \
  .github/workflows/task-601-platform-evidence.yml \
  Tests/CI/test_task601_process_tree_evidence.py
git commit -m "ci(stt): add task 601 platform evidence"
```

### Task 4: Freeze the executable commit and collect the native matrix

**Files:**
- Potentially modify only after a genuine native RED: the smallest affected production/test file, after updating this plan.
- Create after GREEN: `Docs/STT_Evaluation/task-601/README.md`
- Create after GREEN: `Docs/STT_Evaluation/task-601/platform-evidence.json`

- [ ] **Step 1: Rebase before evidence and freeze executable files**

Fetch and rebase onto current `origin/dev` before the evidence run:

```bash
git fetch origin dev
git rebase origin/dev
```

Resolve only owned conflicts. Then run the complete focused local gate:

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  Tests/CI/test_task601_process_tree_evidence.py \
  -q
../../.venv/bin/python -m ruff check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py \
  Tests/CI/test_task601_process_tree_evidence.py \
  .github/scripts/task601_process_tree_evidence.py
../../.venv/bin/python -m ruff format --check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py \
  Tests/CI/test_task601_process_tree_evidence.py \
  .github/scripts/task601_process_tree_evidence.py
../../.venv/bin/python -m py_compile \
  .github/scripts/task601_process_tree_evidence.py
git diff --check origin/dev...HEAD
```

Record the executable HEAD SHA for later comparisons:

```bash
task601_tested_commit="$(git rev-parse HEAD)"
```

From this point until evidence is aggregated, do not change production, tests, the workflow, or the normalizer.

- [ ] **Step 2: Push a PR and trigger only the explicit evidence workflow**

Push `codex/task-601-platform-evidence`, open a PR against `dev`, and ensure the repository label `task-601-platform-evidence` exists with a narrow description. Add the label to trigger the workflow. Do not use broad CI as TASK-601 evidence.

Use GitHub API/run inspection to select the workflow named `TASK-601 native process-tree evidence` whose tested PR head equals the frozen executable SHA. Record its numeric run ID as `task601_run_id`. Do not infer success from overall PR check status.

- [ ] **Step 3: Treat any red lane as a real open gate**

Wait for Linux, Windows, and macOS artifacts. If any lane is red, skipped, timed out, or missing:

1. Download its JSON and inspect only the exact workflow logs needed to classify setup versus native test failure.
2. Preserve the exact failing node and host result.
3. Do not mark AC6 complete or aggregate evidence.
4. If the failure is a product defect, update this plan and TASK-601 notes first, write/verify the focused failing test on that platform, apply the smallest production fix, rebase if required, rerun the local gate, and rerun all three lanes on the new executable commit.
5. Do not waive an outer Job Object/nested-job failure as “CI-only”; Windows hosted execution is the required native environment.

- [ ] **Step 4: Aggregate only one green workflow run**

After all three lanes pass, download exactly these artifacts from the same workflow run into a new `mktemp -d` directory:

```text
task-601-platform-linux-x86_64
task-601-platform-windows-x86_64
task-601-platform-macos-x86_64
```

Create the directory and download each artifact to an explicit child directory:

```bash
task601_evidence_dir="$(mktemp -d)"
gh run download "$task601_run_id" \
  --name task-601-platform-linux-x86_64 \
  --dir "$task601_evidence_dir/linux-x86_64"
gh run download "$task601_run_id" \
  --name task-601-platform-windows-x86_64 \
  --dir "$task601_evidence_dir/windows-x86_64"
gh run download "$task601_run_id" \
  --name task-601-platform-macos-x86_64 \
  --dir "$task601_evidence_dir/macos-x86_64"
mkdir -p Docs/STT_Evaluation/task-601
```

Generate and validate the checked-in aggregate:

```bash
../../.venv/bin/python .github/scripts/task601_process_tree_evidence.py \
  --aggregate \
  "$task601_evidence_dir/linux-x86_64/task-601-platform-evidence.json" \
  "$task601_evidence_dir/windows-x86_64/task-601-platform-evidence.json" \
  "$task601_evidence_dir/macos-x86_64/task-601-platform-evidence.json" \
  --output Docs/STT_Evaluation/task-601/platform-evidence.json
../../.venv/bin/python .github/scripts/task601_process_tree_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-601/platform-evidence.json
```

Expected: both commands exit zero; aggregate status is passed, contains exactly three platforms, one tested commit/run, and no local path/PID/handle/exception text.

- [ ] **Step 5: Add the concise evidence README**

Create `Docs/STT_Evaluation/task-601/README.md` from the validated aggregate. Record:

- purpose and exact scope (process ownership/cleanup, not model or FFmpeg correctness);
- workflow URL and executable tested commit;
- exact three runner labels and host architectures;
- exact three required node IDs and passing status;
- explicit statement that no model/runtime extra, network/model download, or general CI result was used;
- any native defect/fix/rerun history; and
- the aggregate validation command.

Do not copy local artifact paths, PIDs, usernames, or raw failure text.

- [ ] **Step 6: Validate evidence before changing Backlog state**

```bash
../../.venv/bin/python .github/scripts/task601_process_tree_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-601/platform-evidence.json
rg -n '/Users/|/private/|/tmp/|[A-Za-z]:\\|workflow_dispatch.*token|TODO|TBD|PLACEHOLDER' \
  Docs/STT_Evaluation/task-601 \
  || true
git diff --check
```

Expected: aggregate validation passes; the scan produces no local path, secret-like, or placeholder hit requiring correction.

- [ ] **Step 7: Close TASK-601 through Backlog CLI only**

Use `backlog task edit 601` to:

- add this implementation plan and the evidence README/JSON to documentation while preserving existing docs;
- append implementation notes with workflow run URL, exact tested commit, three platform outcomes, focused local test/static results, and any native remediation;
- check acceptance criterion 6; and
- set status Done only after every Definition-of-Done item remains satisfied.

Do not hand-edit the task Markdown.

- [ ] **Step 8: Commit only evidence documentation and task metadata**

Before committing, prove the post-evidence diff contains no executable file:

```bash
git diff --name-only "$task601_tested_commit"
```

Expected at this phase: only `Docs/STT_Evaluation/task-601/*` and the TASK-601 Backlog file differ from the tested commit, including uncommitted evidence changes. After staging, prove the same boundary against the index:

```bash
git add \
  Docs/STT_Evaluation/task-601/README.md \
  Docs/STT_Evaluation/task-601/platform-evidence.json \
  'backlog/tasks/task-601 - Add-generation-fenced-local-STT-executor.md'
git diff --cached --name-only "$task601_tested_commit"
git diff --cached --check
git commit -m "docs(stt): record task 601 platform evidence"
```

Expected staged names: exactly the two evidence files and the TASK-601 Backlog file.

### Task 5: Final review and integration handoff

**Files:**
- Review the full `origin/dev...HEAD` range.
- Modify only files required by an accepted review finding.

- [ ] **Step 1: Run final focused verification from the committed tree**

```bash
../../.venv/bin/python -m pytest \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/test_local_stt_executor.py::test_force_stop_detaches_before_kill_and_cleans_generation_scratch \
  Tests/CI/test_task601_process_tree_evidence.py \
  -q
../../.venv/bin/python .github/scripts/task601_process_tree_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-601/platform-evidence.json
../../.venv/bin/python -m ruff check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py \
  Tests/CI/test_task601_process_tree_evidence.py \
  .github/scripts/task601_process_tree_evidence.py
../../.venv/bin/python -m ruff format --check \
  Tests/STT/test_executor_process_tree.py \
  Tests/STT/executor_test_support.py \
  Tests/STT/test_local_stt_executor.py \
  Tests/CI/test_task601_process_tree_evidence.py \
  .github/scripts/task601_process_tree_evidence.py
../../.venv/bin/python -m py_compile \
  .github/scripts/task601_process_tree_evidence.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: focused tests/static/evidence validation pass and the worktree is clean.

- [ ] **Step 2: Request correctness and Ponytail review**

Ask reviewers to inspect only this branch range against:

- TASK-601 AC6;
- the approved platform-evidence spec;
- native Windows/POSIX proof versus mocked contracts;
- non-destructive liveness/finalization;
- scratch cleanup ordering and quarantine behavior;
- pytest-outcome/JUnit non-vacuity;
- path privacy, commit/run/platform binding, and aggregate consistency; and
- unnecessary evidence abstractions or duplicated containment logic.

- [ ] **Step 3: Apply findings without invalidating evidence silently**

For each accepted finding:

- docs/Backlog-only correction: fix, validate, and commit;
- production/test/workflow/normalizer correction: the existing matrix is invalid. Make the change under TDD, rebase if required, rerun the full local gate and all three native lanes, regenerate aggregate/README/task notes, and commit new evidence.

Never edit the aggregate by hand or preserve an old tested commit after executable changes.

- [ ] **Step 4: Preserve the tested commit through merge**

Do not rebase or rewrite executable commits after final evidence. If `dev` advances and a rebase is required, rerun the matrix and regenerate evidence. Use an integration method that preserves the tested executable commit as an ancestor; do not squash away the only SHA named by the evidence without rerunning against the replacement commit.

- [ ] **Step 5: Hand off the ready PR**

Report:

- PR URL and final branch head;
- tested executable commit and workflow run URL;
- Linux/Windows/macOS native outcomes;
- exact focused test/static results;
- aggregate/task status;
- any rerun/remediation history; and
- confirmation that general CI, model inference, downloads, and unrelated cleanup were not used as TASK-601 evidence.
