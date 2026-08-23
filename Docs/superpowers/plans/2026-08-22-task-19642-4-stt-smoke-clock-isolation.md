# TASK-19642.4 STT Smoke Clock Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the task-602 bounded smoke unit test deterministic without
mutating the process-wide monotonic clock, then refresh its governed platform
evidence.

**Architecture:** Keep the executable smoke runner unchanged. Replace only the
dynamically loaded test module's `time` binding with a private fake clock, pin
that the standard-library clock identity is unchanged, and retain every exact
bounded-output assertion. Verify the pure harness across Python 3.11-3.14 and
standard CI, then rerun TASK-602's governed five-lane native evidence because
its approved design invalidates evidence after a task-owned test change.

**Tech Stack:** Python 3.11-3.14, pytest, Ruff, GitHub Actions, existing
TASK-602 evidence normalizer and native workflow.

**Approved design:**
`Docs/superpowers/specs/2026-08-22-task-19642-4-stt-smoke-clock-isolation-design.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a test-harness isolation repair that preserves existing
runtime, security, platform, evidence, and module boundaries. ADR-025 continues
to govern STT artifact and runtime behavior.

---

## File map

- Modify: `Tests/STT/test_task602_platform_smoke.py` — keep the fake clock
  private and assert the process clock is untouched.
- Modify: `backlog/docs/task-19520-verification-failure-inventory.md` — replace
  the low-confidence entry with the focused traceback and resolution evidence.
- Modify: `backlog/docs/lessons-testing-evidence.md` — record the shared-module
  monkeypatch incident and its deterministic proof.
- Modify: `Docs/STT_Evaluation/task-602/platform-evidence.json` — replace the
  invalidated aggregate with the new same-run five-platform result.
- Modify: `Docs/STT_Evaluation/task-602/README.md` — record the refreshed run,
  commit, and scope.
- Modify:
  `backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md`
  — track the plan, verification, acceptance criteria, and closeout.
- No production, smoke-runner, normalizer, dependency, or workflow file changes.

## Global constraints

- Run commands from the repository root with the project development
  environment activated, so `python` resolves an interpreter with pytest and
  Ruff installed.
- Run no repository-wide local suite and no unrelated test directory.
- Do not change `.github/scripts/task602_platform_smoke.py`; the bug is the
  test's process-global patch.
- Preserve all exact result keys, durations, setup order, offline restoration,
  and path-privacy assertions in the repaired node.
- Never combine lanes from retries or attempts. A repository-caused failure
  requires a test-driven repair and a new reviewed commit. A proven GitHub
  infrastructure failure may use a wholly new run/attempt on the unchanged
  frozen commit.
- All five native results must belong to one commit, workflow run, and attempt.
- After the reviewed executable commit is frozen, later commits may contain
  only aggregate evidence, documentation, and Backlog metadata.

### Task 0: Commit the approved planning baseline

**Files:**

- Add: `Docs/superpowers/plans/2026-08-22-task-19642-4-stt-smoke-clock-isolation.md`
- Modify:
  `backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md`

- [ ] **Step 1: Verify and commit the plan before implementation**

```bash
git diff --check
git add \
  Docs/superpowers/plans/2026-08-22-task-19642-4-stt-smoke-clock-isolation.md \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md'
git commit -m "docs(stt): plan task 602 smoke clock isolation"
git status --short
```

Expected: the commit succeeds and status is empty before Task 1 begins.

### Task 1: Preserve the root-cause evidence

**Files:**

- Read: `Tests/STT/test_task602_platform_smoke.py:198`
- Read: `.github/scripts/task602_platform_smoke.py:666`
- Later modify: `backlog/docs/task-19520-verification-failure-inventory.md`

- [ ] **Step 1: Run the isolated control**

Run:

```bash
python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --tb=long
```

Expected: `1 passed`; this proves the defect is not an isolated deterministic
product failure.

- [ ] **Step 2: Run the complete task-owned test file control**

Run:

```bash
python -m pytest -q Tests/STT/test_task602_platform_smoke.py --tb=long
```

Expected: `20 passed` before the repair.

- [ ] **Step 3: Reproduce the broad-run failure shape deterministically**

Run this bounded one-off probe; it calls `time.monotonic()` once immediately
after the test installs its current `smoke.time.monotonic` patch and leaves no
file behind:

```bash
python -c 'exec("""import importlib.util
import tempfile
import time
import traceback
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("task602_test_probe", Path("Tests/STT/test_task602_platform_smoke.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
shared_patch_seen = [False]

class ProbeMonkeyPatch(pytest.MonkeyPatch):
    def setattr(self, target, name, value, raising=True):
        super().setattr(target, name, value, raising=raising)
        if name == "monotonic":
            assert target is time
            shared_patch_seen[0] = True
            time.monotonic()

patch = ProbeMonkeyPatch()
try:
    with tempfile.TemporaryDirectory() as directory:
        module.test_run_smoke_returns_only_bounded_allowlisted_observations(Path(directory), patch)
except StopIteration as error:
    frames = traceback.extract_tb(error.__traceback__)
    assert frames[-1].filename.endswith(".github/scripts/task602_platform_smoke.py")
    assert frames[-1].lineno == 714
    traceback.print_exception(error)
else:
    raise AssertionError("expected the extra process-clock consumer to exhaust the smoke clock")
finally:
    patch.undo()
assert shared_patch_seen == [True]
""")'
```

Expected traceback:

```text
Tests/STT/test_task602_platform_smoke.py:278
.github/scripts/task602_platform_smoke.py:714
StopIteration
```

Also record that direct inspection proves
`smoke.time is time` and the fourth process-clock call exhausts the three-value
iterator.

### Task 2: Test-drive the private clock binding

**Files:**

- Modify: `Tests/STT/test_task602_platform_smoke.py:1-10`
- Modify: `Tests/STT/test_task602_platform_smoke.py:198-305`

- [ ] **Step 1: Add the permanent failing assertion**

Add the standard-library import and capture the original process clock before
installing the fake:

```python
import time

# inside the test, before the fake clock is installed
process_monotonic = time.monotonic
```

Keep the current process-global patch temporarily, then add:

```python
assert time.monotonic is process_monotonic
```

immediately after it.

- [ ] **Step 2: Run the exact node to verify RED**

Run:

```bash
python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --tb=short
```

Expected: FAIL at the new identity assertion because the process clock was
replaced. This RED is deliberately distinct from Task 1's diagnostic
`StopIteration` traceback.

- [ ] **Step 3: Implement the minimal clock isolation**

Replace only the patch target:

```python
monotonic = iter((0.0, 3.0, 5.0)).__next__
monkeypatch.setattr(smoke, "time", SimpleNamespace(monotonic=monotonic))
assert time.monotonic is process_monotonic
```

Do not add an injectable production clock or change the smoke script.

- [ ] **Step 4: Run the exact node to verify GREEN**

Run the Step 2 command again.

Expected: `1 passed`; the existing exact result still reports acquisition
`3.0` and total `5.0`.

- [ ] **Step 5: Perform the inverse proof**

Temporarily restore the old `monkeypatch.setattr(smoke.time, "monotonic", ...)`
line while retaining the identity assertion. Run the exact node and require the
identity assertion to fail. Restore the private binding and rerun to `1 passed`.

- [ ] **Step 6: Commit the executable repair**

```bash
git add Tests/STT/test_task602_platform_smoke.py
git commit -m "test(stt): isolate task 602 smoke clock"
```

### Task 3: Run focused local and interpreter verification

**Files:**

- Verify: `Tests/STT/test_task602_platform_smoke.py`
- Verify: `Tests/CI/test_task602_platform_evidence.py`

- [ ] **Step 1: Run the task-owned test files**

```bash
python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py \
  Tests/CI/test_task602_platform_evidence.py \
  --tb=short
```

Expected: all nodes pass; no setup, call, teardown, or path-privacy error.

- [ ] **Step 2: Provision the supported interpreters**

```bash
uv python install 3.11 3.12 3.13 3.14
```

Absence of any interpreter blocks closeout unless equivalent exact-node CI
evidence is captured.

- [ ] **Step 3: Run the pure exact node on Python 3.11-3.14**

Run each command separately so one result cannot hide another:

```bash
uv run --python 3.11 --no-project --with pytest python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --confcutdir=Tests/STT --tb=short
uv run --python 3.12 --no-project --with pytest python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --confcutdir=Tests/STT --tb=short
uv run --python 3.13 --no-project --with pytest python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --confcutdir=Tests/STT --tb=short
uv run --python 3.14 --no-project --with pytest python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py::test_run_smoke_returns_only_bounded_allowlisted_observations \
  --confcutdir=Tests/STT --tb=short
```

Expected: `1 passed` four times.

- [ ] **Step 4: Run scoped static checks**

```bash
python -m ruff check Tests/STT/test_task602_platform_smoke.py
python -m ruff format --check Tests/STT/test_task602_platform_smoke.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: Ruff and whitespace checks pass; status contains no uncommitted
implementation files.

### Task 4: Review, rebase, and freeze the executable commit

**Files:**

- Review the complete `origin/dev...HEAD` diff.

- [ ] **Step 1: Run an independent correctness and minimality review**

Require the reviewer to verify the process clock remains untouched, the fake
clock still supplies exactly three smoke reads, every bounded/allowlisted
assertion remains, and no production or workflow change slipped in.

- [ ] **Step 2: Fetch and rebase onto the latest `origin/dev`**

```bash
git fetch origin dev
git rebase origin/dev
```

Rerun Task 3 after any conflict or relevant upstream change.

- [ ] **Step 3: Push and open the focused pull request**

Push the branch, open a ready PR, and wait for Qodo and other configured
reviewers. Address every technically valid comment through a focused RED/GREEN
cycle. Rebase again before freezing if `origin/dev` advances.

- [ ] **Step 4: Require the standard CI platform lanes**

The PR's Python 3.12 Ubuntu and macOS core-test lanes must pass. Inspect the
exact node if either lane fails; do not attribute unrelated failures to this
task without a focused traceback.

Query the repository's branch rules and PR checks. Require every configured
required check to pass. If no performance, security, or licence check applies
to this test-only/no-dependency diff, record each as scoped N/A in the task
notes; the bounded/path-private assertion gate remains the applicable security
check.

```bash
task19642_pr_number="$(gh pr view --json number --jq .number)"
task19642_required_rule_count="$(gh api repos/rmusser01/tldw_chatbook/rules/branches/dev \
  --jq '[.[] | select(.type == "required_status_checks")] | length')"
if test "${task19642_required_rule_count}" -gt 0; then
  gh pr checks "${task19642_pr_number}" --required --watch --interval 10
else
  echo 'No required status-check rule applies to dev.'
fi
test "$(gh pr checks "${task19642_pr_number}" --json name,bucket \
  --jq '[.[] | select(.name | startswith("Core Tests (all but UI) - Python 3.12")) | select(.bucket == "pass")] | length')" -eq 2
```

Expected: every configured required check passes, or the rules API proves none
exist; exactly two Python 3.12 core lanes (Ubuntu and macOS) are in the pass
bucket.

- [ ] **Step 5: Freeze the reviewed executable commit**

Record the PR head SHA after all executable review fixes and rebases. No Python,
test, smoke, normalizer, dependency, or workflow change may follow without
starting a brand-new native evidence run.

### Task 5: Refresh governed five-platform native evidence

**Files:**

- Modify: `Docs/STT_Evaluation/task-602/platform-evidence.json`
- Modify: `Docs/STT_Evaluation/task-602/README.md`

- [ ] **Step 1: Trigger the exact label-gated workflow**

Add the `task-602-platform-evidence` label to the PR. Confirm the selected run's
head SHA equals the frozen executable commit.

```bash
task19642_pr_number="$(gh pr view --json number --jq .number)"
task19642_frozen_sha="$(git rev-parse HEAD)"
gh pr edit "${task19642_pr_number}" --add-label task-602-platform-evidence
gh run list \
  --workflow task-602-platform-evidence.yml \
  --branch codex/task-19642-4-stt-smoke-clock \
  --limit 10 \
  --json databaseId,headSha,status,conclusion,event,url
```

- [ ] **Step 2: Monitor all five lanes**

Require Linux x86_64/aarch64, Windows x86_64, and macOS arm64/x86_64 to pass on
one workflow run and attempt.

Select the completed successful run matching the frozen SHA, then prove its
identity and job set:

```bash
task19642_expected_event=pull_request
task19642_frozen_sha="$(git rev-parse HEAD)"
task19642_run_id=''
for task19642_poll in {1..60}; do
  task19642_run_id="$(gh run list \
    --workflow task-602-platform-evidence.yml \
    --branch codex/task-19642-4-stt-smoke-clock \
    --limit 20 \
    --json databaseId,headSha,event \
    --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "'"${task19642_expected_event}"'") | .databaseId' \
    | head -n 1)"
  test -n "${task19642_run_id}" && break
  sleep 5
done
test -n "${task19642_run_id}"
gh run watch "${task19642_run_id}" --exit-status --interval 10
gh api "repos/rmusser01/tldw_chatbook/actions/runs/${task19642_run_id}" \
  --jq '{id,run_attempt,head_sha,status,conclusion,event,html_url}'
gh api "repos/rmusser01/tldw_chatbook/actions/runs/${task19642_run_id}/jobs?per_page=100" \
  --jq '[.jobs[] | {name,status,conclusion}]'
test "$(gh api "repos/rmusser01/tldw_chatbook/actions/runs/${task19642_run_id}/jobs?per_page=100" \
  --jq '[.jobs[] | select(.name | startswith("platform-evidence")) | {name,conclusion}] as $jobs | (($jobs | length) == 5 and (["linux-x86_64", "linux-aarch64", "windows-x86_64", "macos-arm64", "macos-x86_64"] | all(. as $lane | any($jobs[]; (.name | contains($lane)) and .conclusion == "success"))))')" = true
```

Expected: one successful run whose `head_sha` equals
`task19642_frozen_sha`, one attempt identity, and exactly five successful
`platform-evidence` jobs with the required native lane names.

If a lane fails because of repository behavior, do not aggregate or retry.
Return through Tasks 2, 3, and 4: add a failing test, implement the minimal
repair, rerun focused and Python-version gates, repeat independent review and
latest-dev rebase, require standard CI, and freeze a new SHA before triggering
a new five-lane run. If evidence proves a GitHub infrastructure failure with no
repository repair warranted, trigger one wholly new run/attempt on the same
frozen SHA; never mix attempts.

For that infrastructure-only branch, retrigger explicitly without relying on
an already-present label:

```bash
task19642_frozen_sha="$(git rev-parse HEAD)"
task19642_failed_run_id="$(gh run list \
  --workflow task-602-platform-evidence.yml \
  --branch codex/task-19642-4-stt-smoke-clock \
  --limit 20 \
  --json databaseId,headSha,status,conclusion,event \
  --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "pull_request" and .status == "completed" and .conclusion != "success") | .databaseId' \
  | head -n 1)"
test -n "${task19642_failed_run_id}"
gh workflow run task-602-platform-evidence.yml \
  --ref codex/task-19642-4-stt-smoke-clock
task19642_expected_event=workflow_dispatch
task19642_run_id=''
for task19642_poll in {1..60}; do
  task19642_run_id="$(gh run list \
    --workflow task-602-platform-evidence.yml \
    --branch codex/task-19642-4-stt-smoke-clock \
    --limit 20 \
    --json databaseId,headSha,event \
    --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "'"${task19642_expected_event}"'") | .databaseId' \
    | head -n 1)"
  test -n "${task19642_run_id}" && break
  sleep 5
done
test -n "${task19642_run_id}"
test "${task19642_run_id}" != "${task19642_failed_run_id}"
gh run watch "${task19642_run_id}" --exit-status --interval 10
```

Run the selection only after the new workflow completes, then repeat the
SHA/run/attempt/job proof above. In every remaining Task 5 command block, set
`task19642_expected_event=workflow_dispatch` instead of `pull_request`.

- [ ] **Step 3: Download the five named artifacts**

```bash
task19642_expected_event=pull_request
task19642_frozen_sha="$(git rev-parse HEAD)"
task19642_run_id="$(gh run list \
  --workflow task-602-platform-evidence.yml \
  --branch codex/task-19642-4-stt-smoke-clock \
  --limit 20 \
  --json databaseId,headSha,status,conclusion,event \
  --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "'"${task19642_expected_event}"'" and .status == "completed" and .conclusion == "success") | .databaseId' \
  | head -n 1)"
test -n "${task19642_run_id}"
task19642_evidence_dir="/tmp/task-19642-4-native-evidence-${task19642_run_id}"
test ! -e "${task19642_evidence_dir}"
mkdir "${task19642_evidence_dir}"
gh run download "${task19642_run_id}" --dir "${task19642_evidence_dir}"
```

Resolve `task19642_run_id` from the completed workflow whose head SHA matches
the frozen commit. Do not reuse artifacts from another run or attempt.

- [ ] **Step 4: Validate every platform document separately**

Run all five commands and require exit zero from each:

```bash
task19642_expected_event=pull_request
task19642_frozen_sha="$(git rev-parse HEAD)"
task19642_run_id="$(gh run list \
  --workflow task-602-platform-evidence.yml \
  --branch codex/task-19642-4-stt-smoke-clock \
  --limit 20 \
  --json databaseId,headSha,status,conclusion,event \
  --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "'"${task19642_expected_event}"'" and .status == "completed" and .conclusion == "success") | .databaseId' \
  | head -n 1)"
test -n "${task19642_run_id}"
task19642_evidence_dir="/tmp/task-19642-4-native-evidence-${task19642_run_id}"
python .github/scripts/task602_platform_evidence.py --validate \
  "${task19642_evidence_dir}/task-602-platform-linux-x86_64/task-602-platform-evidence.json"
python .github/scripts/task602_platform_evidence.py --validate \
  "${task19642_evidence_dir}/task-602-platform-linux-aarch64/task-602-platform-evidence.json"
python .github/scripts/task602_platform_evidence.py --validate \
  "${task19642_evidence_dir}/task-602-platform-windows-x86_64/task-602-platform-evidence.json"
python .github/scripts/task602_platform_evidence.py --validate \
  "${task19642_evidence_dir}/task-602-platform-macos-arm64/task-602-platform-evidence.json"
python .github/scripts/task602_platform_evidence.py --validate \
  "${task19642_evidence_dir}/task-602-platform-macos-x86_64/task-602-platform-evidence.json"
```

- [ ] **Step 5: Aggregate only the five same-run files**

```bash
task19642_expected_event=pull_request
task19642_frozen_sha="$(git rev-parse HEAD)"
task19642_run_id="$(gh run list \
  --workflow task-602-platform-evidence.yml \
  --branch codex/task-19642-4-stt-smoke-clock \
  --limit 20 \
  --json databaseId,headSha,status,conclusion,event \
  --jq '.[] | select(.headSha == "'"${task19642_frozen_sha}"'" and .event == "'"${task19642_expected_event}"'" and .status == "completed" and .conclusion == "success") | .databaseId' \
  | head -n 1)"
test -n "${task19642_run_id}"
task19642_evidence_dir="/tmp/task-19642-4-native-evidence-${task19642_run_id}"
python .github/scripts/task602_platform_evidence.py --aggregate \
  "${task19642_evidence_dir}/task-602-platform-linux-x86_64/task-602-platform-evidence.json" \
  "${task19642_evidence_dir}/task-602-platform-linux-aarch64/task-602-platform-evidence.json" \
  "${task19642_evidence_dir}/task-602-platform-windows-x86_64/task-602-platform-evidence.json" \
  "${task19642_evidence_dir}/task-602-platform-macos-arm64/task-602-platform-evidence.json" \
  "${task19642_evidence_dir}/task-602-platform-macos-x86_64/task-602-platform-evidence.json" \
  --output Docs/STT_Evaluation/task-602/platform-evidence.json
python .github/scripts/task602_platform_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-602/platform-evidence.json
```

- [ ] **Step 6: Refresh the evidence README**

Record the tested commit, workflow URL/run/attempt, resolved package versions,
five passing lanes, unchanged artifact identities, and why this refresh was
required. Preserve fixture attribution and scope limits.

### Task 6: Document and close TASK-19642.4

**Files:**

- Modify: `backlog/docs/task-19520-verification-failure-inventory.md`
- Modify: `backlog/docs/lessons-testing-evidence.md`
- Modify:
  `backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md`

- [ ] **Step 1: Update the permanent failure inventory**

Replace the low-confidence/truncated-trace entry with:

- isolated and file-level passing controls;
- deterministic `StopIteration` traceback at the third smoke clock read;
- proof that the old patch replaced the process clock;
- the private-binding repair; and
- interpreter, CI, and native evidence results.

- [ ] **Step 2: Record the general testing lesson**

Add an incident-based entry explaining that patching an attribute on an
imported standard-library module mutates that shared module process-wide. Name
TASK-19642.4, the three-value iterator, the extra-consumer `StopIteration`, and
the safer pattern of rebinding the owner module's imported name to a fake.

- [ ] **Step 3: Complete the Backlog task**

Add concise implementation notes covering approach, RED/GREEN evidence,
platform results, native run identity, modified files, ADR decision, and the
lesson. Check all acceptance criteria and set `status: Done` only after every
gate above passes.

Do not call `backlog task edit 19642.4`: the repository's recorded Backlog CLI
bug for five-digit IDs can create `task-task- - .md` instead of editing the
target. Edit the source-of-truth task file directly, then verify it exactly:

```bash
rg -n '^status: Done$|^- \[x\] #' \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md'
test "$(rg -c '^status: Done$' \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md')" -eq 1
test "$(rg -c '^- \[x\] #' \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md')" -eq 3
if rg -n '^- \[ \] #' \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md'; then
  exit 1
fi
rg -q '^## Implementation Notes$' \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md'
test ! -e 'backlog/tasks/task-task- - .md'
```

Expected: one `status: Done`, all three ACs checked, no unchecked AC, and no
malformed CLI artifact.

- [ ] **Step 4: Verify final evidence and documentation**

```bash
python -m pytest -q \
  Tests/STT/test_task602_platform_smoke.py \
  Tests/CI/test_task602_platform_evidence.py \
  --tb=short
python .github/scripts/task602_platform_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-602/platform-evidence.json
python -m ruff check Tests/STT/test_task602_platform_smoke.py
python -m ruff format --check Tests/STT/test_task602_platform_smoke.py
git diff --check origin/dev...HEAD
git diff --check
```

Expected: focused tests and validation pass, Ruff passes, the committed range
and current working-tree changes contain no whitespace errors. Performance and
licence gates are scoped N/A because no production/dependency file changes;
security is covered by the unchanged bounded/path-private assertion gate.

- [ ] **Step 5: Commit the evidence-only closeout**

```bash
git add \
  Docs/STT_Evaluation/task-602/platform-evidence.json \
  Docs/STT_Evaluation/task-602/README.md \
  backlog/docs/task-19520-verification-failure-inventory.md \
  backlog/docs/lessons-testing-evidence.md \
  'backlog/tasks/task-19642.4 - Diagnose-the-task-602-STT-platform-smoke-error.md'
git commit -m "docs(stt): refresh task 602 platform evidence"
git diff --check origin/dev...HEAD
git status --short
```

Expected: the final range check passes and status is empty. Push the
evidence-only commit. If any executable file changes after the native run,
invalidate the aggregate and return to Task 4 before merging.
