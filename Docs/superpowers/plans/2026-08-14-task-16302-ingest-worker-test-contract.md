# Ingest Worker Test Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the test baseline by aligning one stale assertion with the current parse-worker request identity contract.

**Architecture:** Production remains unchanged. The existing App test will unpack the worker call's third argument and assert the exact `(generation, job_id)` identity that production already supplies.

**Tech Stack:** Python 3.11+, pytest, Ruff.

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a test-only correction to an existing call contract.

---

### Task 1: Repair and verify the stale assertion

**Files:**
- Modify: `Tests/App/test_submit_library_ingest_job.py:2473-2476`
- Modify: `backlog/tasks/task-16302 - Repair-stale-ingest-worker-test-contract.md`
- Test: `Tests/App/test_submit_library_ingest_job.py`

- [ ] **Step 1: Confirm the seven-case RED baseline**

Run:

```bash
../../.venv/bin/python -B -m pytest -q --tb=short \
  'Tests/App/test_submit_library_ingest_job.py::test_invalid_audio_request_allows_next_job_to_dispatch'
```

Expected: seven failures at the stale two-item unpack with
`ValueError: too many values to unpack`.

- [ ] **Step 2: Update the assertion minimally**

Replace the stale unpack with:

```python
_, (source_path, options, progress_context), _, _ = pool.calls[0]
assert source_path == valid.source_path
assert progress_context == (app._ingest_parse_pool_generation, valid.job_id)
```

Keep the existing transcription-provider assertion. Do not modify production.

- [ ] **Step 3: Run focused GREEN verification**

Run the seven-case node, then the complete App test module:

```bash
../../.venv/bin/python -B -m pytest -q --tb=short \
  'Tests/App/test_submit_library_ingest_job.py::test_invalid_audio_request_allows_next_job_to_dispatch'
../../.venv/bin/python -B -m pytest -q --tb=short \
  Tests/App/test_submit_library_ingest_job.py
```

Expected: both commands pass.

- [ ] **Step 4: Run static and scope gates**

```bash
../../.venv/bin/ruff check Tests/App/test_submit_library_ingest_job.py
! ../../.venv/bin/ruff format --check --diff \
  Tests/App/test_submit_library_ingest_job.py
git diff --check
git status --short
git diff --name-only origin/dev
git diff --name-only origin/dev -- tldw_chatbook
set -o pipefail
! git diff origin/dev | rg -n \
  '/User[s]/|Authorizatio[n]:|Beare[r][[:space:]]|MINIMAX_API_KE[Y]=|BEGIN [A-Z ]*PRIVATE KE[Y]'
```

Expected: lint passes. The formatter reports the existing whole-file dev
baseline and its diff leaves the new unpack/assertion unchanged; do not format
unrelated legacy regions. The only changed paths are this plan, the approved
spec, TASK-16302, and the App test; the production-path query and privacy scan
emit nothing. Inspect the complete diff for accidental behavior or
secret-bearing content.

- [ ] **Step 5: Run the full-suite fail-fast gate**

```bash
../../.venv/bin/python -B -m pytest -q -x --tb=short
```

Expected: the full suite passes. If it fails, leave TASK-16302 In Progress and
stop; do not label the failure unrelated without separate pristine-base
reproduction and a revised plan.

- [ ] **Step 6: Close the task and commit**

Self-review the exact diff for scope, assertion strength, privacy, and test
vacuity. Then record the approach, modified behavior, trade-off, exact files,
verification, and ADR result in Implementation Notes; check both acceptance
criteria and set TASK-16302 to Done through Backlog CLI.

Stage and verify only the exact implementation paths:

```bash
git add \
  Docs/superpowers/plans/2026-08-14-ingest-worker-test-contract.md \
  Tests/App/test_submit_library_ingest_job.py \
  'backlog/tasks/task-16302 - Repair-stale-ingest-worker-test-contract.md'
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m 'test: align ingest worker request identity'
```

Expected: the staged diff contains no production file and the commit contains
only the plan, test, and TASK-16302 closeout.
