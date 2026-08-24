# RAG Real-Embedding Capability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the thirteen TASK-19642.3 inventory nodes complete offline without errors by skipping the real-model benchmark before transformer initialization unless its existing capability flag is enabled.

**Architecture:** Add one combined pytest skip marker at the existing `Tests/RAG_Search/conftest.py` marker seam. Apply it and the descriptive integration marker to the one real-model performance test; leave production embeddings, the real-model fixture, and the repository network guard unchanged.

**Tech Stack:** Python 3.12 test environment, pytest markers/fixtures, Ruff, Backlog.md

---

## File Map

- Modify `Tests/RAG_Search/conftest.py`: expose the existing real-embedding environment decision to a combined dependency/capability skip marker.
- Modify `Tests/RAG_Search/test_embeddings_performance.py`: apply that marker and classify the real benchmark as integration coverage.
- Modify `backlog/tasks/task-19642.3 - Keep-RAG-tests-offline-during-embedding-initialization.md`: track plan, evidence, acceptance criteria, and closeout notes.
- Reference `backlog/docs/lessons-testing-evidence.md`: its existing “exact live-test gate must be the first gate” lesson already covers the general trap; update only if implementation reveals distinct new evidence.

## Scope Boundaries

- Do not modify production embedding code.
- Do not add `allow_network` or weaken `Tests/conftest.py::_no_network_io`.
- Do not mock the real-model benchmark.
- Do not change `real_transformers_session` or the opt-in real-integration suite.
- Do not run broad RAG or repository suites; the user limited verification to modified functionality.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a test-only correction applying the existing capability and network-guard policy. It introduces no production, security, storage, dependency, or cross-module contract.

### Task 1: Gate the real-model benchmark before fixture setup

**Files:**

- Modify: `Tests/RAG_Search/conftest.py:20-32,641-660`
- Modify: `Tests/RAG_Search/test_embeddings_performance.py:20,152-164`
- Test: `Tests/RAG_Search/test_embeddings_performance.py::TestEmbeddingPerformance::test_real_model_performance`

- [ ] **Step 1: Establish the isolated control**

Run from the task worktree:

```bash
env -u TLDW_RUN_REAL_EMBEDDINGS -u TLDW_TEST_ALLOW_HF_DOWNLOADS \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/RAG_Search/test_embeddings_performance.py::TestEmbeddingPerformance::test_real_model_performance \
  --tb=short -rs
```

Expected: one ordinary model-unavailable skip, zero errors, and no guard-reported
attempts. This control proves the defect depends on earlier RAG imports freezing
Hugging Face state; the isolated node is not the RED reproducer.

- [ ] **Step 2: Reproduce the order-dependent RED with the exact inventory**

```bash
nodes=()
while IFS= read -r line; do
  nodes+=("$(jq -r . <<< "$line")")
done < <(sed -n '568,580p' backlog/docs/task-19520-verification-failure-inventory.md)
env -u TLDW_RUN_REAL_EMBEDDINGS -u TLDW_TEST_ALLOW_HF_DOWNLOADS \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  "${nodes[@]}" --tb=short -rs
```

Expected: `12 passed, 1 skipped, 1 error`; `_no_network_io` reports blocked
`huggingface.co:443` attempts from the tiny-BERT warm-up and MiniLM
initialization. The captured baseline reports eight, but any nonzero count is
the required RED; the retry count is not the contract.

- [ ] **Step 3: Reuse the parsed capability decision in the RAG_Search conftest**

Change the existing environment block to name the real-embedding decision once:

```python
_TRUTHY = {"1", "true", "yes", "on"}
_RUN_REAL_EMBEDDINGS = (
    os.environ.get("TLDW_RUN_REAL_EMBEDDINGS", "").strip().lower() in _TRUTHY
)
_ALLOW_HF_DOWNLOADS = (
    _RUN_REAL_EMBEDDINGS
    or os.environ.get("TLDW_TEST_ALLOW_HF_DOWNLOADS", "").strip().lower()
    in _TRUTHY
)
```

Do not change the subsequent offline environment assignments.

- [ ] **Step 4: Add one combined dependency/capability marker**

Add beside `requires_embeddings`:

```python
requires_real_embeddings = pytest.mark.skipif(
    not DEPENDENCIES_AVAILABLE.get("embeddings_rag", False)
    or not _RUN_REAL_EMBEDDINGS,
    reason=(
        "Embeddings dependencies not available"
        if not DEPENDENCIES_AVAILABLE.get("embeddings_rag", False)
        else "TLDW_RUN_REAL_EMBEDDINGS is not enabled"
    ),
)
```

This single marker preserves the existing dependency reason and otherwise identifies the explicit capability that is absent.

- [ ] **Step 5: Apply the marker before the benchmark body can request fixtures**

Replace the conftest import and decorator in `test_embeddings_performance.py`:

```python
from .conftest import requires_chromadb, requires_real_embeddings
```

```python
    @pytest.mark.integration
    @requires_real_embeddings
    def test_real_model_performance(self, request):
```

Do not modify the benchmark body. `integration` is descriptive; `requires_real_embeddings` is the setup-time behavioral gate.

- [ ] **Step 6: Run the exact inventory as GREEN**

Run the Step 2 command unchanged.

Expected: `12 passed, 1 skipped`, zero errors, capability skip reason
`TLDW_RUN_REAL_EMBEDDINGS is not enabled`, no transformer session-start output,
and no `huggingface.co` guard attempts.

- [ ] **Step 7: Run file-level static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/RAG_Search/conftest.py Tests/RAG_Search/test_embeddings_performance.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/RAG_Search/conftest.py Tests/RAG_Search/test_embeddings_performance.py
```

Expected: both commands exit 0. The latest-dev baseline already reports both files formatted.

- [ ] **Step 8: Commit the implementation**

```bash
git add Tests/RAG_Search/conftest.py Tests/RAG_Search/test_embeddings_performance.py
git commit -m "test(rag): gate real embedding benchmark"
```

### Task 2: Verify the exact inventory and close the backlog task

**Files:**

- Modify: `backlog/tasks/task-19642.3 - Keep-RAG-tests-offline-during-embedding-initialization.md`
- Optional modify only for distinct new evidence: `backlog/docs/lessons-testing-evidence.md`
- Test inventory: `backlog/docs/task-19520-verification-failure-inventory.md:568-580`

- [ ] **Step 1: Run all thirteen exact inventory nodes under the blocked guard**

Use the JSON-quoted node IDs in the canonical inventory so the long parametrized chunking node remains exact:

```bash
nodes=()
while IFS= read -r line; do
  nodes+=("$(jq -r . <<< "$line")")
done < <(sed -n '568,580p' backlog/docs/task-19520-verification-failure-inventory.md)
env -u TLDW_RUN_REAL_EMBEDDINGS -u TLDW_TEST_ALLOW_HF_DOWNLOADS \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  "${nodes[@]}" --tb=short -rs
```

Expected: `12 passed, 1 skipped`, zero failures/errors, capability skip reason `TLDW_RUN_REAL_EMBEDDINGS is not enabled`, and no guard-reported `huggingface.co` attempts.

- [ ] **Step 2: Re-run static and revision hygiene gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/RAG_Search/conftest.py Tests/RAG_Search/test_embeddings_performance.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/RAG_Search/conftest.py Tests/RAG_Search/test_embeddings_performance.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: Ruff and diff-check exit 0. Status may contain only the planned task/plan closeout edits before their commit.

- [ ] **Step 3: Self-review against the inverse failure and scope boundaries**

Confirm from the diff and RED/GREEN evidence that:

- removing the capability marker restores the nonzero-attempt teardown error;
- the integration marker does not authorize network;
- the dependency-missing reason remains unchanged;
- no production, global network guard, fixture body, or unrelated test changed.

- [ ] **Step 4: Complete task hygiene**

Use Backlog.md to check all three acceptance criteria, add concise Implementation Notes containing the RED/GREEN counts and modified files, and mark TASK-19642.3 Done only after every scoped gate is green. Link the existing live-gate lesson in the notes; do not add a duplicate lesson unless implementation produces materially different evidence.

- [ ] **Step 5: Commit closeout documentation**

```bash
git add 'backlog/tasks/task-19642.3 - Keep-RAG-tests-offline-during-embedding-initialization.md'
git commit -m "docs(rag): close TASK-19642.3"
```

- [ ] **Step 6: Final clean-tree verification**

Re-run the exact thirteen-node command from Step 1, both Ruff commands from Step 2, `git diff --check origin/dev...HEAD`, and `git status --short`.

Expected: `12 passed, 1 skipped`, both Ruff commands and diff-check exit 0, and the worktree is clean.
