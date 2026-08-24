# TASK-3501 Pipeline Fusion Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the original FTS and vector scores in every Chat-RAG pipeline hybrid result without changing fusion, citations, or non-hybrid behavior.

**Architecture:** Extend the existing real pipeline regression to cover overlapping, FTS-only, and vector-only results, then mirror the engine materializer's snapshot-before-mutation order inside `_rrf_merge_parallel_results`. Keep the caller-specific materializers separate and add no helper, dependency, schema, configuration, or new public type.

**Tech Stack:** Python 3.11+, pytest, existing `SearchResult` and reciprocal-rank-fusion utilities.

---

## File Map

- Modify `Tests/RAG/test_fusion.py`: make the existing pipeline fixture expose distinct raw leg scores and assert complete provenance for every row shape.
- Modify `tldw_chatbook/RAG_Search/pipeline_builder_simple.py`: snapshot the two optional leg scores before mutating the selected result and include them in `hybrid_fusion`.
- Modify `backlog/tasks/task-3501 - pipeline_builder_simple.py-hybrid-merge-has-the-same-leg-score-aliasing-bug-Task-2-fixed-in-rag_service.md`: check acceptance criteria, add implementation notes, and close the task after verification and review.
- Do not create a shared materializer or change `tldw_chatbook/RAG_Search/fusion.py`; the pure fusion primitive already preserves both input items.

### Task 0: Synchronize the implementation base

**Files:**

- Verify only; no planned file changes.

- [ ] **Step 1: Rebase the documentation commits onto current `origin/dev`**

Run:

```bash
git fetch origin dev
git rebase origin/dev
```

Expected: the design and plan commits replay cleanly. At plan-authoring time, `origin/dev` was three commits ahead but had no changes to the production, test, TASK-3501, or Superpowers paths in this plan. If a later fetch changes any of those paths, inspect that semantic delta and update this plan before touching code.

- [ ] **Step 2: Re-establish the clean fusion baseline**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_fusion.py -q --tb=short
git status --short --branch
```

Expected: the fusion suite passes and the worktree is clean before the RED test edit. Record the exact count and warnings.

- [ ] **Step 3: Record the touched-file formatter baseline**

Run before either Python file changes:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
```

Expected on untouched `origin/dev`: nonzero with both files reported as `Would reformat`. This is inherited whole-file formatting debt, not permission to reformat either file. Record the exact file set so the post-change check can require no branch-added formatter delta.

### Task 1: Pin all pipeline provenance shapes and implement the minimal fix

**Files:**

- Modify: `Tests/RAG/test_fusion.py:642-648,703-777`
- Modify: `tldw_chatbook/RAG_Search/pipeline_builder_simple.py:397-419`
- Reference: `tldw_chatbook/RAG_Search/simplified/rag_service.py:2939-2983`

- [ ] **Step 1: Extend the existing test fixture with explicit raw scores**

Change the class-local helper so only this test class gains a score parameter:

```python
@staticmethod
def _pipeline_result(source, doc_id, *, score=1.0):
    from tldw_chatbook.RAG_Search.pipeline_types import SearchResult

    return SearchResult(
        source=source,
        id=doc_id,
        title=doc_id,
        content=f"content {doc_id}",
        score=score,
    )
```

In `test_parallel_step_rrf_merge_fuses_legs`, assign distinct scores to every row shape:

```python
media = [
    self._pipeline_result("media", "m1", score=0.11),
    self._pipeline_result("media", "shared", score=0.22),
]
conversations = [self._pipeline_result("conversation", "c1", score=0.33)]
notes = []
semantic = [
    self._pipeline_result("media", "shared", score=0.88),
    self._pipeline_result("media", "s2", score=0.77),
]
```

- [ ] **Step 2: Assert exact provenance for overlapping and single-leg rows**

After the existing fused-score and ordering assertions, retain the rank/alpha checks and add:

```python
by_key = {(result.source, result.id): result for result in results}
raw_leg_scores = {
    ("media", "m1"): (0.11, None),
    ("conversation", "c1"): (0.33, None),
    ("media", "shared"): (0.22, 0.88),
    ("media", "s2"): (None, 0.77),
}
for key, (fts_score, vector_score) in raw_leg_scores.items():
    result = by_key[key]
    fusion = result.metadata["hybrid_fusion"]
    assert fusion["fts_score"] == fts_score, key
    assert fusion["vector_score"] == vector_score, key
    for raw_score in (fts_score, vector_score):
        if raw_score is not None:
            assert result.score != pytest.approx(raw_score), key
```

- [ ] **Step 3: Run the regression alone and verify RED**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_fusion.py::TestPipelineRrfMerge::test_parallel_step_rrf_merge_fuses_legs -q --tb=short
```

Expected: FAIL on the untouched implementation with missing `fts_score`/`vector_score` in `hybrid_fusion`. Confirm the failure is the new provenance assertion, not fixture setup or import failure.

- [ ] **Step 4: Snapshot both optional leg scores before result mutation**

In `_rrf_merge_parallel_results`, add the two snapshots immediately after selecting `entry.item`, before `result.score` changes:

```python
for entry in fused_entries:
    result = entry.item
    fts_score = entry.fts_item.score if entry.fts_item is not None else None
    vector_score = entry.vector_item.score if entry.vector_item is not None else None
    result.score = entry.score
```

Expand only the existing fusion metadata mapping:

```python
"hybrid_fusion": {
    **entry.provenance(),
    "fts_score": fts_score,
    "vector_score": vector_score,
    "alpha": alpha,
    "rrf_k": rrf_k,
},
```

Do not change item selection, result keys, fusion math, citation merging, or any merge branch.

- [ ] **Step 5: Run the regression alone and verify GREEN**

Run the Step 3 command again.

Expected: `1 passed`.

- [ ] **Step 6: Prove both snapshot-order assertions discriminate**

Use `apply_patch` for each temporary mutation and restore it before continuing:

1. Move only the `fts_score` read below `result.score = entry.score`; run the Step 3 command and expect an exact raw-score mismatch on an FTS-present row.
2. Restore the FTS snapshot, move only the `vector_score` read below the mutation; rerun and expect an exact raw-score mismatch on `("media", "s2")`.
3. Restore both snapshots above the mutation and rerun once more; expect `1 passed`.

This proves the regression detects both selected-item alias directions rather than only the presence of new keys.

- [ ] **Step 7: Run the focused fusion and provenance suites**

Run:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RAG/test_fusion.py Tests/RAG/test_local_citation_capture.py Tests/RAG_Search/test_hybrid_fusion_metadata.py -q --tb=short
```

Expected: all selected tests pass. Record the exact count and any warnings; do not describe unrelated or unrun CI checks as green.

- [ ] **Step 8: Inspect the production diff and static hygiene**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --diff tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
git diff --check
git diff -U0 -- tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
git diff -U0 origin/dev...HEAD -- tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
git diff -- tldw_chatbook/RAG_Search/pipeline_builder_simple.py Tests/RAG/test_fusion.py
git status --short
```

Expected: Ruff lint passes. Ruff format-check retains the exact Task 0 baseline (`Would reformat` for the same two legacy files), and the format diff has no hunk overlapping a branch-added line in the applicable zero-context Git diff (working-tree diff before commit, `origin/dev...HEAD` after commit); do not apply its unrelated whole-file rewrite. No whitespace errors exist, production changes are confined to the two snapshots and two metadata entries, and tests are confined to the existing pipeline fusion regression/helper.

- [ ] **Step 9: Commit the tested behavior**

```bash
git add Tests/RAG/test_fusion.py tldw_chatbook/RAG_Search/pipeline_builder_simple.py
git commit -m "fix(rag): preserve pipeline hybrid leg scores"
```

### Task 2: Review and close TASK-3501

**Files:**

- Modify: `backlog/tasks/task-3501 - pipeline_builder_simple.py-hybrid-merge-has-the-same-leg-score-aliasing-bug-Task-2-fixed-in-rag_service.md`
- Review: the full `origin/dev...HEAD` diff

- [ ] **Step 1: Request an independent correctness review**

Use `superpowers:requesting-code-review` with the approved spec and this plan as requirements. Review the production/test commit against `origin/dev`, including raw-score correctness, absent-leg `None` semantics, mutation-test discrimination, downstream metadata compatibility, citation preservation, and non-hybrid isolation.

Expected: no unresolved Critical or Important findings. Address technically valid findings with focused tests before closeout.

- [ ] **Step 2: Commit any technically valid review fixes**

If review changes production or tests, repeat the relevant RED/GREEN and mutation checks, then run Task 1 Steps 7 and 8 and commit those changes separately:

```bash
git add Tests/RAG/test_fusion.py tldw_chatbook/RAG_Search/pipeline_builder_simple.py
git commit -m "fix(rag): address TASK-3501 review feedback"
```

If no code changes are needed, record that outcome and do not create an empty commit.

- [ ] **Step 3: Re-run verification after review**

Repeat Task 1 Steps 7 and 8 against the final source tree.

Expected: the focused suites pass and `git diff --check` is clean.

- [ ] **Step 4: Run the repository-wide local test gate**

Run from the repository root against the stable, fully committed source tree:

```bash
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=short
```

Expected: the full local suite passes. If it reports failures, inspect and rerun the exact failing nodes; any failure attributable to this branch must be fixed before closeout, while inherited failures must be reproduced on the exact `origin/dev` base and documented honestly. Do not use GitHub or other CI checks as work items.

- [ ] **Step 5: Complete acceptance criteria and implementation notes while the task remains In Progress**

With `apply_patch`:

- change all three acceptance criteria to `[x]`;
- add `## Implementation Notes` with the minimal approach, modified files, exact focused/full test evidence, static-analysis evidence, review result, and the explicit no-new-ADR decision;
- preserve the reviewed Implementation Plan, ADR check, and TASK-4110 rationale (the detailed design and plan remain the durable records if the CLI later normalizes free-form prose).

Run `backlog task 3501 --plain` and verify the task is still In Progress with all three ACs checked and complete notes before changing its status.

- [ ] **Step 6: Close the backlog task through the CLI**

Only after Step 5 is complete, run the CLI so the board transition is recorded:

```bash
backlog task edit 3501 -s Done
```

Immediately inspect both `backlog task 3501 --plain` and the task-file diff because this CLI has previously removed free-form sections. It must still render every AC checked, the Implementation Plan/ADR check, and the Implementation Notes. If the CLI removes a mandatory section, restore that section with `apply_patch`, re-render the task, and document the CLI normalization in the notes; do not omit required content merely to keep the CLI's rewrite.

- [ ] **Step 7: Verify task and repository hygiene**

Run:

```bash
backlog task 3501 --plain
git diff --check
git status --short
git diff origin/dev...HEAD --stat
```

Expected: TASK-3501 renders as Done with every AC checked; no nameless task file was created; only the intended task, design/plan, production, and test files differ from `origin/dev`.

- [ ] **Step 8: Commit closeout documentation**

```bash
git add "backlog/tasks/task-3501 - pipeline_builder_simple.py-hybrid-merge-has-the-same-leg-score-aliasing-bug-Task-2-fixed-in-rag_service.md"
git commit -m "docs(rag): close TASK-3501"
```

- [ ] **Step 9: Final verification before integration handoff**

Run Task 1 Step 7 once against the committed tree, plus:

```bash
git diff --check origin/dev...HEAD
git status --short --branch
git log --oneline origin/dev..HEAD
```

Expected: focused tests pass, the worktree is clean, and the branch contains the design, reviewed plan, implementation, and closeout commits only. Report exact evidence and explicitly state that CI checks were not used as work items.
