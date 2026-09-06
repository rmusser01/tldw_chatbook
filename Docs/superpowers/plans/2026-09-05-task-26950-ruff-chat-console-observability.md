# TASK-26950 Console observability formatter cleanup

Apply the approved TASK-26000 formatter contract to the exact 20 paths in TASK-26950 (15 tests, five production modules). Initial integration base: `d3bf8b5397c9f92cf9fcc722193f75665cc0192a`. Authority cut: `e555df102c950c29beed5e7119f433d35eee1f3c`.

ADR required: no
ADR path: N/A
Reason: mechanical formatting under the existing TASK-26000 contract; no architecture or behavior changes.

1. Reconcile the assigned manifest, digest, and upstream lineage. Capture structural evidence and run the 15 assigned test modules before formatting.
2. Apply Ruff 0.15.22 only to assigned files. Verify type-comment-aware AST and comment/directive parity, deterministic base-blob replay, lint, format, and identical focused-test results using Python 3.12.11.
3. Verify persistent diagnostic inventory and Backlog uniqueness; record exact commands/results and review the complete diff. Mark the task Done only after its acceptance criteria are met.
4. Rebase on current dev if necessary, repeat affected verification, publish a PR, address Qodo feedback, wait for required CI, merge, and clean up the isolated worktree.

The focused test command uses all 15 task-owned test modules. Structural parity and deterministic replay provide the mutation-sensitive evidence for this formatting-only change; no new behavioral test or full-suite run is planned.

Owner-approved amendment: fix the two E741 generator variable names and one F811 redundant import in the assigned test files. Prove the six identifier substitutions and one import-alias removal explicitly; retain exact AST parity elsewhere and formatter parity against the corrected baseline. The inherited citation failures are assessed by complete before/after outcome parity with identical bounded test commands.
