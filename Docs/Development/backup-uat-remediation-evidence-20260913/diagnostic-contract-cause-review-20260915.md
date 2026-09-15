# dc96 Fast Lane contract failure review

Read-only review; no repository changes, Models/UI tests, app boots, or workflow runs. Proposed patch only: `/private/tmp/uat-dc96-fastlane-contract-suggestion.patch`.

**Finding:** the approved backup startup diagnostic changed in commit `9dff9dd1b2a73795eece24970ac43f1dc571c2b6`, but its exact CI contract was not updated. This is a backup-diagnostic integration regression already present on PR HEAD, not a new Models fixture requirement or a merge-only discrepancy.

## Exact tested source

PR2642 HEAD is `dc96b849dc4f05e64843572016c63f645753b150`. Fast Lane run34942519054/job104294249791 checked out genuine PR merge `783d71b4ef608e53bbe14e4702f12d25f36683cc`, as explicitly recorded by checkout and `git log -1`. Its parents are dev `4e4558bff21a6c158432df4e08dad73b832eea79` and dc96. The API's presently reported dev tip has since advanced to `2f97a42c9aa9cc737cf304f927e6d861824e1b3a`; that is not the tested merge parent.

Read the two files directly from that genuine merge through GitHub's contents API and compared decoded bytes to the local HEAD files:

- `Tests/CI/test_task2062_2_gguf_source_evidence.py`: identical; Git blob `6466e441ae18237c290784f9f1f2bef690a78a26`; SHA256 `daa75e510fd231e1dba7b20c21b3593fca12574af423ed25890a67ca3fbd812d`.
- `.github/workflows/task-2062-2-gguf-source-evidence.yml`: identical; Git blob `213b8a53924cdd9b1bc9b3c3333c2d8c828875ba`; SHA256 `3c1ef3c5a2bb684dab750f28ade471453f6018de53a9f8483b3c235ec5bbe084`.

Full job evidence: `/private/tmp/uat-dc96-fastlane-full.log`; API receipts `/private/tmp/uat-dc96-fastlane-{pr,merge-ref,merge-commit,merge-test,merge-workflow}.json`. Final actual result: **1128 passed, 2 failed, 1 warning in706.21s**.

## Failure and missing follow-up

The only failing cases are `test_workflow_is_one_read_only_exact_three_os_matrix` (line129) and `test_windows_failure_diagnostic_keeps_exact_read_only_commands` (line164→90). Both immediately encounter the obsolete step name, `Diagnose Windows backup startup against the dev source baseline`, instead of the actual `Compare Windows keyboard waits with the dev source baseline`.

Changing the expected name alone is insufficient: lines169–181 still expect one claim/recompose node, dev then candidate, with profiling. The actual workflow at91–106 loops over the two existing provider keyboard cases, dev then candidate for each, using `--no-profile`. Commit9dff introduced exactly those changes under TASK32562, described as observing remaining Windows backup startup/keyboard waits without cProfile overhead and without changing product behavior, assertions or deadlines. The contract last changed in `c3ad686072315bed0f9aca4469d0c9e7ddc6f268` for the preceding diagnostic shape.

## Minimal reviewable correction

Update **only** the test's expected final step name and exact command-token tuple: retain pinned baseline4631b60, fetch/worktree/helper path, failure accumulation, the exact two existing `REQUIRED_UI_NODES[2:4]` in order, dev-before-candidate source/label pairs, required `--no-profile` on each, loop closure and exit status. The attached proposed patch makes only those expectation changes.

Keep all strict matrix/read-only/checkout/permissions checks, failure-only Windows condition, one diagnostic `continue-on-error`, original evidence-node selection, dependency restrictions, forbidden commands, job20-minute limit and native primary60-second deadline. Do not broaden the test to accept arbitrary steps/commands, revert the approved diagnostic, or touch `_mount_models` or any Models assertion.

After separately authorizing the correction, the finite relevant validation is the five CI-contract cases plus mutation checks rejecting a wrong baseline/node/order, removed no-profile, altered diagnostic condition or added permissive primary step. None was executed in this review. The separate supported Models-initial fixture proposal remains unapplied and outside this patch; its automatic approval rejection and pending user scope decision are unchanged.
