# CI admission across pull requests

Status: approved 2026-10-30: bounded full feedback. This follows PR #3316. Cost reporting
is an independent slice (TASK-38100), and the shared composer failure already
has a tested repair in PR #3308 / TASK-38061.

## Decision: bounded full feedback (TASK-38101)

Keep preflight on every update. Move the seven existing application job
definitions into one local reusable workflow, preserving selectors, dependencies,
timeouts and all 28 UI shards. The caller still owns the unchanged required
check name, verdicts and queue tick. The called workflow exposes each lane result;
the aggregate rejects non-successful qualification or any selected lane result.
No successful aggregate may arise from a skipped/cancelled wrapper.

The call holds one native Actions concurrency slot for its entire qualification:
one `queue-front` slot and one `background` slot. Both use `queue: max` and never
cancel useful in-progress evidence. Up to 100 pending calls can wait per slot;
overflow remains visibly cancelled/failed and may be retried, never greened.
This bounds concurrent qualifications without a database, polling runner or
custom lease service. No new credentials or default-branch trigger is needed.

Classify the current head using a read-only, paginated live queue snapshot at
the end of preflight, matching the existing queue eligibility and ordering. Only
the live front receives the reserved class; API uncertainty falls back to the
background class and is disclosed. The class is a scheduling snapshot, not an
authorization or merge decision. A later front change cannot preempt useful work;
the next front can wait behind an already admitted former front.

Use UI max-parallel 17 for the front and 8 for background. Together with six
other application jobs each, the two qualifications make at most 37 application
jobs runnable simultaneously. Preflight, aggregate, queue operations, other
workflows/repositories and old PR workflow versions remain outside that bound.
This is not an account-wide runner reservation. Full tests still run on ordinary
updates, drafts and manual full-gate requests; draft opt-in is deferred rather
than silently changing feedback policy.

Tradeoff: background qualification takes more waves, and its waiting call can
occupy the slot while GitHub has no runner. It avoids hundreds of competing test
jobs and gives the observed front a separate path without changing coverage.
Moving jobs requires updating source-based workflow contracts and checking any
job-name consumers (for example shard replay) against reusable-workflow naming.

## Rejected alternative: qualify only for merging or explicit requests

Run preflight automatically; run expensive tests only for the queue front or an
explicit full-CI request. This avoids more speculative work, but authors lose
automatic full regression feedback before arming their PR. It also needs a
reliable PR-associated restart path for an already-current front: dispatching a
new required workflow does not count toward this repository's protection.
The existing approve/rerun semantics must remain, and unqualified PRs cannot
receive the required green context. This is a larger change to author workflow.

## Validation and rollout

Require exact selector/dependency conservation, fail-closed wrapper/result tests,
front eligibility/order/race tests, pending-overflow disclosure, and unchanged
failed-job-only retry handling. Hosted qualification must verify native queued
calls, job names, PR-associated required checks and retry results; local workflow
contracts alone do not prove GitHub scheduler behavior. Compare runner minutes
and waiting time against the audit after rollout. Existing PRs only adopt the
new graph when rebased; do not promise an immediate account-wide cap.

ADR required: yes, amend ADR-103 and ADR-218 before implementation.

Sources checked 2026-10-29:
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
- https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations
