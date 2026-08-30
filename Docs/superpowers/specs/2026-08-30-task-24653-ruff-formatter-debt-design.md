# TASK-24653 Current-Dev Ruff Formatter Debt Design

**Status:** approved by the owner on 2026-08-30

**Task:** `TASK-24653`

## Goal

Replace the stale, feature-local statement "61 inherited formatter failures" with
an exact census of current `origin/dev`, explain every change from that historical
set, and assign the remaining paths once to conflict-safe atomic cleanup tasks.

This characterization task changes no Python source. It produces the evidence and
Backlog boundaries that later formatter-only tasks execute.

## Pinned Evidence

The initial current-development pin is:

- Git revision: `d2ff9c05ca91d7f7b7be80a2401f78f7142e1aff`
- Ruff: `0.15.22`
- Universe: Python paths tracked by Git at the pinned revision

TASK-22514 supplies the historical comparison, not the current truth. Its scoped
closeout used implementation base `31ed49bb368f54211d6482599e00a5c1340f80b2`
and pre-closeout census `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`.
The historical residue is the 61-path intersection that failed Ruff 0.15.22 at
both revisions after TASK-22514 removed its own 16 regressions.

If `origin/dev` advances before the census and child-task records are committed,
the task rebases, updates the pin, and reruns the census. It never combines results
from different revisions.

## Chosen Approach

Use owner-aligned batches. Group remaining paths by subsystem and likely test
surface, then separate files under active concurrent ownership. Large or unusually
high-risk files may form a batch by themselves. Every failing path belongs to
exactly one child task.

Rejected alternatives:

- Fixed-size batches are easy to count but split shared test surfaces and ignore
  merge-conflict risk.
- One repository-wide formatting PR is mechanically simple but creates excessive
  review churn and conflicts with active feature branches.

## Census and Comparison

The current census runs Ruff against the explicit Git-tracked Python path list.
It records the exact failing set rather than parsing an unrestricted filesystem
walk that could include worktrees, generated files, or untracked scratch.

The plan records three disjoint comparisons:

- paths still present in both the historical residue and current failures;
- historical failures no longer current, with deletion or formatting lineage;
- current failures absent from the historical residue, with introduction lineage.

Deleted paths need no cleanup task. Renames follow the current path and record the
old identity. Any parse or configuration error is a blocker, not formatter debt,
and is filed separately rather than hidden in a formatting batch.

## Cleanup Task Contract

Each child task is independently mergeable and must:

1. rebase onto current `origin/dev` and reproduce its assigned failures;
2. run Ruff formatting on only its assigned paths;
3. prove parsed Python AST equality before and after formatting;
4. prove comment-token text is unchanged;
5. run Ruff lint and format checks on every touched Python path;
6. run focused tests selected from the files' owning subsystem;
7. pass `git diff --check` and the Backlog task-ID guard; and
8. contain no hand-written production behavior change.

The last child depends on all earlier cleanup tasks and owns the final explicit
Git-tracked repository-wide Ruff format check. A new failure introduced after the
characterization pin is either assigned through an amended census or remains a
separate feature regression; it is not silently absorbed.

## Records and Scope

The implementation plan is the compact manifest: it lists the pinned revision,
current failing paths, historical comparison, batch membership, child task IDs,
and verification commands. No new generator, runtime dependency, or permanent
formatter-baseline file is introduced.

TASK-24653 itself modifies only its Backlog record, this design, the plan, and the
child Backlog records. It does not run the local full test suite because it changes
no executable code.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this work characterizes and schedules behavior-preserving mechanical
formatting. It changes no runtime, dependency, storage, security, privacy, schema,
or cross-module ownership decision.
