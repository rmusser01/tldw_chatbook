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
- Interpreter and Ruff executable: the repository virtual environment at
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff`
- Universe: Python paths tracked by Git at the pinned revision
- Configuration: the Ruff configuration committed at the pinned revision

TASK-22514 supplies the historical comparison, not the current truth. Its scoped
closeout used implementation base `31ed49bb368f54211d6482599e00a5c1340f80b2`
and pre-closeout census `1f4f72ac5ff02f5237a4946745e82e8932cd41cf`.
The historical residue is reconstructed by running the same per-path census at
both pinned revisions and intersecting their failing sets. The reconstruction must
assert cardinality 61 before it is accepted as the historical comparison set.

Every census runs in an isolated clean worktree or checkout at its exact revision.
The executable must report exactly Ruff 0.15.22 before any result is retained. If
`origin/dev` advances before the characterization records are committed, the task
rebases, updates the pin, and reruns the current census. It never combines path
lists, contents, or configuration from different revisions.

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

The current census enumerates Git-tracked Python paths with NUL-delimited Git output
inside the exact checkout. A standard-library Python driver consumes those bytes,
prefixes each repository-relative path with `./`, and invokes Ruff once per path.
This avoids shell word splitting, leading-dash ambiguity, and command-length limits.
Each invocation uses `ruff format --check --force-exclude`, so explicit paths retain
the revision's repository exclusions.

Exit zero means clean and exit one means Ruff 0.15.22 would reformat the path. Any
other exit is a blocker captured with its repository-relative path and diagnostic
category. It is not silently classified as formatter debt. The aggregate repository
command is also run once as a control, but the per-path exit code is the manifest
source; human output is not parsed to decide membership.

The plan records three disjoint comparisons:

- paths still present in both the historical residue and current failures;
- historical failures no longer current, with deletion or formatting lineage;
- current failures absent from the historical residue, with introduction lineage.

Deleted paths need no cleanup task. Renames explicitly pair the historical removal
with the current addition and identify the rename lineage; they cannot appear as two
unrelated explanations. Any parse or configuration error is a blocker, not formatter
debt, and is filed separately rather than hidden in a formatting batch.

The sorted historical candidate set, sorted current failure set, comparison sets,
blockers, rename mappings, and stable batch labels are persisted in one point-in-time
JSON evidence file. A one-shot standard-library checker proves that batch unions
equal the current failure set, batches are pairwise disjoint, and blocker or excluded
paths occur in no batch.

## Cleanup Task Contract

Each child task is independently mergeable and must:

1. rebase onto current `origin/dev` and reproduce its assigned failures;
2. run Ruff formatting on only its assigned paths;
3. parse each file before and after with the supported Python version and
   `type_comments=True`, then compare `ast.dump(..., include_attributes=False)`;
4. prove the ordered comment-token text is unchanged and additionally preserve the
   code-line attachment of `# noqa`, `# type: ignore`, `# ruff:`, and Ruff-format
   directives;
5. run Ruff lint and format checks on every touched Python path;
6. record the rationale for its focused test selection plus exact commands and
   results;
7. pass `git diff --check` and the Backlog task-ID guard; and
8. contain no hand-written production behavior change.

After rebasing, an assigned path that is deleted, renamed, modified, or already clean
must be reconciled in that cleanup record before formatting. The record identifies
the upstream lineage, preserves or amends exact path ownership, and reruns its
mechanical proof. It never silently drops a path or absorbs an unassigned one.

The final cleanup record is created after the earlier cleanup records so its
dependencies only point to lower task IDs. It owns the final explicit Git-tracked
repository-wide Ruff format check. A new unassigned failure blocks that gate. The
series either waits for its independently owned correction or creates a superseding
final record after the new correction record exists; the final task never absorbs
the regression as formatting debt.

## Records and Scope

The point-in-time JSON evidence is the mechanically auditable manifest. The
implementation plan lists the pin, comparison counts, stable batch labels, evidence
path, and verification commands. It does not list future cleanup task IDs: later
cleanup records reference TASK-24653, while the final cleanup record may depend only
on earlier-created cleanup records. No generator, runtime dependency, or permanent
formatter-baseline file is introduced.

TASK-24653 itself modifies only its Backlog record, this design, the plan, the
point-in-time evidence, and newly created cleanup Backlog records. Its completion
requires those records and their contracts to exist; it does not require the future
cleanup work to execute. It does not run the local full test suite because it changes
no executable code.

## ADR Check

ADR required: no.

ADR path: N/A.

Reason: this work characterizes and schedules behavior-preserving mechanical
formatting. It changes no runtime, dependency, storage, security, privacy, schema,
or cross-module ownership decision.
