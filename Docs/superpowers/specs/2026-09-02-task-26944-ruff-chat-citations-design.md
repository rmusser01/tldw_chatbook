# TASK-26944 Ruff Chat Citations Cleanup Design

**Status:** proposed for implementation

**Task:** `TASK-26944`

## Goal

Remove the Ruff 0.15.22 formatter debt in the `ruff-chat-citations` batch without
changing Python behavior, comment meaning, or any Python file outside the batch.
The cleanup stays within the ownership boundary established by `TASK-26000`.

## Scope

The only Python paths owned by this task are:

- `Tests/Chat/test_citation_service_factory.py`
- `Tests/Chat/test_citation_trace_builder.py`
- `tldw_chatbook/Chat/citation_trace_builder.py`

Task documentation and this design may also change. No other Python path is in
scope. If completion appears to require another Python path or a hand-written
behavior change, implementation stops and the task boundary is reconsidered.

## Current Baseline and Reconciliation

The isolated implementation branch starts from the fetched `origin/dev` revision
`c7096ea1513539b124aa040d6a040a91bc7702d9`. At that revision:

- all three assigned paths exist;
- Ruff 0.15.22 reports that all three would be reformatted;
- the two assigned test modules pass together: 57 tests passed, with one existing
  dependency-version warning; and
- the `TASK-26000` manifest assigns the paths together with no captured conflict
  basis at its evidence pin.

Before formatting, implementation rechecks the branch against current
`origin/dev`. Any upstream deletion, rename, content change, or completed
formatting is recorded in the task notes and reconciled mechanically. A path is
never silently dropped or replaced by an unassigned path.

## Implementation

Use the repository's pinned Ruff 0.15.22 through Python 3.12.11 to format exactly
the three assigned paths in one invocation. Do not hand-edit the formatted Python
output and do not add a dependency, production helper, or abstraction for this
one-time cleanup.

The task is a formatter-only maintenance change. The expected Python diff is
layout-only. Any semantic or comment-invariant difference is a failure, not a
change to explain away.

## Semantic and Comment Invariants

Capture the following evidence from every assigned path before formatting and
compare it with the same capture after formatting:

1. Parse with Python 3.12.11 using `ast.parse(..., type_comments=True)`.
2. Recursively normalize only `TypeIgnore.lineno`.
3. Require identical `ast.dump(..., include_attributes=False)` output.
4. Preserve ordered comment-token text.
5. For inline `# noqa`, `# type: ignore`, and single-target Ruff directives,
   preserve the deepest AST-node path and significant-token position to which the
   comment is anchored.
6. For standalone file directives, preserve the adjacent statement paths between
   which each directive appears.
7. For every `# fmt: off` / `# fmt: on` pair, preserve the ordered AST-node
   interval enclosed by the pair.

The capture can use an ephemeral script outside the repository. Only its result,
commands, and relevant summary belong in the task notes. If an invariant differs,
restore the affected Python path to the pre-format state and investigate before
continuing.

## Verification

After formatting, verify all of the following against the exact owned paths:

- the AST and comment evidence is equal before and after;
- `ruff check` passes;
- `ruff format --check` passes;
- `pytest Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py`
  passes because these are the direct tests for the production helper and the
  batch's recorded focused-test surface;
- `pytest Tests/CI/test_backlog_task_id_uniqueness.py` passes;
- `git diff --check` passes;
- the changed-Python set is exactly the assigned set, with no unassigned Python
  path; and
- review of the Python diff confirms Ruff-generated layout changes only.

The task does not require the repository-wide test suite: its only Python changes
are formatting in one production helper and its direct tests, and the focused
tests exercise that behavior. Any failure is diagnosed before additional changes
are made.

## Closeout

Once verification passes:

- check every acceptance criterion in `TASK-26944`;
- add concise Implementation Notes containing the reconciliation result, focused
  test rationale, exact commands and results, invariant result, and files changed;
- run a final self-review of the scoped diff; and
- set `TASK-26944` to `Done` through the Backlog CLI, then verify that the
  canonical task file and task-ID uniqueness remain intact.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this task mechanically applies the formatter-only contract and ownership
boundary already established by `TASK-26000`. It introduces no storage, runtime,
security, service, dependency, or long-lived UX decision.
