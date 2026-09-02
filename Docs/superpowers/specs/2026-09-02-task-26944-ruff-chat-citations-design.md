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

Before formatting, implementation fetches `origin/dev`, records the fetched SHA,
rebases the task branch onto it, and then reproduces Ruff status at the rebased
head. Each original assigned path is compared with the `TASK-26000` evidence pin
and traced through Git history so that unchanged paths, content changes, deletions,
renames, and already-clean paths are explicit.

That reconciliation produces an effective owned-path set. An unambiguous rename
may replace its original path after the task record is amended with the lineage; a
deleted path is recorded and has no formatter input; and an already-clean path
remains owned and is still a valid Ruff input but need not produce a diff. An
ambiguous rename or copy, an unassigned destination already owned elsewhere, or
any other ownership ambiguity fails closed for user review. A path is never
silently dropped and an unassigned path is never absorbed.

## Implementation

Use the repository's pinned Ruff 0.15.22 through Python 3.12.11 to format exactly
the present files in the reconciled owned-path set in one invocation. The Ruff
input list must equal that set. Do not hand-edit the formatted Python output and do
not add a dependency, production helper, or abstraction for this one-time cleanup.

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
5. Express every AST-node path as a stable route of named AST fields and list
   indexes from the module root. For each inline `# noqa`, `# type: ignore`, and
   single-target Ruff directive, select the deepest node whose source span covers
   the comment's physical line and record that route. Also record the comment's
   significant-token ordinal within its containing logical statement as the
   position tie-breaker. Require both values to remain equal.
6. For standalone file directives, preserve the adjacent statement paths between
   which each directive appears.
7. For every `# fmt: off` / `# fmt: on` pair, preserve the ordered AST-node
   interval enclosed by the pair.

The capture can use an ephemeral script outside the repository. Only its result,
commands, and relevant summary belong in the task notes. Tokenization or parse
errors, non-unique or missing anchors, and unmatched, nested, or otherwise invalid
`fmt` ranges fail closed. If an invariant differs, restore the affected Python path
to the pre-format state and investigate before continuing.

## Verification

After formatting, verify all of the following against the exact owned paths:

- the AST and comment evidence is equal before and after;
- `ruff check` passes;
- `ruff format --check` passes;
- `pytest Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py`
  passes because these are the two directly assigned test modules and exercise the
  production helper; this is a proportionate narrowing from the manifest's broader
  recorded `Tests/Chat` surface for a formatter-only change;
- `pytest Tests/CI/test_backlog_task_id_uniqueness.py` passes;
- `git diff --check` passes;
- every formatter input equals a present path in the reconciled owned set, and the
  changed-Python set is contained within that set; if reconciliation confirms all
  three original paths still exist and fail at those paths, the changed-Python set
  must equal the original three-path set; and
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
