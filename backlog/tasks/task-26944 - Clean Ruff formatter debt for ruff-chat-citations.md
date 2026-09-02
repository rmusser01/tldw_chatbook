---
id: TASK-26944
title: Clean Ruff formatter debt for ruff-chat-citations
status: Done
assignee:
  - '@codex'
created_date: '2026-08-31 18:31'
updated_date: '2026-09-02 21:06'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/specs/2026-09-02-task-26944-ruff-chat-citations-design.md
  - Docs/superpowers/plans/2026-09-02-task-26944-ruff-chat-citations.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chat-citations` Ruff formatter batch at the owner boundary recorded as: Chat citation construction and trace helpers with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chat"]`.
<!-- SECTION:DESCRIPTION:END -->

<!-- TASK-26000-BATCH: ruff-chat-citations -->
<!-- TASK-26000-PATHS-SHA256: 6149803c4606eb131c95d8713504d1213a98aac1b4fc15f4fb5aea4fb9a73129 -->
<!-- TASK-26000-FINAL: false -->

## Assigned Paths

```json
[
  "Tests/Chat/test_citation_service_factory.py",
  "Tests/Chat/test_citation_trace_builder.py",
  "tldw_chatbook/Chat/citation_trace_builder.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] #2 Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] #3 Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] #4 Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] #5 Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] #6 Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] #7 `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] #8 The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: this is a formatter-only application of the existing TASK-26000 cleanup
contract and introduces no architectural boundary or durable policy.

1. Fetch and rebase onto current `origin/dev`, reconcile every assigned path, and
   capture the pre-format AST/comment evidence with the pinned toolchain.
2. Run Ruff 0.15.22 on only the reconciled owned paths and require identical
   semantic/comment evidence.
3. Run the scoped Ruff, focused-test, governance, and diff checks; self-review the
   layout-only change.
4. Record exact evidence, check all acceptance criteria, set the task to `Done`,
   and commit the closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Final base and lineage: fetched and rebased onto `origin/dev`
  `985265b91f23540648f332316477112bee827127`; the rebased implementation head
  before this closeout record was `de3c269fffa07b22133e38f87c0cc6bdaf1239b0`,
  and `git merge-base --is-ancestor origin/dev HEAD` exited 0. The final rebase
  retained the unrelated upstream Anthropic subscription, media-UX,
  single-page-pager, TASK-25900 planning, console trace-compaction, and semantic
  trace-ledger closeout, Library review-set phase-one, and MCP live-wiring
  evidence commits; none touched the assigned paths. All three
  original assigned paths still existed at their original names;
  their blobs at
  TASK-26000 pin `e555df102c950c29beed5e7119f433d35eee1f3c` equaled the blobs at the
  formatting parent, proving no rename, deletion, or content drift before
  formatting. Each path reproduced as a Ruff failure (`3 files would be
  reformatted`).
- Toolchain and formatter: Python 3.12.11 and Ruff 0.15.22. Ruff formatted exactly
  the three manifest paths in commit
  `de3c269fffa07b22133e38f87c0cc6bdaf1239b0` (`style(chat): format citation
  helpers`, parent `eb930bae6473c003702dd03618ba49cb8dbede81`). Replaying Ruff 0.15.22
  against those parent blobs produced byte-for-byte matches for all three
  committed blobs; the rebased blobs also equal the accepted pre-rebase
  formatting commit.
- Semantic/comment guard: `xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /tmp/task26944_format_guard.py compare /tmp/task26944_before.json < /tmp/task26944_paths.txt`
  reported `structural evidence matches for 3 paths`. The comparison preserved
  the normalized type-comment AST, ordered comments, inline and standalone
  directive anchors, and format-range node intervals; source SHA-256 alone was
  excluded from equality.
- Focused verification: the two directly assigned citation test modules exercise
  the production citation trace helper and are a proportionate narrowing from
  the recorded `Tests/Chat` surface for a formatter-only change.
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_citation_service_factory.py Tests/Chat/test_citation_trace_builder.py -q`
  passed 57 tests. The non-blocking output comprised the existing
  `RequestsDependencyWarning` and pytest temporary-directory cleanup warnings.
- Ruff and governance verification:
  `xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check < /tmp/task26944_paths.txt`
  reported `All checks passed!`;
  `xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check < /tmp/task26944_paths.txt`
  reported `3 files already formatted`;
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -q`
  passed 3 tests; `git diff --check` and `git diff --cached --check` exited 0.
  Comparing `/tmp/task26944_paths.txt` with both the formatting commit's changed
  paths and `git diff --name-only origin/dev..HEAD -- '*.py'` produced empty
  diffs, proving exactly the three assigned Python paths and no unassigned Python
  path changed. Fetch also emitted the known non-blocking Git `gc.log` and
  unreachable-loose-object housekeeping warning.
- Added files: `Docs/superpowers/specs/2026-09-02-task-26944-ruff-chat-citations-design.md`
  and `Docs/superpowers/plans/2026-09-02-task-26944-ruff-chat-citations.md`.
- Modified files: `Tests/Chat/test_citation_service_factory.py`,
  `Tests/Chat/test_citation_trace_builder.py`,
  `tldw_chatbook/Chat/citation_trace_builder.py`, and
  `backlog/tasks/task-26944 - Clean Ruff formatter debt for ruff-chat-citations.md`.
- ADR required: no. ADR path: N/A. Reason: this applies the existing formatter-only
  contract without changing architecture. No new lesson was required because no
  new generalizable incident arose; the known Backlog CLI filename normalization
  behavior is already covered by `backlog/docs/lessons-backlog-hygiene.md`.
<!-- SECTION:NOTES:END -->
