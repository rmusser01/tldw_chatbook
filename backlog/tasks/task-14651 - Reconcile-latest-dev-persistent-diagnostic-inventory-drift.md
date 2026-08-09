---
id: TASK-14651
title: Reconcile latest-dev persistent diagnostic inventory drift
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 21:05'
updated_date: '2026-08-09 22:39'
labels:
  - testing
  - baseline
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2118 final verification reproduced the persistent-diagnostic architecture failure on exact origin/dev f6911b37b after removing TASK-2118's sole branch-owned LLM_API_Calls.py digest delta. The generated-versus-stored baseline differs across 16 unrelated owner entries while persistent sink topology remains unchanged. Review those current-dev diagnostic changes under ADR-029 and reconcile only the accepted baseline; TASK-2118 must not bless them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every generated-versus-stored owner delta on the recorded dev incident baseline is reviewed under ADR-029
- [x] #2 Diagnostics introduced since the stored baseline comply with ADR-029: unsafe private values and exception details are replaced by metadata-only diagnostics, while safe diagnostics are accepted without changing unrelated production behavior or sink topology
- [x] #3 The focused persistent-diagnostic architecture test passes after the reviewed refresh
- [x] #4 A focused regression test rejects the reviewed exception-detail and private-value logging shapes without constructing a test application
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: `backlog/decisions/029-local-private-data-boundary.md`
Reason: ADR-029 already defines the persistent-log privacy boundary and the permitted metadata.

1. Reproduce the generated-versus-stored inventory failure on current `origin/dev`, enumerate every owner delta, identify its introducing commit, and classify each changed diagnostic under ADR-029.
2. Add a focused source-level regression test that fails on the reviewed private-value and exception-detail log shapes without constructing a test application.
3. Replace only the unsafe diagnostic arguments with fixed operational messages and permitted metadata such as exception type or counts; preserve functional control flow and persistent-sink topology.
4. Regenerate the inventory, review every resulting manifest delta, and accept only the reconciled owner entries after proving topology is unchanged.
5. Run only the focused diagnostic architecture/privacy tests and static checks for edited Python files, then record exact evidence and close TASK-14651.
6. Rebase the completed branch onto the latest `origin/dev`, repeat the touched-scope verification, push, and open a pull request against `dev`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reviewed all 16 generated-versus-stored owner deltas from the recorded incident baseline and their introducing commits under ADR-029. The MCP delta removed a raw exception diagnostic; safe provider/backend/config-key/count/type metadata was accepted. The final rebase onto `origin/dev` `60fa4859f` exposed one additional, newer `rag_service.py` owner delta, which was reviewed before its manifest entry was accepted.
- Replaced private IDs, search queries, paths, filenames, style identifiers, exception messages, and implicit traceback capture in 39 reviewed diagnostic call shapes with fixed operational text plus permitted counts, backend names, or exception types. Functional control flow and user-facing error reporting were unchanged. The related RAG degradation test now requires its operational warning while proving the private database filename is absent.
- Added a source-level architecture regression over the real production modules. It failed before both repair waves and now enforces the precise metadata fields for 40 reviewed call shapes while rejecting `logger.exception` and `logger.opt(exception=True)`; it constructs no application or simplified substitute.
- Regenerated and hand-reviewed the manifest: 482 owner files, 1,158 TASK-492 calls, 6,949 TASK-494 calls, and 6 persistent-sink files. Persistent-sink topology is unchanged.
- Final post-rebase verification: 144 focused TASK-14651 architecture/RAG/affected-function tests passed; TASK-2118 remained green at 70 privacy tests, 8 HuggingFace tests (60 deselected), and 2 debug-log hygiene tests. All currently edited Python files pass full-file Ruff lint and compilation, every edited range passed Ruff format, the inventory checker passed, and `git diff --check` passed. The earlier repair wave also compiled all 16 edited Python files and had zero changed-line Ruff violations; the 28 full-file `chat_screen.py` findings were pre-existing and outside the edited ranges.
- ADR decision: no new ADR. `backlog/decisions/029-local-private-data-boundary.md` directly governs the repair. Added the formatter-before-manifest incident to `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
