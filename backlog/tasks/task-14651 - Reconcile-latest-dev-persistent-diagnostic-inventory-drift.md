---
id: TASK-14651
title: Reconcile latest-dev persistent diagnostic inventory drift
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 21:05'
updated_date: '2026-08-10 08:53'
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
- Reviewed all 16 generated-versus-stored owner deltas from the recorded incident baseline and their introducing commits under ADR-029. The MCP delta removed a raw exception diagnostic; safe provider/backend/config-key/count/type metadata was accepted. The first final rebase onto `origin/dev` `60fa4859f` exposed one additional `rag_service.py` owner delta. The PR integration rebase onto `d2d303d69` then exposed 17 newer diagnostic additions: one fixed safe config warning and 16 unsafe traceback, exception-message, private-ID, media-ID, or user-entered trim-value shapes. Every addition was reviewed before the manifest was regenerated; the 16 unsafe shapes were converted to fixed operational text plus permitted exception types, counts, or screen-class metadata.
- Replaced private IDs, search queries, paths, filenames, style identifiers, exception messages, and implicit traceback capture in the reviewed diagnostic call shapes. Functional control flow and user-facing error reporting were unchanged. The related RAG degradation test still requires its operational warning while proving the private database filename is absent.
- Added a source-level architecture regression over the real production modules. It failed before each repair wave and now enforces the precise metadata fields for 57 reviewed call shapes. It rejects `logger.exception`, every `logger.opt(exception=...)` value except explicit `False`, stdlib `exc_info`/`stack_info` capture, and private values hidden in chained `logger.bind(...)` or direct keyword-format fields; it constructs no application or simplified substitute.
- Regenerated and hand-reviewed the manifest: 485 owner files, 1,167 TASK-492 calls, 6,962 TASK-494 calls, and 6 persistent-sink files. Persistent-sink topology is unchanged.
- Final PR-integration verification used only affected functions and modules: 86 architecture/privacy/debug-log tests passed; 8 HuggingFace tests passed with 60 deselected; 11 direct roleplay store/controller tests passed with 331 deselected; 4 chat-display-name config tests passed with 25 deselected; and 7 direct FFmpeg trim tests passed. The inventory checker is included in the architecture set. No test application, simplified application, or repository-wide suite was used.
- Ruff lint passed for every edited line, all 14 edited-range formatter checks passed, all 8 edited Python files compiled, and `git diff --check` passed. The only full-file Ruff findings are latest-dev baseline imports/names outside this PR's hunks: `QueryError` in `mcp_workbench.py` and duplicate/unused help-panel imports in `settings_screen.py`; the focused lint rerun ignored only those exact baseline rule codes in those two files.
- ADR decision: no new ADR. `backlog/decisions/029-local-private-data-boundary.md` directly governs the repair. Added the formatter-before-manifest and rebase-review-boundary incidents to `backlog/docs/lessons-testing-evidence.md`.
- Final merge integration rebased conflict-free onto `origin/dev` `37e634cbb`. The new Console-rail commit added no persistent logger calls, so no manifest entry required re-review. Fresh focused evidence passed: 99 combined architecture/privacy/debug-log/RAG-keyword tests, 8 HuggingFace tests, 11 roleplay store/controller tests, 4 chat-display-name config tests, and 7 FFmpeg trim-argument tests. The inventory checker remained green at 485/1,167/6,962/6; all 70 branch-changed Python hunks were formatter-clean, all 26 changed Python files compiled, and diff checks passed. Current-dev lint debt outside the PR hunks is limited to F401/E713/E731 in `chat_screen.py`, F821 in `mcp_workbench.py`, and F401/F811 in `settings_screen.py`; lint passed with only those exact base-owned codes excluded for those three files.
<!-- SECTION:NOTES:END -->
