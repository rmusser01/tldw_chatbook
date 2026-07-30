---
id: TASK-1455
title: >-
  Delete orphaned test files and stale test docs; correct Tests/README.md's dead-marker and parallelism claims
status: In Progress
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - cleanup
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-07-30 test-suite audit found mechanically-dead test infrastructure that misleads readers and tools: `Tests/UI/ingestion_test_helpers.py` (536 lines) imports the deleted `tldw_chatbook.Widgets.Media_Ingest.*` package and has zero importers; `Tests/Chatbooks/test_chatbook_ui_integration.py.skip` is a git-tracked disabled-by-rename file using `textual.testing.AppTest` (removed from Textual ~5 majors ago); `Tests/RAG/README.md` documents 7 test files that do not exist; `Tests/UI/README_TEST_SUITE.md` documents the deleted Media_Ingest product code and instructs coverage of an empty package; `Tests/TEST_RESULTS_SUMMARY.md` and `Tests/RAG_Search/test_results_summary.md` are undated pass/fail baselines contradicting current code. `Tests/README.md` documents a marker workflow (`optional_deps`) that selects zero tests and a `pytest -n auto` command that fails because pytest-xdist is not installed. Judgment-based deletions (skipped/vacuous tests) are explicitly out of scope — they are task-1464's decision table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [ ] The six dead files listed above are removed
- [ ] `Tests/README.md` no longer documents the unused `optional_deps` marker or presents `-n auto` as available by default
- [ ] `pytest --collect-only` count is unchanged vs baseline (none of the deleted files were collected)

## Implementation Plan

1. Re-verify each target on dev tip (importer grep for the helper; existence of the described files for the docs)
2. `git rm` the six files
3. Targeted `Tests/README.md` corrections (marker section, parallelism note)
4. Verify collect-only count unchanged

## Implementation Notes

All six targets re-verified on dev tip before deletion: the helper's only
references were itself and the README deleted alongside it; `Widgets/Media_Ingest`
no longer exists. README edits are surgical (three false claims) rather than a
rewrite — the fuller marker-strategy rework belongs to task-1457/task-1464.
Deleted: `Tests/UI/ingestion_test_helpers.py`, `Tests/Chatbooks/test_chatbook_ui_integration.py.skip`,
`Tests/RAG/README.md`, `Tests/UI/README_TEST_SUITE.md`, `Tests/TEST_RESULTS_SUMMARY.md`,
`Tests/RAG_Search/test_results_summary.md`. Modified: `Tests/README.md`.
