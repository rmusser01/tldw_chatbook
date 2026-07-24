---
id: TASK-514
title: Remove retired embeddings selectors from CSS QA guard
status: Done
assignee: []
created_date: '2026-07-24 18:23'
updated_date: '2026-07-24 18:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the Textual highlight-selector QA contract with the removal of the unreachable legacy embeddings UI, while retaining checks for live source and generated-bundle selectors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The QA guard no longer requires selectors owned by the deleted legacy embeddings UI
- [x] #2 Live ListView, chatbooks, and config-search highlight selectors remain required
- [x] #3 The source scan still rejects the invalid double-hyphen Textual highlight state
- [x] #4 The focused QA guard and CSS integrity tests pass
- [x] #5 No production CSS or UI code changes are made
- [x] #6 Task documentation records the merge-base failure, retirement evidence, ADR decision, and verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact failure on both feature branch and merge base, then verify the missing selectors belonged to the retired legacy embeddings UI.
2. Remove only those retired selectors from the QA expected-selector tuple, keeping live highlight and invalid-state checks.
3. Run the focused QA guard and CSS build/integrity tests.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: N/A
Reason: This updates a stale test expectation after an already-recorded UI removal; it changes no CSS, application structure, runtime boundary, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Summary: Aligned the Textual highlight-selector QA contract with the intentional removal of the unreachable legacy embeddings UI.

Approach and base comparison:
- The focused test failed identically on merge base ba6b45cdf4dd548796e072f5933cdcf44c8c0344 and the feature branch because the generated bundle no longer contains #embeddings-model-list ModelListItem.-highlight or #embeddings-collection-list CollectionListItem.-highlight.
- Commit 551193f86 removed the unreachable legacy SearchWindow/Embeddings UI stack, including tldw_chatbook/Widgets/embeddings_list_items.py and tldw_chatbook/css/features/_embeddings.tcss, then rebuilt the generated bundle.
- Current production search found no ModelListItem or CollectionListItem definitions or uses.
- Removed exactly those two retired expectations. The live ListView, chatbooks, and config-search selectors remain required, and source/generated scans still reject .--highlight.

Verification:
- Tests/QA/test_textual_highlight_selectors.py plus Tests/UI/test_css_build_integrity.py: 7 passed.
- Ruff format check: file already formatted.
- Ruff check: all checks passed.
- git diff --check: clean for owned files.
- Self-review: the test diff is exactly two deleted tuple entries; no production CSS or UI file changed.

ADR required: no
ADR path: N/A
Reason: Test-only expectation maintenance follows an already-completed UI retirement and changes no application structure or runtime boundary.

Files modified:
- Tests/QA/test_textual_highlight_selectors.py
- backlog/tasks/task-514 - Remove-retired-embeddings-selectors-from-CSS-QA-guard.md
<!-- SECTION:NOTES:END -->
