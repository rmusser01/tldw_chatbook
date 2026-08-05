---
id: TASK-608
title: Remove stale compact model bar settings patch from chat UI tests
status: Done
assignee: []
created_date: '2026-07-24 18:51'
updated_date: '2026-07-24 19:10'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the chat approvals and resume UI module after CompactModelBar stopped importing get_cli_setting but its shared test fixture continued patching that retired module symbol.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fixture patches only configuration seams used by the mounted chat widgets
- [x] #2 All chat approvals and resume UI tests pass
- [x] #3 The merge-base failure and no-ADR decision are documented
- [x] #4 The first-run orientation and enhanced chat-window integration modules also pass without retired settings patches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact setup error on the feature branch and merge base.
2. Remove the stale compact_model_bar.get_cli_setting patch while retaining the provider/model seam and settings patches still used by other widgets.
3. Run the full chat approvals/resume module, Ruff, and diff checks.
4. Review and document the task before completion.

ADR required: no
ADR path: N/A
Reason: This is a test-only correction for a retired import and changes no runtime contract or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed stale monkeypatches for get_cli_setting from both compact_model_bar and enhanced_settings_sidebar across the approvals/resume, first-run orientation, and enhanced chat-window integration fixtures. Those modules no longer expose that symbol; the fixtures retain the provider/model seams and Chat_Window_Enhanced or chat-tab settings seams that the mounted widgets still use.

The feature branch and merge base both fail during fixture setup with AttributeError when monkeypatch tries to resolve the retired module attributes. Verification: all three affected modules pass (17 passed); the combined TASK-608/524/525 batch passes 79 tests; Ruff and diff checks pass. test_chat_first_run_orientation.py has unrelated pre-existing whole-file Ruff-format debt on the merge base, so no broad mechanical reformat was included. Independent review approved the correction and specifically confirmed the enhanced_settings_sidebar cleanup is in scope.

ADR required: no. This removes stale test-only patches and changes no production settings contract or architecture.
<!-- SECTION:NOTES:END -->
