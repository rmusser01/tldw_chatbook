---
id: TASK-223
title: Deduplicate vision gate in Console controller
status: To Do
assignee: []
created_date: '2026-07-13 11:15'
labels:
  - console
  - tech-debt
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleChatController.submit_draft re-implements the vision-capability check via its module-level is_vision_capable seam and then calls attachment_core.vision_block_reason, which performs the same check internally — a test-seam-only divergence where the two could theoretically disagree (they resolve the same function in production). Call vision_block_reason once and branch on its result, keeping the monkeypatch seam story coherent for tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single capability check in submit_draft; blocked copy still comes from vision_block_reason
- [ ] #2 Existing controller vision tests pass against one documented patch seam
<!-- AC:END -->
