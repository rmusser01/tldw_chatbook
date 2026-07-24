---
id: TASK-223
title: Deduplicate vision gate in Console controller
status: Done
assignee:
  - '@claude'
created_date: '2026-07-13 11:15'
updated_date: '2026-07-16 20:31'
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
- [x] #1 Single capability check in submit_draft; blocked copy still comes from vision_block_reason
- [x] #2 Existing controller vision tests pass against one documented patch seam
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
vision_block_reason gains optional is_capable predicate (additive kwarg, attachment_core internal seam remains the default for the screen-side caller); submit_draft injects controller_module.is_vision_capable — one check, one documented seam, blocked copy unchanged. Divergence tests pin both directions (controller-says-no blocks even when attachment_core-says-yes, and vice versa sends). Existing controller vision tests pass unedited.
<!-- SECTION:NOTES:END -->
