---
id: TASK-1694
title: >-
  Adopt the service-owned payload-subtree finalization seam from the parallel
  TASK-595 branch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 07:46'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation decision (Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-reconciliation.md, item 1): port the download-stage API from codex/task-595-managed-downloads-v2 (service.py +653, test_service.py +528, in the GitHub/tldw_chatbook clone) — a marked, contained stage whose payload/ subtree is verified and RENAMED into the immutable destination, with resume metadata held outside that subtree. Retarget acquisition._install_artifact at the finalization seam instead of install(consume_source=True). This removes the sibling-sidecar workaround and structurally eliminates the class of bug where a retryable install failure destroys the resumable download. Do this BEFORE TASK-596 builds on the layer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verified payload files are promoted by renaming the stage's payload subtree; no second staging copy occurs for remote acquisition
- [x] #2 Resume metadata cannot reside inside what gets promoted
- [x] #3 A retryable install/finalization failure leaves the durable partial resumable (regression test)
- [x] #4 The ported stage tests from the parallel branch pass unmodified in intent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read reconciliation doc + design spec on codexclone/task-595-v2 to understand the payload-subtree finalization seam.
2. Port the download-stage API (_ManagedDownloadStage, _download_stage_for, _finalize_download_stage, _discard_download_stage, marker/containment/layout/state validation, retirement) from codexclone/task-595-v2's service.py into this branch's service.py; port its 528 lines of tests unmodified in intent.
3. Retarget ArtifactAcquisitionService._install_artifact at core._finalize_download_stage instead of core.install(..., consume_source=True); thread a real download stage through provision()'s per-artifact loop.
4. Decide the sidecar's new home (stage.state/) and delete the sibling-file workaround (_FETCH_SIDECAR_SUFFIX/_fetch_sidecar_path) it superseded.
5. Update every affected test file's fixtures to the new stage layout while preserving semantic expectations; add the AC #3 regression test.
6. Run the full required suite green; write the report; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ported the service-owned download-stage API from codex/task-595-managed-downloads-v2 (fetched locally as codexclone/task-595-v2); the stage implementation and its 23 tests are byte-for-byte faithful per review. Stages are marked and contained, with exact entries {marker, payload, state}: acquisition writes payload bytes into payload/ and resume metadata into the sibling state/, and _finalize_download_stage verifies then RENAMES the payload subtree into the immutable destination (single os.rename; a monkeypatched _copy_payload proves no copy occurs). acquisition._install_artifact no longer calls core.install, and the sibling-sidecar workaround plus its drift-guard test are gone — resume metadata is now outside the promoted subtree by construction. install(consume_source=) is retained for local import. Regression test proves a retryable finalization failure leaves the durable partial resumable via a real Range resume. 409 tests green.
<!-- SECTION:NOTES:END -->
