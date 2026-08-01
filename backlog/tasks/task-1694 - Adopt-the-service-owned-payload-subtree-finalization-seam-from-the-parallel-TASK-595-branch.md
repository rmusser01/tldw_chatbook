---
id: TASK-1694
title: >-
  Adopt the service-owned payload-subtree finalization seam from the parallel
  TASK-595 branch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 07:33'
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
Ported the download-stage API (_ManagedDownloadStage, _download_stage_for,
_finalize_download_stage, _discard_download_stage, marker/containment/
layout/state validation, retirement) from codexclone/task-595-v2's
service.py verbatim (+691 lines here; all helper symbols it depends on
already existed on this branch with matching signatures). Ported its 31
tests unmodified in intent into test_service.py (+540 lines).

Retargeted ArtifactAcquisitionService._install_artifact at
core._finalize_download_stage instead of core.install(...,
consume_source=True). provision()'s per-artifact loop now opens one
marked stage per artifact before fetch starts and threads stage.payload
through _fetch_artifact/_preverify_artifact (unchanged signatures -- they
stay stage-agnostic) and the stage object itself into _install_artifact
(new signature: (descriptor, stage)). _finalize_download_stage renames
stage.payload directly into the immutable destination -- no second
staging copy -- and retires the whole stage operation on success, so
_install_artifact needs no cleanup step of its own. install() itself is
unchanged for local-import callers.

Sidecar decision: deleted _FETCH_SIDECAR_SUFFIX/the old sibling-FILE
_fetch_sidecar_path -- the stage's own state/ subtree supersedes it
structurally (_validate_download_stage_layout's exact-entries check means
a sidecar can only live inside state/, never beside payload/ inside the
operation dir). _fetch_sidecar_path now returns
staging_dir.parent/"state"/"fetch-state.json". Left the OLD
_MANAGED_FETCH_SIDECAR_SUFFIX/_gc_managed_staging GC machinery in
service.py untouched (reconciliation doc item 4, separate follow-up).

Updated every test file whose fixtures assumed the old
staging/managed/<id>/<rev>/<variant> + sibling-sidecar-file layout
(test_provision_install.py, test_provision_fetch.py, test_preflight.py,
test_provision_crash_recovery.py, test_credentials_and_boundaries.py) to
construct/locate the new download-stage layout instead, preserving every
semantic assertion (retryable flags, byte content, resumability,
ordering). Added the AC #3 regression test
(test_retryable_finalize_failure_leaves_staged_bytes_resumable_via_range).
Removed one now-meaningless drift-guard test in
test_reconcile_staging_gc.py that asserted acquisition._FETCH_SIDECAR_SUFFIX
== service._MANAGED_FETCH_SIDECAR_SUFFIX (the constant it checked is gone).

Full required suite: PYTHONPATH=<worktree> pytest Tests/Model_Artifacts/
Tests/STT/test_boundaries.py -q -> 409 passed (was 379; +31 ported service
tests, -1 removed drift guard). Full report with file-by-file detail at
.superpowers/task-1694-report.md.
<!-- SECTION:NOTES:END -->
