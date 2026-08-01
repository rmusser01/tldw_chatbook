---
id: TASK-1566
title: >-
  Wrap ArtifactPathError from core.install in the acquisition never-trap
  taxonomy
status: Done
assignee: []
created_date: '2026-08-01 01:57'
updated_date: '2026-08-01 03:00'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final-review residual on TASK-595 (branch feat/managed-model-acquisition): _run_core_call in tldw_chatbook/Model_Artifacts/acquisition.py catches ArtifactIntegrityError/ArtifactConflictError/ArtifactStateError but not ArtifactPathError, which core.install documents raising — it escapes provision() raw, breaking the spec's rule that every failure surfaces typed with a retryable flag. Reproduced live by the re-reviewer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ArtifactPathError from core.install surfaces as TransferError(retryable=False) with the cause chained
- [x] #2 Regression test monkeypatches core.install to raise ArtifactPathError and asserts the wrapper
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `ArtifactPathError` to `_run_core_call`'s except tuple alongside `ArtifactIntegrityError`/`ArtifactConflictError` (all three are non-retryable siblings under `ArtifactError`, not subclasses of each other, so ordering doesn't matter) in `tldw_chatbook/Model_Artifacts/acquisition.py`; imported `ArtifactPathError` from `.service`. Test: `test_install_artifact_wraps_core_path_error_as_non_retryable` in `Tests/Model_Artifacts/test_provision_install.py` monkeypatches `core.install` to raise `ArtifactPathError` and asserts `TransferError(retryable=False)` with the cause chained — confirmed it fails against pre-fix code (raw `ArtifactPathError` escapes `_install_artifact`).

Landed as part of a broader PR #1157 review fix-wave alongside seven other findings (P1 security/race fixes in fetch.py and service.py, several P2s in acquisition.py). See that PR/commit for the full set.
<!-- SECTION:NOTES:END -->
