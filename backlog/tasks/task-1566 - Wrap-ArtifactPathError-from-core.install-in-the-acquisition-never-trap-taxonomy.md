---
id: TASK-1566
title: >-
  Wrap ArtifactPathError from core.install in the acquisition never-trap
  taxonomy
status: To Do
assignee: []
created_date: '2026-08-01 01:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final-review residual on TASK-595 (branch feat/managed-model-acquisition): _run_core_call in tldw_chatbook/Model_Artifacts/acquisition.py catches ArtifactIntegrityError/ArtifactConflictError/ArtifactStateError but not ArtifactPathError, which core.install documents raising — it escapes provision() raw, breaking the spec's rule that every failure surfaces typed with a retryable flag. Reproduced live by the re-reviewer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ArtifactPathError from core.install surfaces as TransferError(retryable=False) with the cause chained
- [ ] #2 Regression test monkeypatches core.install to raise ArtifactPathError and asserts the wrapper
<!-- AC:END -->
