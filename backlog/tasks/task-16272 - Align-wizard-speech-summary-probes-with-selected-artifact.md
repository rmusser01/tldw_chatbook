---
id: TASK-16272
title: Align wizard speech summary probes with selected artifact
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 20:56'
updated_date: '2026-08-14 21:01'
labels:
  - testing
  - speech
  - wizard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Summary-step installation tests aligned with model-and-precision-aware Parakeet artifact lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Summary tests patch the live managed Parakeet lookup and verify selected identity.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the deleted-helper failures and confirm the current lookup contract.
2. Patch the live helper and assert the recommended model/precision arguments without changing production behavior.
3. Run focused, wizard, checkpoint, and static verification.

ADR required: no
ADR path: N/A
Reason: test-only reconciliation with the existing artifact lookup contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated both Summary-step tests to patch the live `active_managed_parakeet_dir` helper. The existing-root case now proves the recommended v2 model and INT8 precision are forwarded; the absent-root case still proves zero lookup and zero directory creation.
- Preserved RED evidence: both isolated nodes failed while patching the deleted v2-only symbol. Reverting the symbol reproduces those failures.
- Verified the four affected nodes (4 passed), the owning modules (103 passed), and the original 25-file checkpoint (524 passed). `git diff --check` passes. Ruff reports the same three pre-existing F401 diagnostics on `HEAD` and the changed wizard file; format drift is likewise unchanged.
- ADR required: no. This is test-only reconciliation with the selected-artifact lookup contract.
<!-- SECTION:NOTES:END -->
