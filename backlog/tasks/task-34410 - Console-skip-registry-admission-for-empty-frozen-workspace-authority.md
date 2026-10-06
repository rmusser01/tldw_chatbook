---
id: TASK-34410
title: 'Console: skip registry admission for empty frozen workspace authority'
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-05 20:00'
updated_date: '2026-10-05 20:09'
labels:
  - console
  - resource-ownership
  - bugfix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3024'
documentation:
  - Docs/QA/task-33620.9/README.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An agent send with no admitted workspace folders still constructs the process-wide workspace registry solely to return an empty tracked-root set. Avoid this unnecessary database admission without altering authority, cache lifetime, or nonempty live-binding validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty captured workspace authority returns no roots without creating a default registry database or reopening a provided registry handle, including empty iterators.
- [x] #2 Nonempty authority still validates live binding identity, locator, existence and access without admitting later or retargeted folders.
- [x] #3 Original agent/direct shutdown controls pass a strict physical resource census with no warning suppression or shared-cache teardown, and raw failed receipts remain recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve current-head agent/direct RED with read-only registry allocation stack and strict physical census. 2. Add real-SQLite empty tuple/iterator regressions for default construction and supplied-handle reopening; verify they fail before production changes. 3. Materialize only the supplied authority iterable inside the existing fail-closed helper and return before any registry admission when empty; retain every nonempty live binding/root/locator check. 4. Run the original direct/agent shutdown controls, affected preparation/recovery controls and nonempty frozen-authority regression with strict physical census, static/artifact guards and independent review. Record all prior failures and separate broader/native/Windows/participant/scale/Qodo gates; no full sweep or global cache teardown. ADR required: no. ADR path: N/A; existing ADR069/085/120/198. Reason: same empty result and frozen-authority maximum, avoiding an unnecessary read/constructor only; no new storage/runtime/service/security or global lifetime/GC policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Avoided unnecessary registry admission with a three-line empty captured-authority guard inside the existing fail-closed helper. Real-SQLite tuple/iterator RED: four failures; original direct/agent allocation RED: two passing bodies but strict exit 1 with three workspace descriptors. GREEN: 22 affected controls in 7.77s, no warnings, strict exit 0 and zero DB files at every teardown. Nonempty identity/root/locator and no-retarget controls unchanged. Independent scoped review has no actionable finding; all eleven artifact guards pass. New test lint/format clean; eight inherited source Ruff findings and normalized formatting debt unchanged. QA receipt and incident lesson updated. ADR required: no; existing ADR069/085/120/198 apply, with no authority/cache/global lifetime policy change. Status remains In Progress: broader affected resource, emergency warning, native/Windows/participant/scale and current-head Qodo review gates are not waived.
<!-- SECTION:NOTES:END -->
