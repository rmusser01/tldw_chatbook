---
id: TASK-35521
title: Restore served Canvas terminal rendering and reconnect UAT evidence
status: In Progress
created_date: 2026-10-18 15:33
assignee:
- '@codex'
labels:
- canvas
- web
- tests
- uat
priority: medium
documentation:
- Docs/Canvas/UAT_2026-10-18.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct stale DOM-text observations in the existing served Canvas real-browser journeys while retaining their actual terminal input, painted output, exact revision selection and reconnect/isolation coverage. The production renderer deliberately uses Canvas/WebGL; a DOM-text-only oracle cannot qualify visible terminal output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Direct TLS and trusted-proxy journeys observe fresh semantically correct terminal output plus actual visible text-layer painting after real keyboard reopen and post-control-loss ping, without forcing a DOM renderer or modifying the runtime.
- [ ] #2 The shared observer cannot accept stale earlier output, missing fresh output, cursor/selection-only repaint or a frozen/blank terminal text layer; deterministic negative controls demonstrate these refusals.
- [ ] #3 All original exact-root, v1/Pinned, child isolation, revocation/replacement, copied/stale URL refusal and same-origin traffic assertions remain covered and both complete two-child reconnect cases pass.
- [ ] #4 The same production-renderer-compatible evidence supports the actual Console initial Composer, typed prompts and final reply markers in original served UAT cases, with no direct action substitution, deadline increase, admission bypass or sync activation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Reason: test evidence correction for the existing Canvas/WebGL renderer contract under TASK-33130 and ADR-121; no product protocol or capability change. 1. Preserve the real input/binary screen write/visible paint/root selection diagnosis for both TLS paths. 2. Write deterministic missing/stale-output and frozen-text-layer RED controls before implementing the shared test observer. 3. Reuse the existing installed terminal decoder where appropriate and keep observer state bounded and per page/connection, retaining fresh receipt and independent text paint evidence rather than DOM text. 4. Integrate only the affected live terminal assertions and run both complete TLS paths and actual Console cases after startup repair. Parent owns all Backlog/docs and final 41-case UAT; no full suite, renderer change, external server work, dependency installation or publication.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
