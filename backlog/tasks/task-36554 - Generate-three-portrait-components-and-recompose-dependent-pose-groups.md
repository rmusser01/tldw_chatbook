---
id: TASK-36554
title: Generate three portrait components and recompose dependent pose groups
status: To Do
created_date: 2026-10-24 14:24
dependencies:
- TASK-36551
- TASK-36552
- TASK-36553
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace independently generated full-frame poses with reviewed component generation so lighting and exact eye/mouth pairs survive batch generation and retries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generate components discloses exactly three sequential reference-image requests and the configured destination/model; all use the same prepared base, with no provider fallback, automatic paid retry or backend mask capability requirement.
- [ ] #2 Validated returned edits contribute pixels only through the saved regions, yielding the exact six-role matrix; malformed or differently sized outputs fail without silently cropping/stretching or damaging the active portrait.
- [ ] #3 A failed or uncertain component retains other successful components in its source-bound candidate, blocks incomplete Apply and permits an explicit one-request retry; successful dependent frame groups stage atomically.
- [ ] #4 Regenerate eyelids updates all three blink variants, regenerate small/open mouth updates its open/closed-eye pair, and unaffected frames remain exact; reopening validated provenance supports the same retry semantics without repeated feathering.
- [ ] #5 Changing a region recipe/source/profile/editor/config while work is in flight cancels remaining submissions and rejects late results; successfully importing or explicitly clearing an individual pose drops component lineage but retains saved regions and normal six-frame review; cancelled/invalid imports preserve frames and lineage.
- [ ] #6 Mounted real-request-validation tests cover three-call disclosure, each retry group, partial failure, generation/save races, late uncancellable completions, manual-import transition, reopen and explicit Apply; TTS preview ownership and playback remain unchanged.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
