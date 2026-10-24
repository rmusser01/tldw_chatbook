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
- [ ] #1 Generate components discloses exactly three sequential reference-image requests and configured destination/model, using one identical prepared reference; no provider fallback, automatic paid retry or backend-native mask requirement is introduced.
- [ ] #2 After consent and before its first request, a full batch atomically replaces all disposable frame/component state with a rest-only/zero-component snapshot; cancelled consent or failed initial validation preserves the old draft and sends no request.
- [ ] #3 A failed or uncertain fresh-batch component retains only successes from that batch and cannot use old slots to satisfy Apply/sample/publication; these require all three current components and six roles, while active DB artwork stays unchanged.
- [ ] #4 Validated returned edits contribute pixels only through saved regions; malformed/differently sized outputs fail without silent crop/stretch, candidate-state damage or publication.
- [ ] #5 Eyelid retry updates all three blink variants and small/open-mouth retry updates its two eye variants; unaffected bytes remain exact, failed retry preserves its prior valid snapshot/readiness, and reopen uses canonical pixel identity without double feathering.
- [ ] #6 Region/source/profile/editor/config changes cancel remaining requests and reject late snapshots; successful manual Import/Clear changes mode atomically, while cancelled/invalid imports preserve frames and lineage.
- [ ] #7 An in-app Review pairs action presents a labeled 2x3 overview and common unresampled eye/mouth detail crops from original normalized frame PNGs, with keyboard pan, Static/Reduce motion support and no provider/audio/save/Apply side effects.
- [ ] #8 The pair viewer bounds its raster to 1,048,576 pixels, sequentially decodes/closes original frames under shared budgets and retires stale/closed/resized work; a raw one-pixel crop witness survives even when absent from a fitted thumbnail.
- [ ] #9 Mounted request/UI tests cover disclosure, old-complete-to-failed-fresh batches, each retry group, snapshot failure/late-completion races, manual transitions, actual pair review, reopen and explicit Apply; TTS ownership and playback stay unchanged.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
