---
id: TASK-36554
title: Generate three portrait components and recompose dependent pose groups
status: Done
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
assignee:
- '@codex'
updated_date: 2026-10-24 17:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace independently generated full-frame poses with reviewed component generation so lighting and exact eye/mouth pairs survive batch generation and retries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generate components discloses exactly three sequential reference-image requests and configured destination/model, using one identical prepared reference; no provider fallback, automatic paid retry or backend-native mask requirement is introduced.
- [x] #2 After consent and before its first request, a full batch atomically replaces all disposable frame/component state with a rest-only/zero-component snapshot; cancelled consent or failed initial validation preserves the old draft and sends no request.
- [x] #3 A failed or uncertain fresh-batch component retains only successes from that batch and cannot use old slots to satisfy Apply/sample/publication; these require all three current components and six roles, while active DB artwork stays unchanged.
- [x] #4 Validated returned edits contribute pixels only through saved regions; malformed/differently sized outputs fail without silent crop/stretch, candidate-state damage or publication.
- [x] #5 Eyelid retry updates all three blink variants and small/open-mouth retry updates its two eye variants; unaffected bytes remain exact, failed retry preserves its prior valid snapshot/readiness, and reopen uses canonical pixel identity without double feathering.
- [x] #6 Region/source/profile/editor/config changes cancel remaining requests and reject late snapshots; successful manual Import/Clear changes mode atomically, while cancelled/invalid imports preserve frames and lineage.
- [x] #7 An in-app Review pairs action presents a labeled 2x3 overview and common unresampled eye/mouth detail crops from original normalized frame PNGs, with keyboard pan, Static/Reduce motion support and no provider/audio/save/Apply side effects.
- [x] #8 The pair viewer bounds its raster to 1,048,576 pixels, sequentially decodes/closes original frames under shared budgets and retires stale/closed/resized work; a raw one-pixel crop witness survives even when absent from a fitted thumbnail.
- [x] #9 Mounted request/UI tests cover disclosure, old-complete-to-failed-fresh batches, each retry group, snapshot failure/late-completion races, manual transitions, actual pair review, reopen and explicit Apply; TTS ownership and playback stay unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Implement the approved three-component generation/retry and bounded native-detail review contracts without changing provider or speech boundaries.

1. Read Unit 4, task, current generation/request choke point and mounted generation tests.
2. Write red tests for exactly three shared-reference requests, rest-only reset before dispatch, failed/successful group retry, unchanged byte preservation and source/settings/config fences.
3. Replace the old six-request body with the component workflow using immutable candidates, existing consent/worker/cancel/source ownership and mode-aware readiness; keep ordinary six-frame imports.
4. Implement actual Review pairs modal with labeled matrix and original PNG eye/mouth crops, bounded sequential decoding, keyboard pan and explicit stale-owner retirement, without speech/provider calls.
5. Run affected generation, draft/UI/viewer/speech-sample and token/CSS checks, review actual viewer captures, update docs and self-review before committing and starting TASK-36555.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replaced six independent edits with exactly three shared-reference component requests and one-component retries using existing consent/registry/validation. Full batches reset to rest-only before dispatch; immutable candidates retain only current successes and reject late source/settings/config/snapshot work. Retries preserve unaffected encoded bytes, including restored alternate PNG encodings; manual import/clear drops lineage only on success, and Apply carries optional provenance. Implemented actual original-PNG pair viewer with bounded sequential native crops, one-pixel witness, keyboard/numeric pan and explicit deferred retirement under covering modals. Visual review fixed native-render fitting and zero-width pan fields; shared renderer now has an explicit no-resize native mode and closes only owned temporary cores. Speech/Console ownership unchanged. Final affected gate: 143 passed/one coalesced Console-label test failure; isolated test passed, painted-state barrier added, then full Console file/CSS sync 19 passed. Additional held snapshot/settings/config controls 3 passed. Ruff/format/diff and CSS/token guards pass; production wide/narrow screenshots and raw matrices retained. All task findings resolved in self-review; see Docs/superpowers/reviews/2026-10-24-task-36554-review.md and updated Architecture/user guide. ADR-243 applies; native-render incident lesson recorded. No full suite or live provider claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
