---
id: TASK-35614
title: Keep Console return input responsive during reconciliation
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 20:10'
updated_date: '2026-10-19 20:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate the saved Windows return outlier and remove verified input queueing and redundant action styling without weakening storage or attachment safety.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard input remains responsive during held reconciliation while runtime attachment stays gated; retries, stale visits and retirement remain correct.
- [x] #2 Send and Stop state transitions preserve final styles and behavior with batched class updates.
- [x] #3 Targeted tests and serial native Windows before/after profiles record raw latency, exits and guard outcomes.
- [x] #4 Review, documentation and PR 3166 update complete.
- [x] #5 Hook display refresh runs once after completed resume instead of competing twice with attachment; Send hook checks remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: routine scheduling correction within existing finite view-worker custody and native retirement under ADR-126/230; no authority, storage, or cache contract changes.
1. Reproduce screen message-pump queueing with held reconciliation and inspect saved native samples.
2. Dispatch reconciliation through existing managed view-worker ownership, preserving all visit gates and retirement; batch same-button classes using Textual.
3. Run targeted lifecycle and behavior checks, then serial native ABBA profiles pinned to 191fb7d7a3368062d8c4adea2f053ba4588d4f45.
4. Review changes, document evidence and limitations, update PR, clean exact owned fixtures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Published the reviewed fixes and complete evidence to PR #3166 (implementation commit 5f9e513de6). Reconciliation launches through a deduplicated finite view worker instead of blocking the screen message pump; exact native retirement and all visit/storage/Send checks remain. Batched Send/Stop classes reduce nine style updates to two. Removed the duplicate early resume-triggered hook display refresh; fresh Send authorization and independent refresh paths remain.
Final native pair: input median 721.84 to 177.79 ms (75.4% faster), worst 1532.84 to 200.64 ms. Attachment remains slower, 727.33 to 976.42 ms; reported explicitly as remaining work. All 36 return journeys retained drafts and attached, six exits zero, zero network attempts/profile refusals. Earlier ABBA results and failures remain in RETURN_RESULTS.md and return-verification.json.
Targeted keyboard/lifecycle, original config/workspace/hook native-read retirement, composer, unsafe native Windows ACL and timer-inventory checks passed after documented fixture corrections. Timer pins retained through the exact worker retry edge. New diagnostic/regression lint/format checks pass; pre-existing product findings unchanged. Independent reviews found no blockers; no full suite. Updated Console architecture and testing-evidence lesson. ADR required: no; ADR-126/230 boundaries preserved. Removed all 17 exact owned temporary fixture roots. Auto-merge remains disabled for fresh review.
<!-- SECTION:NOTES:END -->
