---
id: TASK-35614
title: Keep Console return input responsive during reconciliation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-19 20:10'
updated_date: '2026-10-19 20:36'
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
- [ ] #4 Review, documentation and PR 3166 update complete.
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
Confirmed Textual after-refresh reconciliation awaited inside the screen message pump. Dispatch now uses a deduplicated finite view worker under existing exact-worker retirement; all reconciliation/storage/Send gates remain intact. Batched Send/Stop classes reduce nine synchronous style refreshes to two. Deferred the duplicate resume-triggered hook display refresh until reconciliation completes; independent session/Send refreshes and fresh authorization stay unchanged.
Native serial ABBA followed by a final combined-code pair: final input median 721.84 to 177.79 ms (75.4% faster), worst 1532.84 to 200.64 ms. Attachment remains slower, 727.33 to 976.42 ms; no completed-reconciliation speedup is claimed. All 36 return journeys retained drafts/attached, six zero exits, zero network attempts/profile refusals. See RETURN_RESULTS.md and raw receipts.
Targeted input, lifecycle/retry/visit, original config/workspace/hook native-read retirement, Send/Stop, and native unsafe-ACL checks passed after fixture corrections. Timer inventory retains the same worker-backed retry root and portable paths; four checks passed. Initial failures and rechecks retained in return-verification.json. New code lint/format passes; pre-existing product lint/format debt unchanged. Independent review found no blockers. No full sweep. ADR required: no; existing ADR-126/230 custody and freshness boundaries preserved. Updated Console architecture and testing-evidence lesson. PR publication and exact fixture cleanup pending.
<!-- SECTION:NOTES:END -->
