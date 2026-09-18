---
id: TASK-32782
title: Keep Tool Profile removal outcomes truthful and visible
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 10:21'
updated_date: '2026-09-18 10:36'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users an accurate, visible outcome and current profile state after removal refuses or cannot determine whether it completed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Uncertain removal is never presented as a definite failure; refusals explain an appropriate recovery without automatic mutation retries.
- [x] #2 Every completed removal attempt refreshes current profile facts; retained or missing profiles have a visible outcome and keyboard continuation.
- [x] #3 Outcome delivery and text reflow preserve focus and never scroll a newer category, dialog or profile action; literal profile text stays literal.
- [x] #4 Targeted mounted and real-service boundary checks pass; native compact/wide dark/light refusal-and-retry journeys show the outcome and verify private-profile lifecycle.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce misleading uncertain-removal copy, stale listing after a late removal outcome, and offscreen outcome text using mounted production-CSS controls.
2. Keep removal receipts beside the affected profile action (or Import when the row disappears), preserve literal text and current focus, and avoid scrolling newer navigation. Reuse existing settings text/layout classes and focus rules.
3. Distinguish uncertain outcomes from definite refusals, provide recovery guidance, and refresh presentation after attempted removal without automatically retrying the mutation.
4. Verify mounted outcome, concurrency and navigation cases plus real service boundaries; inspect native compact/wide dark/light refusal-and-retry journeys. Review independently and save evidence to draft PR 2707.
ADR required: no
ADR path: backlog/decisions/107-portable-tool-use-packs.md and backlog/decisions/150-design-token-system-and-design-language.md
Reason: Bounded outcome-presentation and refresh repair under the existing removal authority and design language; no service, persistence or permission policy changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removal now reports uncertain outcomes without claiming definite failure, explains refusal recovery, and refreshes current profile facts after terminal attempts. Profile-local plain-text receipts stay beside their controls, fall back to Import when a row disappears, and reveal only while the originating/replacement control owns focus. Existing mutation, revision, reference, lease and tombstone rules remain unchanged; no automatic retry.

167 targeted cases pass: 18 new mounted outcomes, 31 focus/loading, 62 prior workflows, 26 governance, 5 CSS synchronization and 25 real-service boundaries. Four native compact/wide dark/light journeys use an actual runtime lease to force refusal, verify unchanged policy bytes, release the lease and explicitly confirm a retry. All twelve captures rendered and inspected; exact-source, normal shutdown, healthy private DBs and unchanged default-profile checks pass.

Ruff introduces no diagnostics (Settings 114 to 114; panel 1 to 0); scoped formatting, backlog/diff and diagnostic inventory guards pass. Independent review found no introduced blocker, including an overlapping-render probe. One follow-up fixture initially compared an ongoing scroll animation; waiting for the animation repaired the comparison without changing production code. Existing text-reflow/focus lessons apply; no new general lesson added.

Evidence and gallery: Docs/superpowers/qa/2026-09-18-tool-profile-removal/README.md. Updated the Tool Profiles review ledger. Existing ADR-107 and ADR-150 apply: backlog/decisions/107-portable-tool-use-packs.md and backlog/decisions/150-design-token-system-and-design-language.md. No new ADR required. Management handoffs and concurrent-workflow review remain open.
<!-- SECTION:NOTES:END -->
