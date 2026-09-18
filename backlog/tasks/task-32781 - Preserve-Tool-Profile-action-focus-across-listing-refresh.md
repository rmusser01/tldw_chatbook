---
id: TASK-32781
title: Preserve Tool Profile action focus across listing refresh
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 10:00'
updated_date: '2026-09-18 10:17'
labels:
  - settings
  - ui
  - tool-packs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep keyboard users on the same profile action through refresh and modal return, with a visible continuation after an action disappears.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unchanged listings preserve focused controls; changed or reordered listings retain the same profile/action, with a safe visible fallback if it disappears or becomes disabled.
- [x] #2 Refresh cannot reclaim focus from newer user navigation or another dialog, and overlapping listing renders settle on the newest listing without cancelling a teardown.
- [x] #3 Queued controls retain their existing profile/revision authority and cancelled import/removal actions leave data unchanged.
- [x] #4 Targeted focus and workflow regressions pass; native compact/wide dark/light import, refresh and removal journeys verify focus visibility, actual data outcomes and normal private-profile shutdown.
- [x] #5 Compact Tool Profile action rows paint every focused button label in full, including Remove, using existing layout and spacing tokens.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce focus loss through unchanged/changed/reordered listings, modal cancellation and overlapping renders with production CSS.
2. Skip identical listings; serialize changed-list recomposition, capture focus by profile/action, clear only the displaced focus, and restore only if no newer focus has arrived. Use a same-profile or Import continuation when needed.
3. Retain queued-action ownership tests, exercise newer focus and concurrency, and review independently.
4. Verify real native import/removal and focus journeys in four theme/viewport cells, record evidence and save to draft PR 2707.
ADR required: no
ADR path: backlog/decisions/107-portable-tool-use-packs.md and backlog/decisions/150-design-token-system-and-design-language.md
Reason: Repairs focus and refresh behavior within existing UI interaction and Tool Pack authority rules; no service, storage or permission changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved Tool Profile action focus through unchanged, reordered and overlapping listing refreshes while retaining original queued-control authority and newer navigation. Serialized panel renders, safe same-profile/Import fallback, and guarded reveal repair teardown focus loss. Compact action rows now use existing tokens in a two-column grid after full-paint tests exposed Remove clipping.

112 final-source targeted checks pass; 12 prior action/loading cases also passed before the compact-only CSS adjustment. Four native dark/light compact/wide journeys verify revise/cancel, real unbound import, reorder focus, real tombstone removal and normal private-profile shutdown. All 24 captures rendered and inspected. No introduced Ruff diagnostics; scoped formatting, CSS/token governance, backlog/diff and diagnostic inventory guards pass. Independent reviews found no introduced blocker.

Evidence: Docs/superpowers/qa/2026-09-18-tool-profile-focus/README.md. Updated Tool Profiles review ledger and testing-evidence lesson. Existing ADR-107 and ADR-150 apply (backlog/decisions/107-portable-tool-use-packs.md; backlog/decisions/150-design-token-system-and-design-language.md); no new ADR required. Remaining removal boundaries and MCP handoffs stay in the review ledger.
<!-- SECTION:NOTES:END -->
