---
id: TASK-32838
title: Resolve Audit destinations from the current catalog before rendering
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:54'
updated_date: '2026-09-19 20:22'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Audit tool and permission drilldowns consistent with catalog replacements that finish while navigation is pending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Audit destinations render the current same-ID tool definition after a pending catalog refresh, including updated description and availability.
- [x] #2 A vanished catalog target clears detail and warns; profile changes continue to reject stale navigation.
- [x] #3 Navigation rendering and the existing catalog publication lock preserve coherent row and permission detail, verified by targeted races and private native journeys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase the existing PR2726 onto merged dev cccf0acdad, retaining both report histories and all PR2724 row-selection, profile and control-ownership guards.
2. Resolve both destination identities under the existing publication lock before selecting and rendering. Preserve post-selection profile validation and vanished-row warnings; verify old and new regressions against the combined implementation.
3. Refresh the native launcher to shared validation and supported terminal warm-up; verify private dark/light compact/wide journeys, focused tests, guards and independent review. Update the existing draft with exact conflict choices and fresh visuals for its own final owner approval.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine integration of existing catalog-publication and inspector contracts without new authority, persistence or service boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased PR2726 onto merged dev cccf0acdad, retaining both report histories, row-selection failures and post-selection profile validation. Both Audit destinations re-resolve current definitions under the existing publication lock. Fourteen catalog regressions include two profile-switch negative controls; all 168 targeted cases and seven preflight guards pass. New files and changed ranges are formatted, diagnostic baselines unchanged, independent review found no blockers. Shared validated native bootstrap and two private dark/light compact/wide journeys verify exact identity/current metadata with no execution, network or permission changes; clean lifecycle and source hashes pass. Eight replay captures and exact conflict choices are ready for separate owner approval. The first run exposed an intermittent painted header mismatch, absent on unchanged-source replay; retained in HEADER-FOLLOWUP.md for separate baseline reproduction and repair, not claimed fixed. ADR-150/161 apply; no new ADR. Both ledgers and the PR2724 actual-merge verification receipt are updated. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/README.md. Current-head CI/review and PR2726 visual approval remain merge gates.
<!-- SECTION:NOTES:END -->
