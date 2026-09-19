---
id: TASK-32838
title: Resolve Audit destinations from the current catalog before rendering
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:54'
updated_date: '2026-09-19 21:04'
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
- [x] #4 The integrated dev startup selector guard passes without increasing its limit, while affected wide-modal geometry and appearance remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the approved catalog-freshness integration and all merged navigation/profile guards. 2. Close out current-head CI: reproduce the inherited 277/274 bare-type selector breach from PR2742, scope its three wide-tier Vertical targets with exact container classes, update the wide-tier registry and rebuild generated CSS. Preserve width/cap tokens, behavior and visual geometry; compare targeted and native before/after evidence. 3. Address accumulated Qodo findings, verify the exact resulting head, and merge after current-head CI/review passes under the existing visual approval. ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md. Reason: mechanical selector indexing and CI integration repair preserving existing runtime, authority and visual contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit navigation resolves current tool definitions under the existing catalog lock while retaining row-failure, profile and control ownership guards. Owner approved integrated visuals/conflict choices at a4158a6a9a; Qodo found zero issues there. Final CI exposed three inherited PR2742 bare-Vertical wide-modal rules (277/274). Unique subject classes preserve their exact targets, dimensions and token values; registry and generated bundle updated, no budget increase. 203 closeout targeted cases and seven guards pass, including all14 catalog cases, selector/byte/style-fidelity guards, wide-modal geometry/coverage, picker behaviors and56 native argument cases; earlier168-case approval evidence is retained. Independent review found no blockers; static baselines and changed-range formatting pass. All12 modal before/after and8 approved Audit captures are pixel-identical, with healthy isolated lifecycle and unchanged defaults. Initial fixture body-selection error is recorded. ADR097/150/161 apply; no new ADR. Evidence and current source hashes: Docs/superpowers/qa/2026-09-18-mcp-audit-catalog-freshness/CLOSEOUT.md. Final-head CI/review and live dev check remain before authorized merge. The intermittent header layout race remains a separate documented follow-up.
<!-- SECTION:NOTES:END -->
