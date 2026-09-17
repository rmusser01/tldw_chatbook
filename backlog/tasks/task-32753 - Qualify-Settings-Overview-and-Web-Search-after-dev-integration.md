---
id: TASK-32753
title: Qualify Settings Overview and Web Search after dev integration
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:10'
updated_date: '2026-09-17 20:32'
labels:
  - ui
  - design-system
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Requalify the existing Overview and Web Search journeys after integration, restoring meaningful test evidence under interpreter-lifetime profile ownership and repairing only confirmed user-facing gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Overview readiness and category links remain truthful and keyboard reachable at wide and compact widths.
- [x] #2 Web Search uses the selected private profile and preserves masked drafts, atomic Save/Revert, and explicit saved-only test behavior across navigation.
- [x] #3 Targeted tests and representative native dark/light journeys qualify the reviewed scope with explicit backend limits.
- [x] #4 At compact widths the Overview status body grows to its wrapped content and every focused category action paints its complete label.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md; backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: existing configuration, staged save and UI ownership contracts remain unchanged.
1. Inspect existing implementation, task history and configuration ownership; retain the reproducible fixture failure.
2. Migrate affected Web Search tests to the existing private-process helper, selecting config before imports and preserving original assertions.
3. Confirmed gap: Overview primary Vertical shrinks to one row at 80x24 while its content is 15 rows, clipping focused actions. Let the primary body grow with content and use existing stacked compact action layout when required; preserve the single detail scroll owner.
4. Exercise Overview links/readiness and Web Search keyboard, compact layout, draft, persistence and failure recovery with production CSS. Repair a confirmed product gap only after extending acceptance criteria and recording its failing case.
5. Run representative real private-profile native journeys in dark/light and wide/compact sizes; retain evidence and explicit provider limits.
6. Run targeted checks and scoped static analysis, self-review, update the completion ledger and save the verified changes to draft PR #2704.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Overview now grows its primary status body with wrapped content and stacks compact action rows, keeping keyboard-focused links fully painted. Rebuilt the lazy Settings sheet; no token values changed. Restored 19 Overview and 21 Web Search cases under the established private-profile lifetime harness, updated the existing Canvas privacy banner expectation, and added four production-CSS keyboard/theme/size journeys including resize focus, masked draft retention, Save/Revert, failure recovery and explicit saved-only tests.

76 distinct targeted cases and all seven preflight checks pass; scoped Ruff/formatter comparison adds no debt. Independent review found no remaining blocker. Four final native journeys (PID 35146) perform exact private config writes and real loopback HTTP 401-to-200 recovery, with eight inspected captures, eleven healthy databases, clean shutdown and unchanged default files. External provider availability, generation, sync, source switching and Backup/Restore remain outside this bounded review. Earlier runner attempts are documented honestly.

Updated the Settings user guide, Web Search surface brief, completion ledger and testing lesson. Existing ADR-012/033/150/161 apply; no new ADR. Evidence: Docs/superpowers/qa/2026-09-17-settings-overview-web-search/README.md.
<!-- SECTION:NOTES:END -->
