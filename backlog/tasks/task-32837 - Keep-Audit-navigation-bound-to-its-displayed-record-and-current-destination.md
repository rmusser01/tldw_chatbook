---
id: TASK-32837
title: Keep Audit navigation bound to its displayed record and current destination
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:23'
updated_date: '2026-09-19 19:44'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Audit Open tool and Adjust permission actions keep the identity the user selected and handle destinations that disappear during navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retired Audit controls cannot navigate to a replacement record; current controls preserve rendered tool and profile identity.
- [x] #2 Missing destination rows do not leave stale tool or permission detail; current and filtered targets still navigate correctly.
- [x] #3 Targeted ownership/routing regressions and private dark/light native journeys verify the bounded repair.
- [x] #4 Queued Audit navigation is ignored when its control or owning view becomes hidden, disabled or covered; live controls remain repeatable and keep their rendered identity.
- [x] #5 Audit destination selection waits for active catalog publication so a newly published tool is not falsely reported unavailable while its row is still being built.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve approved control identity, availability guards, row-loss handling and visuals.
2. Reproduce Qodo findings 4/5 by pausing real catalog publication before tables update and navigating to a newly published identity; serialize row availability and rendering with the existing publication lock. Same-ID definition refresh remains PR2726.
3. Add structured method docs and group the modified test imports for Qodo findings 1-3. Verify targeted tests, fresh native evidence, independent review and current-head CI; resolve addressed review threads and merge under the owner approval.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair existing publication and selection contracts using the existing lock, no new storage, authority or service boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit actions retain rendered identity, reject retired/unavailable controls, wait for complete catalog row publication and honor missing destination rows. Both report conflicts retained current-dev history and Audit notes; owner approved the gallery and continuation. Addressed all five Qodo findings: structured method docs, contiguous test import groups and both false-unavailable publication races. Final targeted run: 147 passed; two publication race tests failed before and passed after. Seven preflight guards, unchanged Ruff baselines, formatted changed ranges and independent review pass. Final native replay matches all 24 approved terminal captures except synthetic timestamps; clean private lifecycle, ten healthy databases, unchanged defaults/permission profiles and no network or execution. Evidence: Docs/superpowers/qa/2026-09-18-mcp-audit-navigation/current-dev/qodo. ADR-150/161 apply; no new ADR. Same-ID definition freshness remains PR2726. Current-head CI/review is required before the authorized merge.
<!-- SECTION:NOTES:END -->
