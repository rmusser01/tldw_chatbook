---
id: TASK-32837
title: Keep Audit navigation bound to its displayed record and current destination
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 05:23'
updated_date: '2026-09-19 19:27'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve both report histories while rebasing the existing Audit PR onto merged dev.
2. Exercise actual queued Button.Pressed messages, pending pruning and unavailable control/view ownership. Reproduce the unavailable-view failure before applying the adjacent permission-jump guard.
3. Refresh the validated native runner and private dark/light compact/wide evidence. Run targeted tests, preflight and independent review; update draft PR2724 for final visual approval.
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine hardening of the existing action ownership and selection contracts; no new runtime authority, storage or application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audit actions capture rendered tool/profile identity, invalidate retired controls before pruning, reject unavailable controls or owning views, and remain repeatable when current. Both destinations honor failed row selection and clear stale detail with the existing warning. Updated existing draft PR2724 by rebasing onto merged dev 29b0a31df4701160a3c805e1bf490c76b9353964; both documentation conflicts preserve current dev history and Audit notes, while production code auto-merged. Current verification: 145 distinct targeted checks, seven preflight guards, no new Ruff diagnostics, formatted changed ranges and independent review without blockers. Twelve unavailable-view cases failed before the guard. Three old inspector setup errors were fixed using the existing private-profile process wrapper without changing their assertions or production recovery gates. The refreshed native launcher validates paths/arguments, uses the supported terminal warm-up and blocks network connects. Twenty-four inspected dark/light captures at 120x40 and 170x48 verify focused actions, filtered destinations and missing-target warnings; clean private lifecycle, unchanged defaults and permission profiles, no execution/network. Evidence and exact conflict choices: Docs/superpowers/qa/2026-09-18-mcp-audit-navigation/README.md and CURRENT-DEV-REVIEW.md. ADR-150/161 apply; no new ADR. Same-ID catalog freshness remains PR2726. Current-head CI/review and PR2724 visual approval are still required before merge.
<!-- SECTION:NOTES:END -->
