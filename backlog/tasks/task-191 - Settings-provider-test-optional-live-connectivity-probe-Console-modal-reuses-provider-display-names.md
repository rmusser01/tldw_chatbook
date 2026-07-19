---
id: TASK-191
title: >-
  Settings provider test: optional live connectivity probe; Console modal reuses
  provider display names
status: Done
assignee: []
created_date: '2026-07-12 05:33'
labels:
  - ux
  - settings
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-ups from the 2026-07-11 remediation: (1) the Settings 'Test' action is a local readiness check by design - a live probe (GET /v1/models // /health with short timeout, like the Console gateway probes) would let the toast report reachable/refused/timeout; (2) the Console Settings modal's provider dropdown still renders raw keys - reuse the Settings PROVIDER_DISPLAY_NAMES mapping (single source) so both surfaces match.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Test toast can report actual endpoint reachability for local providers,Console modal provider options show the same display names as Settings,Display-name mapping has a single source of truth
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on claude/uat-upgrade-wave-2026-07 (commits b843daac/c5aa69d2/54746ffa): shared Chat/provider_catalog is the single display-name source consumed by Settings and the Console modal; Settings Test now live-probes URL-based providers and reports reachable(with model count)/refused/timeout/HTTP status. Residual: console_model_popover still renders raw keys (minor).
<!-- SECTION:NOTES:END -->
