---
id: TASK-32674
title: Drain plugin work before applying updates and rollback
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:21'
updated_date: '2026-10-01 02:57'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32673
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply one reviewed package revision without overlapping incompatible active work or silently restoring historical permissions.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Applying update fences new old-revision admission, shows leases and approvals, and prevents Stop continuations from extending the drain.
- [x] #2 Users can wait, cancel before commitment or explicitly cancel affected work; updates publish only after confirmed drain and cleanup.
- [x] #3 Rollback uses the same review and commit gates with current disables, mappings and grants; shared-data and external-effect rollback remain explicitly unsupported.
- [x] #4 New or newly supported components stay unselected, immutable prior packages remain available for comparison, and quota/retention protects live and recovery material.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
Reason: implements accepted revision drains, authenticated bounded retention, issued retry custody and managed continuation restrictions through existing owners.
1. Port reviewed F7 behavioral fixtures and establish RED through real retained run and retention entries.
2. Integrate the reviewed F7 increment, adapting current child definition bounds, hook consent/carriers, exact worker ownership and current continuation APIs.
3. Qualify actual held approvals/runs, proposal and work cancellation, rollback exclusions, quota/retirement fault boundaries, legacy cutover and managed archive/fleet resume with positive controls.
4. Run targeted neighboring checks, owned lint/format and shared baseline comparison; self-review all ACs, record limits/docs, complete via CLI and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated reviewed F7 revision drains, immutable update/rollback, authenticated bounded retention, issued phase-preserving retries and managed continuation pins through current owners. Preserved hook consent/carriers, child definition limits and worker custody. ADR-162/163/063. Core 105 passed; clean neighbor authority/codec 319 and native fleet/Console 78; final fleet/Console/archive 125 passed without warnings after fixing selected-profile fixtures and actual fixture database teardown. Direct archive transport 2 passed. Owned lint/format, changed syntax/whitespace and shared lint baseline comparison pass. Evidence and platform limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. No external providers/keychain, full sweep, Windows/Linux or hardware power-loss claim; native Stop remains H5.
<!-- SECTION:NOTES:END -->
