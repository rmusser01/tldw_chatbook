---
id: TASK-32673
title: Stop and revoke plugin work within the requested scope
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:20'
updated_date: '2026-10-01 02:33'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32672
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users immediately stop plugin activity while preserving unrelated authorized work and reporting persistence honestly.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disable here, global-default changes, Disable everywhere and uninstall invalidate exactly their specified scopes; old approvals and callbacks never revive on re-enable.
- [x] #2 Live admission closes and host cancellation starts before waiting for trust unlock or storage writes; affected plugin cleanup callbacks are suppressed immediately.
- [x] #3 Durable disable/removal success is distinct from a session-only block and confirmed local stop; retry and failure cannot reopen the live scope or release surviving resource ownership.
- [x] #4 Uninstall removes only installation-owned grants and registrations, retains data by default and leaves independent credentials/connections owned by their existing services.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements accepted scoped revocation, cancellation ownership and protected persistence contracts.
1. Port reviewed F6 behavioral fixtures and establish RED before production edits.
2. Integrate immediate live fences, retained cancellation and separate durable disable/uninstall receipts through existing owners, preserving F5/H4 fixes.
3. Qualify real scoped Console/provider cancellation, stalled/failed storage, re-enable fencing, uninstall retention, surviving processes and exact retries.
4. Run targeted neighbor checks, syntax/whitespace and owned lint/format; compare shared baseline diagnostics, update evidence/docs and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented scoped synchronous admission sealing and retained exact host cancellation before storage access. Durable disable/uninstall, confirmed stop and pending cleanup remain separate; stale callbacks cannot revive. Uninstall commits owned tombstones/grant removals before descriptor-relative package cleanup and retains data/independent credentials. Changes: Plugins revocation/admission/coordinator/service/runtime-owner/review, actual Console provider closure, operation documentation and real A/B/process/persistence fixtures. ADR required: yes; existing ADR-162/163 apply. RED reached missing revocation after real activation/child controls. GREEN: 45 core checks in 155.21s and 123 native/admission/coordinator/recovery/lineage neighbors in 248.90s, no warnings/skips. Full plugin Ruff/format, changed syntax/whitespace and unchanged controller lint baseline pass. Self-reviewed all ACs; evidence and platform limits: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. No full sweep or real external provider/keychain/cross-platform qualification. Cross-session issued lookup and history retention remain the next dependency F7.
<!-- SECTION:NOTES:END -->
