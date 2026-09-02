---
id: TASK-29227
title: Compose the portable Tool Pack application service
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-02 14:00'
labels:
  - tool-packs
  - app-wiring
  - startup
  - receipts
dependencies:
  - TASK-29226
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the completed portable Tool Pack lifecycle behind one deferred app-owned service, attach its binding authority exactly once, and reconcile receipt storage without expanding startup-critical imports or leaking sensitive failure details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One app-facing `ToolPackService` exposes separate export capture/publication, import inspection/unbound activation, first-bind review/confirmation, removal, and immutable non-tombstoned profile-listing operations without becoming a direct policy-rule editor.
- [ ] #2 Profile listings report exact id, origin, lifecycle validity, binding state, active/archived references, compact policy counts, receipt health, and removal eligibility from current strict authority and workspace state; tombstones stay hidden while their ids remain reserved.
- [ ] #3 Service composition shares one lifecycle coordinator across activation, binding, and removal; uses the sealed V1 inventory assembly and user-data receipt root; and exposes only stable bounded error categories without paths, credentials, commands, environment values, or archive/receipt excerpts.
- [ ] #4 `app.py` composes exactly one complete service only after MCP and workspace prerequisites exist, attaches its binding guard exactly once, exposes a stable unavailable state on failure, and never attaches a partial guard or eagerly imports Tool Pack implementation modules on the startup-critical path.
- [ ] #5 Receipt reconciliation runs only after UI readiness/first use, protects every authoritative referenced and live-owned receipt, honors orphan grace and authenticated regular-file checks, and remains bounded/fail-safe for fresh, symlinked, unknown, and corrupt entries.
- [ ] #6 Focused service, startup-hygiene, and startup-performance tests plus scoped Ruff/format and diff checks pass; independent review has no unresolved Critical or Important findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing facade tests for split review/commit operations, immutable presentation listings, stable error privacy, and no rule-edit mutation surface.
2. Compose the existing export, publication, import, activation, binding, removal, inventory, and receipt owners behind one shared-lifecycle `ToolPackService`.
3. Add focused app-wiring tests for prerequisite ordering, exact guard attachment, unavailable/partial-failure behavior, user-data receipt placement, and deferred imports.
4. Add bounded post-ready receipt reconciliation using current profile receipt links plus in-flight activation ownership; prove only eligible expired orphans are reclaimed.
5. Run the prescribed targeted suites, scoped static/diff checks, self-review, and independent review before closeout.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the app/service ownership boundary, deferred composition, receipt authority and reconciliation, stable privacy-safe outcomes, and Settings-versus-MCP editing split.
<!-- SECTION:PLAN:END -->
