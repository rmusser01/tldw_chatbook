---
id: TASK-32689
title: Browse Git catalogs and import selected marketplace sources
status: To Do
assignee: []
created_date: '2026-09-16 04:31'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32687
  - TASK-32688
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users discover packages and copy selected existing app sources while keeping discovery separate from installation and authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All three supported marketplace paths normalize source-qualified listings and package/catalog overlays; local entry paths resolve against the catalog root.
- [ ] #2 Refresh changes cached listings only, check-updates changes candidates only, and unsupported source types remain visible without hiding valid siblings.
- [ ] #3 Explicit Cursor/Codex import previews allowlisted source/package references without foreign credentials, approvals, trust or automatic enablement; broken caches require clean sources.
- [ ] #4 Source removal does not uninstall packages; stale cache, refresh errors, pagination and evidence invalidation remain distinct and bounded.
<!-- AC:END -->
