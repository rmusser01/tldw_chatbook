---
id: TASK-15998
title: 'De-quirk or justify the 24 comma-list selector leaks now live from boot'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - css
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual scopes only the LAST selector of a comma list in scoped DEFAULT_CSS (parser-level, confirmed in the 15450 review); the consolidation de-quirks the screen sheets (`build_css.py` `scope_every_selector=True`) precisely because they are live from boot — but the identical argument applies to the 50 consolidated widget classes, whose leaked selectors used to go live at first mount and are now live from app start. Enumerated exposure: 24 leaked selectors across 6 classes (`MCPAuditMode`, `MCPToolsMode`, `MainNavigationBar`, `LibraryScreen`, `MCPScreen`, `SyncStatusWidget`) — all ID selectors or feature-specific class chains, inert in practice, and the boot-stop computed-style diff (dev vs branch) was identical. Either extend de-quirking to the widget sheets or record the asymmetry as a decision with the enumeration pinned by a test, so the leak set cannot grow silently. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Widget-sheet comma-list selectors are either fully scoped like the screen sheets, or the current 24-selector leak set is pinned by a test that fails on growth
- [ ] #2 Computed-style parity evidence for whichever path is taken
- [ ] #3 The decision and rationale are recorded next to the builder code
<!-- AC:END -->
