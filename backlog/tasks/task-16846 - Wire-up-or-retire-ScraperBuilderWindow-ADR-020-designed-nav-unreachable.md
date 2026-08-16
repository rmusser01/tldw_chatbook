---
id: TASK-16846
title: 'Wire up or retire ScraperBuilderWindow (ADR-020 designed, nav-unreachable)'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - ui
  - dead-code
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15991 (PR #1701) made `UI/ScraperBuilderWindow.py` *openable* — it had never once
composed successfully (nonexistent `FormBuilder.create_switch`, a `Collapsible`
positional-string `MountError`, two backwards Selects), proof that nothing had ever
reached it. But it remains **nav-unreachable** at dev `ee741cf10`: repo-wide grep finds
`ScraperBuilderWindow` referenced only by its own file and its regression test
(`Tests/UI/test_scraper_builder_window.py`); zero matches in
`UI/Navigation/screen_registry.py` or any command-palette provider. User impact of the
15991 fix is nil until this decision is made.

The design record says it is a feature, not a leftover: "ADR-020: Visual Scraper Builder"
(`Docs/Development/Subscriptions/Subscriptions-Implementation-1.md:303`, Status:
Accepted, 2025-08-01 — note the number collides with an unrelated `backlog/decisions/`
ADR-020, a pre-existing doc-set collision) describes an interactive UI for testing
selectors and building extraction rules. Nothing anywhere marks it retired.

Decide (owner call): wire it into navigation/the Watchlists surface it was designed to
serve, or retire it and amend the ADR — the same fork its nav-unreachable sibling
`UI/SiteConfigSettings.py` sits on (whose own live Select bug is filed separately). If
wired, a live-drive check of the full build-test-export flow is due, since the window has
never been exercised by a user.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An explicit decision is recorded against ADR-020 (wire or retire), with the ADR/doc updated to match
- [ ] #2 If wired: the window is reachable through real navigation, and its primary flow works in a live drive (not just the mount test)
- [ ] #3 If retired: the window, its test, and its `SiteConfigSettings` sibling's disposition are handled together with reachability evidence
<!-- AC:END -->
