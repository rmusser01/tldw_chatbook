---
id: TASK-394
title: Scope global selectors in the Settings splash/theme CSS module
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 18:35'
labels: [css, tech-debt]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`components/_settings_splash_theme.tcss` (extracted verbatim from the generated bundle in the PR #723 resync, originally hand-written into the bundle by 778f75813) contains selectors that apply app-wide because TCSS compiles into one global stylesheet: a bare `VerticalScroll` type selector (line ~134) and generic class names `.section-header` / `.setting-label` that other modules also define (`stats_screen.css`, `_chat.tcss` — 6 occurrences bundle-wide). This is pre-existing shipped behavior, deliberately preserved byte-identical during the resync; tightening it means renaming/scoping the selectors AND updating the paired widget code (`settings_splash_screen_viewer.py`, `settings_theme_editor.py`), then re-checking the other screens that share those class names. A dev's own second bare `VerticalScroll` exists at another module (bundle ~8218) — consider sweeping all modules for unscoped type selectors while here. Splitting the module into per-feature files (theme vs splash) can ride along if done, keeping manifest cascade order intact.

Raised by review on PR #723 (Gemini findings on `_settings_splash_theme.tcss:105/:138`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No bare type selectors (e.g. `VerticalScroll`) or cross-module generic class names remain in the splash/theme module; styles are scoped or feature-prefixed
- [x] #2 Settings splash gallery and theme editor render unchanged after the scoping (visual check or snapshot)
- [x] #3 Screens that shared the old generic class names (`stats_screen`, chat) are verified unaffected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Investigation reframed the task: the splash/theme module's whole TOP section was
the OLD (pre-Settings) splash viewer's stylesheet. The current viewer + theme
editor use only the module's BOTTOM section (#settings-theme-*, .settings-splash-*,
.preview-*-demo). The top section held (a) app-wide GENERIC component classes
(.setting-label 63 uses, .settings-section 35, .help-text 23, .action-buttons 18,
.section-header 12, .card-list/.preview-panel/.preview-container/.preview-content —
used across settings/coderepo/dataset/eval/chunk widgets), (b) a bare VerticalScroll
that (with a second bare one in _conversations) was the app's de-facto scrollbar
default, and (c) seven zero-importer orphans. These are NOT scopeable leaks — they
are app-wide components misplaced in a feature module. FIX: (1) moved the shared
component rules VERBATIM to a purpose-named components/_shared_components.tcss placed
at the SAME manifest position (immediately after the splash module) so the bundle
cascade is byte-for-byte equivalent; (2) consolidated the two bare VerticalScroll
rules into ONE deliberate app-wide default in core/_base.tcss (union of both -> same
effective scrollbar) and removed them from splash + _conversations; (3) removed the
7 orphans (card-list-container/-panel, card-info, preview-widget, checkbox-button,
modal-container, splash-gallery-modal). VERIFICATION: a static before/after effective-
style diff proved ALL 9 relocated classes + VerticalScroll have byte-identical merged
styles app-wide (AC#2/#3 — zero rendering change); the app boots clean on the new
bundle and Settings renders (served-app). 3 new regression tests lock: splash module
has no bare/generic selectors, the moved rules are in _shared_components + bundle, and
VerticalScroll is in core. css_build_integrity + bundle-sync guard green.
<!-- SECTION:NOTES:END -->
