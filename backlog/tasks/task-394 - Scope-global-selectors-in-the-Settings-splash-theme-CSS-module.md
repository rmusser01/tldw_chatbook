---
id: TASK-394
title: Scope global selectors in the Settings splash/theme CSS module
status: To Do
assignee: []
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
- [ ] #1 No bare type selectors (e.g. `VerticalScroll`) or cross-module generic class names remain in the splash/theme module; styles are scoped or feature-prefixed
- [ ] #2 Settings splash gallery and theme editor render unchanged after the scoping (visual check or snapshot)
- [ ] #3 Screens that shared the old generic class names (`stats_screen`, chat) are verified unaffected
<!-- AC:END -->
