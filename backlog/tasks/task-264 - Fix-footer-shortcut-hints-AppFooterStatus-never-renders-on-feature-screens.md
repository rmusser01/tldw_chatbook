---
id: TASK-264
title: 'Fix footer shortcut hints: AppFooterStatus never renders on feature screens'
status: To Do
assignee: []
created_date: '2026-07-17 02:00'
labels:
  - ux
  - infrastructure
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live QA (MCP Hub Phase 4) found the footer shortcut-hint system is dead app-wide: AppFooterStatus is mounted once on the App's base screen, but every feature screen (MCP, Chat/Console, Personas, Library — all BaseAppScreen subclasses) is a separately-pushed Screen with its own MainNavigationBar + Footer, permanently covering it. Every set_workbench_shortcuts registration (Console F6 hints, MCP '1-4 mode / a add server / t test tool / space cycle permission') renders nowhere. The New_UI mockups show a persistent bottom key-hint strip as the intended design. Evidence: Docs/superpowers/qa/mcp-hub-phase4-2026-07/README.md defect 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registered workbench shortcut hints visibly render on every BaseAppScreen that registers them,MCP and Console hint sets verified live,Decision recorded: fix the AppFooterStatus mounting vs ship a per-screen hint strip
<!-- AC:END -->
