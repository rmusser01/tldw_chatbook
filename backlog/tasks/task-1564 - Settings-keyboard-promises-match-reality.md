---
id: TASK-1564
title: 'Settings: keyboard promises match reality (footer bindings, setting-level filter)'
status: To Do
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P2): the static footer advertises "t test category" on
categories that answer with "No test action is available for this Settings
category yet." (toast contradicts chrome); RAG appends extra accelerators
only in its own footer. The "/" filter matches category names+descriptions
(verified live: "console" -> 6 matches, names its Enter target) but cannot
find an individual setting, even though the Scope Inspector already knows
every owned TOML key per category.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer bindings reflect the active category (no advertised key that no-ops).
- [ ] #2 "/" filter also indexes owned config keys / setting labels and opens the owning category.
- [ ] #3 Filter status still names the Enter target for the top match.
<!-- AC:END -->
