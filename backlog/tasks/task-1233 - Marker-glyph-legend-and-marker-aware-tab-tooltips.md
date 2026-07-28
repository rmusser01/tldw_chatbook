---
id: TASK-1233
title: 'Marker glyph legend and marker-aware tab tooltips'
status: To Do
assignee: []
created_date: '2026-07-28 09:30'
labels: [console, fleet-ux, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F4: the fleet's status vocabulary (● running, ◆ needs approval, ✓ finished, ✗ failed) has no legend anywhere, and tab tooltips say only "Switch to Console tab: X" even when the tab carries a marker. Recognition-over-recall failure for the core status language.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tab (and sidebar-row) tooltips include the marker meaning when one is present ("X — waiting for approval").
- [ ] #2 A legend exists in Help (rides task-1232's Agents section if both land).
<!-- AC:END -->
