---
id: TASK-194
title: console_model_popover uses the shared provider display-name catalog
status: To Do
assignee: []
created_date: '2026-07-12 12:44'
labels:
  - ux
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from task-191: Settings and the Console settings modal render display names from Chat/provider_catalog, but the alt+m model popover still shows raw provider keys (llama_cpp etc.).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Model popover provider labels match Settings/modal display names,Single source of truth (Chat/provider_catalog) - no duplicated name maps
<!-- AC:END -->
