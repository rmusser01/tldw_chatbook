---
id: TASK-189
title: >-
  Split Settings Providers & Models into Connect section above collapsed
  Generation defaults
status: To Do
assignee: []
created_date: '2026-07-12 03:05'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Upgrade opportunity from core-loop UAT 2026-07-11: the only first-run job (provider, model, endpoint, credentials, test) sits below ~14 sampling fields, and gated fields render 8 rows of 'Unavailable for <provider>' noise. Restructure the category: a Connect block first, then a collapsed Generation defaults (sampling/transport) block; hide or summarize gated-unavailable fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Credentials and endpoint are visible without scrolling past sampling fields on first entry,Sampling and provider-specific tuning live in a collapsed Generation defaults section,Fields unavailable for the selected provider are collapsed or summarized rather than listed row-by-row
<!-- AC:END -->
