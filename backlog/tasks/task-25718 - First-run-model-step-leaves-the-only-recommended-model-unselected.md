---
id: TASK-25718
title: First-run model step leaves the only recommended model unselected
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The model step queries the configured provider, finds the available models, and labels one as recommended, but leaves every option unselected. Pressing Next proceeds with no model chosen and stamps the step complete. When exactly one model is offered and it is already marked recommended, requiring a separate click adds a step whose only outcome is a misconfigured install.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A sole available model marked recommended is selected by default
- [ ] #2 Advancing without a model selected is either prevented or clearly reported
- [ ] #3 The summary reflects the model actually chosen
<!-- AC:END -->
