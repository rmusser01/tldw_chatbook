---
id: TASK-25717
title: First-run wizard discards the provider endpoint the user entered and tested
status: To Do
assignee: []
created_date: '2026-08-31 05:08'
labels:
  - console
  - ux-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provider step accepts an endpoint, confirms it with a successful connection test, and then does not persist it. The saved draft records only the provider key while the sibling voice step persists its own endpoint correctly. The one value that determines whether the product can reach a model is dropped after the interface confirmed it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An endpoint entered on the provider step is persisted with the rest of that step's draft
- [ ] #2 A successful connection test does not report success for a value that will not be retained
- [ ] #3 Completing first-run setup with a local provider leaves a configuration that can reach that provider
<!-- AC:END -->
