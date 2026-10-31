---
id: TASK-38425
title: 'Console: a model_not_found failure is categorised ''provider unavailable'''
status: To Do
assignee: []
created_date: '2026-10-31 03:32'
labels:
  - console
  - errors-recovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A provider's explicit model_not_found arrives as ChatModelUnavailableError, a ChatProviderError, so the gateway's failure sentence categorises it 'provider unavailable' even when the status is 400: 'Provider error from X: provider unavailable. Status: 400. ...'. Since TASK-33621.11 the row then advises choosing another model, which is right, but the category contradicts it. Only a 404 gets 'model or endpoint not found' today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A model_not_found failure on any status reads 'model or endpoint not found' (or equivalent) in its category
- [ ] #2 Other provider-unavailable failures keep their category
- [ ] #3 A test drives a 400 model_not_found body through the real gateway
<!-- AC:END -->
