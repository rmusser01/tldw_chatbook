---
id: TASK-183
title: >-
  First-run setup card fixes: button label, honest step states, local-provider
  path
status: To Do
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: the blocking card's primary button is literally 'Configure API+API Key'; step 2 'Pick a model' is pre-checked on a virgin profile because the template default gpt-4o counts as picked; step 1 'Add an API key' misleads local-model users (llama.cpp needs none); the card says 'Type below, Enter to send' while the composer is blocked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Primary button reads as plain language (e.g. Set up a provider)
- [ ] #2 Checklist steps reflect user actions not template defaults
- [ ] #3 Copy offers the no-API-key local path
- [ ] #4 Blocked state does not invite typing below
<!-- AC:END -->
