---
id: TASK-183
title: >-
  First-run setup card fixes: button label, honest step states, local-provider
  path
status: Done
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
- [x] #1 Primary button reads as plain language (e.g. Set up a provider)
- [x] #2 Checklist steps reflect user actions not template defaults
- [x] #3 Copy offers the no-API-key local path
- [x] #4 Blocked state does not invite typing below
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
