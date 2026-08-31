---
id: TASK-25712
title: >-
  Console send is blocked before provider dispatch when trace capture cannot be
  saved
status: To Do
assignee: []
created_date: '2026-08-31 05:07'
labels:
  - console
  - ux-review
  - p0
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console send that cannot record a trace is refused entirely: the interrupt card states "The provider was not contacted". Reproduced on a clean first-run profile and on a repaired existing profile, against a local provider that answers the same request in 0.8s over curl. This makes the product's core loop unusable rather than degraded, and a diagnostics feature the user never enabled becomes a hard dependency of chatting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send whose trace capture fails still reaches the provider
- [ ] #2 Trace-capture failure is surfaced as a non-blocking warning, not a card that halts dispatch
- [ ] #3 A clean first-run profile with a reachable provider completes a send end to end without any interrupt card
<!-- AC:END -->
