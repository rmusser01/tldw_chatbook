---
id: TASK-32349
title: >-
  Library: a new profile with a pre-written config lands on the full rail with
  'Back to Get started' orphaned
status: To Do
assignee: []
created_date: '2026-09-11 06:15'
labels:
  - library
  - onboarding
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An empty profile whose config.toml existed before first launch opens the full nine-destination rail (Media (0) … Collections (0)) with 'Back to Get started' at the bottom, a destination the user never saw (B D3 cap 63/64, A cap 04). Cause PROVEN in critique #8: coerce_library_lifecycle(raw=None, is_new_profile=False) resolves EXPANDED; the same happens to a real user who completes setup and quits before visiting Library. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A profile with no content yet lands on Get started regardless of whether its config file pre-dates the first Library visit
- [ ] #2 'Back to Get started' is offered only after the user has seen Get started
- [ ] #3 Pinned
<!-- AC:END -->
