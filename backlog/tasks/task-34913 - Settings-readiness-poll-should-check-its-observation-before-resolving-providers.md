---
id: TASK-34913
title: >-
  Settings-readiness-poll-should-check-its-observation-before-resolving-providers
status: To Do
assignee: []
created_date: '2026-10-13 14:44'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings 0.25s _poll_subscription_readiness timer runs get_provider_readiness (env walk + provider settings + endpoint registry + credential snapshots) on EVERY tick, and only afterwards compares the observation that gates the repaint (settings_screen.py:5103-5111). The expensive work runs 4x/second whenever Settings sits on OVERVIEW (the default landing category) or PROVIDERS_MODELS, even when nothing changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cheap inputs (category and provider key) are compared before any expensive readiness resolution,get_provider_readiness runs only when a cheap fingerprint changes or on explicit refresh,Poll still refreshes within one tick when a credential genuinely changes (test)
<!-- AC:END -->
