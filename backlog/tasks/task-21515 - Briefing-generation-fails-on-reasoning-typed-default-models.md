---
id: TASK-21515
title: Briefing generation fails on reasoning-typed default models
status: To Do
assignee: []
created_date: '2026-08-30 05:59'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live verification of TASK-21513 (Daily Reports demo) reproduced: with the config-default deepseek-v4-flash, the ~15k-char briefing prompt makes reasoning consume all of BRIEFING_MAX_TOKENS=2000 in Subscriptions/briefing_service.py, so the provider returns finish=length with empty content and the briefing row fails with 'returned an empty response'. deepseek-chat completes fine. This is pre-existing Watchlists behavior, not a TASK-21513 regression, but it breaks the demo's one-click promise for users whose default endpoint is a reasoning-typed model. Candidate fixes: raise BRIEFING_MAX_TOKENS, exclude/override reasoning models for briefings, or provider-aware max_tokens.
<!-- SECTION:DESCRIPTION:END -->
