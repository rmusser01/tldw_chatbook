---
id: TASK-404
title: Stop sending default temperature/top_p to OpenAI reasoning models
status: To Do
assignee: []
created_date: '2026-07-21 07:12'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
chat_with_openai force-includes default temperature and top_p in the request; OpenAI reasoning models (o-series, gpt-5 family) reject these with HTTP 400 'Unsupported parameter', so ANY call routed to them through this handler fails — including the Responses-API branch. Found during task-403 verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reasoning-model requests omit unsupported sampling parameters,Non-reasoning models keep today's parameter behavior,Regression test covers both shapes
<!-- AC:END -->
