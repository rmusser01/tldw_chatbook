---
id: TASK-226
title: 'Console rail row labels clip the [N Sub-Agents] badge at default width'
status: To Do
assignee: []
created_date: '2026-07-14 03:33'
labels:
  - console
  - agents
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The agent-runtime live gate (Docs/superpowers/qa/agent-runtime-2026-07/) showed the conversation-row badge renders as '[1' at the rail's default width — the same truncation every row label already has (titles clip at ~20 chars). Pre-existing display constraint, surfaced by the new badge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The [N Sub-Agents] badge is fully visible on conversation rows at the default rail width,Row titles degrade gracefully (ellipsis) without swallowing trailing badges
<!-- AC:END -->
