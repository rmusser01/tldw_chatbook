---
id: TASK-1356
title: Expose LLM-summarized deep search (generate_and_search) as an opt-in tool
status: To Do
assignee: []
created_date: '2026-08-05 06:04'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebSearch_APIs.py contains a full pipeline (sub-query generation, result scraping, relevance filtering, LLM aggregation) that no tool exposes. Once web_search/web_fetch land (task-1354), surface it as an opt-in deep-research tool for the Console/MCP.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tool registered in hub with Ask default,Returns synthesized answer + cited sources,LLM calls mocked in tests; config flag documented
<!-- AC:END -->
