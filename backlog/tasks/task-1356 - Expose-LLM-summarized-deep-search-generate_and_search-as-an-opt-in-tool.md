---
id: TASK-1356
title: Expose LLM-summarized deep search (generate_and_search) as an opt-in tool
status: In Progress
assignee:
  - '@claude'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec `Docs/superpowers/specs/2026-08-07-deep-search-tool-design.md` + plan `Docs/superpowers/plans/2026-08-07-deep-search-1356.md`: 6 SDD tasks — phase-1 hardening + backend timeouts, phase-2 port from tldw_server2's live module (FinalAnswerDict/chunking/confidence/cancel_event/citation integrity), pre-scrape SSRF guard in Utils/egress.py, [SearchSettings] config revival, loop-safe web_deep_search tool core, double-opt-in gated registration.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tool registered in hub with Ask default,Returns synthesized answer + cited sources,LLM calls mocked in tests; config flag documented
<!-- AC:END -->
