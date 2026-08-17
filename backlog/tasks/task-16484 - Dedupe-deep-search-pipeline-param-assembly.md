---
id: TASK-16484
title: Dedupe deep-search pipeline param assembly
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 03:39'
updated_date: '2026-08-16 03:41'
labels:
  - research
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The web_deep_search tool, the Console /research handler, and the baseline script each hand-assemble the search_params dict from _deep_search_settings - three copies of the same ~15-key mapping that will drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One shared public helper assembles the pipeline params from tool settings with override support,The tool, the Console handler, and the baseline script all use it,Tests pin the helper output shape and override behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `deep_search_pipeline_params(*, engine, max_results, subquery, max_queries, respect_robots, extra)` in web_tool_impls: one assembly from `[SearchSettings]` with per-key overrides and an `extra` merge. The web_deep_search tool, the Console `/research` handler, and the baseline script now all call it (the script's spend bounds -- subquery off, one query, robots on -- are overrides, not a fourth copy). Tool behavior unchanged (its suite passes untouched).
<!-- SECTION:NOTES:END -->
