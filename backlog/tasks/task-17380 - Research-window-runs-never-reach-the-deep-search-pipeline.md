---
id: TASK-17380
title: Research window runs never reach the deep-search pipeline
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-17 07:45'
labels:
  - research
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every research run launched from the Research window fails immediately without searching anything. The window builds the execution engine without the deep-search pipeline settings, so the pipeline rejects the run on its own required-parameter check, and the run's recorded reason names neither what was missing nor where it comes from. The same omission leaves the engine's gap-driven replanning permanently inert for window runs, because gap analysis reads its LLM from those settings. The Research screen is reachable from navigation again, so this is the shipped path a user meets; the Console /research command is unaffected because it assembles the settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run launched from the Research window reaches the search phase instead of failing on missing pipeline parameters
- [ ] #2 A window-launched run can perform gap-driven replanning, i.e. its synthesis LLM is configured like the Console command's
- [ ] #3 When the pipeline settings cannot be assembled, the window reports that in place of launching a run that cannot succeed
- [ ] #4 A run that reaches the real pipeline without usable parameters fails naming the missing keys and their configuration source, not with the pipeline's opaque message
- [ ] #5 Callers that inject their own search function keep running without pipeline parameters
- [ ] #6 Tests cover the window's assembly and both sides of the engine's pre-flight
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure exactly as the window launches a run (engine built with no search_params) and capture the recorded terminal reason.
2. Export the pipeline's required-parameter list from the pipeline itself so a caller can check its own assembly without duplicating the list.
3. Pre-flight those parameters in the engine, but only when the real pipeline is the search function, so injected search functions keep their own contract.
4. Assemble the settings in the window through the same shared assembly the Console command and the baseline recorder use; report an assembly failure instead of launching.
5. Cover both sides of the pre-flight and the window's assembly with tests.
<!-- SECTION:PLAN:END -->
