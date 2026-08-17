---
id: TASK-17371
title: Enable sub-question decomposition and bounded multi-hop for shipped research runs
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-17 07:30'
labels:
  - research
  - web-tools
dependencies:
  - task-17370
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-question fan-out and gap-driven replanning both ship but are off by default: fan-out is opt-in via search settings, and the local research engine's `max_iterations` defaults to 1, so a real research run is single-facet and single-pass. Once task-17370 has measured what decomposition is worth, decide the shipped defaults from those numbers rather than from caution, and make the resulting spend legible to the user before a run starts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shipped default for research-run decomposition is set from the task-17370 measurement, and the choice is recorded with the numbers that justify it.
- [ ] #2 A user can see and change the decomposition settings for a run before launching it, and the setting persists.
- [ ] #3 The expected spend implication of the chosen default is documented where a user meets it, since fan-out multiplies gate LLM calls per run and iterations multiply rounds on top.
- [ ] #4 Existing runs, artifacts and tests that assume single-pass behaviour continue to pass or are updated with the reason recorded.
<!-- AC:END -->
