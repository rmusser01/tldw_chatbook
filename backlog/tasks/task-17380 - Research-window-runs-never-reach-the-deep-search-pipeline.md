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
- [x] #2 A window-launched run can perform gap-driven replanning, i.e. its synthesis LLM is configured like the Console command's
- [x] #3 When the pipeline settings cannot be assembled, the window reports that in place of launching a run that cannot succeed
- [x] #4 A run that reaches the real pipeline without usable parameters fails naming the missing keys and their configuration source, not with the pipeline's opaque message
- [x] #5 Callers that inject their own search function keep running without pipeline parameters
- [x] #6 Tests cover the window's assembly and both sides of the engine's pre-flight
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure exactly as the window launches a run (engine built with no search_params) and capture the recorded terminal reason.
2. Export the pipeline's required-parameter list from the pipeline itself so a caller can check its own assembly without duplicating the list.
3. Pre-flight those parameters in the engine, but only when the real pipeline is the search function, so injected search functions keep their own contract.
4. Assemble the settings in the window through the same shared assembly the Console command and the baseline recorder use; report an assembly failure instead of launching.
5. Cover both sides of the pre-flight and the window's assembly with tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The window built the engine with no `search_params` at all, so the pipeline's
own required-key check rejected every window-launched run before a single
search. An empty dict is falsy, so the message was
`ValueError("Invalid search_params parameter")` -- it named neither the
missing keys nor their source. Reproduced first by launching a run exactly
the way `_start_local_engine` does: `status: failed`,
`progress_message: "Invalid search_params parameter"`.

Three changes, smallest first:

- `WebSearch_APIs` exports `GENERATE_AND_SEARCH_REQUIRED_PARAMS` and its own
  validation loop reads it, so a caller can check its assembly without
  duplicating the list.
- `LocalResearchEngine` pre-flights those keys inside `execute_run`'s try, so
  the existing terminal-failure path reports it, and only when
  `search_fn is None` at construction -- an injected search function carries
  its own contract, and enforcing the web pipeline's requirements on it would
  have broken every engine test and any future non-web lane.
- `Research_Window._start_local_engine` assembles via
  `deep_search_pipeline_params()` (the same assembly Console `/research` and
  the baseline recorder use, task-16484) and reports an assembly failure
  instead of spending a run that cannot succeed.

Second defect closed by the same fix: `_default_gap_fn` reads
`final_answer_llm` out of `search_params` and returns "no gaps" when it is
unset, so gap-driven replanning (task-16324) was permanently inert for window
runs regardless of `max_iterations`.

AC #1 is deliberately left open: the assembly and the pre-flight are unit
verified, but a window run actually reaching the search phase needs a live
run against a configured LLM endpoint, and the local Qwen endpoint used for
every other live verification in this stream is down. It stays open rather
than being claimed from unit evidence.

Modified: `tldw_chatbook/UI/Research_Window.py`,
`tldw_chatbook/Research_Interop/local_research_engine.py`,
`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`,
`Tests/UI/test_research_screen.py`,
`Tests/Research/test_local_research_engine.py`.
<!-- SECTION:NOTES:END -->
