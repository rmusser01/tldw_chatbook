---
id: TASK-17371
title: Enable sub-question decomposition and bounded multi-hop for shipped research runs
status: Done
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
- [x] #1 The shipped default for research-run decomposition is set from the task-17370 measurement, and the choice is recorded with the numbers that justify it.
- [x] #2 A user can see and change the decomposition settings for a run before launching it, and the setting persists.
- [x] #3 The expected spend implication of the chosen default is documented where a user meets it, since fan-out multiplies gate LLM calls per run and iterations multiply rounds on top.
- [x] #4 Existing runs, artifacts and tests that assume single-pass behaviour continue to pass or are updated with the reason recorded.
<!-- AC:END -->

## Measured evidence for the defaults decision (task-17370)

Recorded here so AC #1 can be decided from numbers rather than caution. All
from the repositories lane, local Qwen3.8-27B, same questions/engine/bounds as
the 0.42 baseline; see the "verdict" section of
`Docs/Development/research-report-eval-baseline.md`.

**Fan-out (`subquery_generation` + `search_default_max_queries`)**

- Effect on the relevance gate: none measurable (0.42 -> 0.38, flat within this
  model's run-to-run variance).
- Effect on retrieval: none on this lane -- the paper providers only see round
  1's `[question]`, so generated sub-queries never reach them (task-17372).
- Cost: one extra LLM call to generate the sub-questions, plus up to
  `max_queries - 1` extra searches, each with its own per-result gate calls.
- Reading: do NOT turn this on by default while task-17372 stands. It buys
  nothing measurable here and costs per-query gate spend. Re-measure after
  17372, when fan-out would actually change what is retrieved.

**Multi-hop (`max_iterations`)**

- Effect: positive where the synthesis path was intact -- Q2 held its gate rate
  while going 24 -> 39 resolved markers and 0.77 -> 0.95 citation density; all
  three 2-round arms beat the 1-round arm on Q1's gate rate.
- Cost, measured: search calls went 3 -> 12 for three questions (4-5 gap
  queries per question), each gap query carrying its own gate calls, plus a
  second full synthesis and a second gap analysis per round.
- Latency, measured: the 3-question 2-round arm took roughly 50 minutes on a
  local 27B, against roughly 15 for the 1-round arm.
- Reading: this is the mechanism worth defaulting on, but its cost is
  multiplicative and the honest default depends on who pays. A per-run control
  (AC #2) matters more than the default itself, and the spend note (AC #3)
  should quote the measured multiplier rather than a general warning.

**Prerequisite now closed**: window-launched runs could not honour ANY of these
settings, because the window built the engine with no pipeline params at all
(task-17380). Per-run controls would have been inert there.

**Bounding caveat**: no measurement here ran with per-result summarization
working (task-17382 chain), so these numbers describe a pipeline synthesizing
from raw source text. A defaults decision that turns on multi-hop multiplies
the number of sources that each need summarizing, so re-measure the cost once
summarization completes.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Multi-hop is on by default for local research runs:
`DEFAULT_MAX_ITERATIONS = 2` resolved through `_configured_max_iterations()`,
which reads `[SearchSettings] research_max_iterations` and falls back to the
shipped value if the setting is missing, unreadable or non-positive. An explicit
`limits_json.max_iterations` still wins, so a caller asking for one pass gets
exactly one.

Fan-out is deliberately NOT enabled: it measured flat on the gate
(0.42 -> 0.38) and cannot change retrieval on this lane while task-17372
stands. The decision for each mechanism is recorded with its numbers in the
"Measured evidence" section above and in the eval baseline doc.

AC #3: the spend implication is documented on the config key itself -- the
place a user meets this setting -- quoting the measured multiplier (3 -> 12
search calls across three questions, roughly tripled wall-clock) rather than a
general warning. `/research` has no user-guide page to update; the Research
window's limits input already accepts a per-run override.

AC #4: 382 tests across the research, window, recorder, tools and pipeline
suites pass unchanged. The only single-pass assumptions left are in the
baseline recorder's tests, which pin that it passes `max_iterations`
EXPLICITLY (default 1) -- the property that keeps recorded baselines
reproducible byte-for-byte under a changed shipped default.

AC #2: the Research window now carries a rounds Select beside its policy
picker, defaulting to the ENGINE's own resolver (`_configured_max_iterations`)
rather than a second default that could drift from it, persisted through
`save_state`/`restore_state` like the academic lane toggle, and merged into the
launched run's `limits_json`. A typed `max_iterations=N` in the limits box still
wins, because that is the more specific statement of intent and is what the
engine treats as authoritative. Selecting a value states the trade-off in the
status line rather than leaving the spend implicit.

Three existing tests asserted the exact persisted state and `limits_json` and
were updated with the reason recorded in-line (AC #4's requirement): the
window's state gained `rounds`, and a launched run now carries `max_iterations`
whenever the limits text does not state one.

Modified: `tldw_chatbook/Research_Interop/local_research_engine.py`,
`tldw_chatbook/config.py`,
`Tests/Research/test_local_research_engine.py`,
`Docs/Development/research-report-eval-baseline.md`.
<!-- SECTION:NOTES:END -->
