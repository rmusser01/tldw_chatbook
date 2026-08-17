---
id: TASK-17370
title: Measure the relevance gate with sub-question decomposition enabled
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-17 07:30'
updated_date: '2026-08-17 07:35'
labels:
  - research
  - web-tools
  - benchmarks
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every recorded gate number — the 0.29 repositories baseline and the 0.42 source-type-note re-measurement — was produced with both decomposition mechanisms switched off: the baseline recorder hard-codes `subquery=False, max_queries=1` as a spend bound, and it launches runs without `limits_json`, so gap-driven replanning defaults to a single iteration. The relevance gate prompt takes the generated sub-questions as a required placeholder, so those runs asked the gate to judge every result against one broad question with an empty sub-question list. A repository record has no narrower facet to be relevant to under those conditions, which makes the recorded "genuine residual" for non-paper evidence unfalsifiable. Make the spend bound a caller choice and re-measure so the residual is either confirmed or explained.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The baseline recorder exposes sub-query fan-out and multi-hop iteration count as flags instead of hard-coded constants, with the current spend-bounded values as defaults so existing recorded baselines stay reproducible byte-for-byte.
- [ ] #2 The recorder prints the decomposition settings in its run header and carries them into the emitted aggregate JSON, so a recorded result can never be read without knowing whether decomposition was on.
- [ ] #3 Tests pin that the default invocation still assembles single-query, single-iteration parameters, and that the flags reach the pipeline params and the run's limits.
- [ ] #4 The repositories lane is re-measured live with decomposition enabled against the same question set, engine and endpoint as the recorded 0.42 run.
- [ ] #5 The baseline doc records the decomposition off/on comparison and the amended reading of the non-paper residual, including the case where decomposition closes little of the gap.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD the recorder's decomposition controls: default invocation must assemble the same single-query, single-iteration params the recorded baselines used (even when config says search_enable_subquery is on), and the flags must reach the pipeline params and the run's limits_json
2. Derive sub-query generation from the total-query cap so the two can never be set contradictorily, and pass max_iterations explicitly even at its default so a run states what it measured
3. Expose the phase-2 wall clock as a flag -- the configured 240s is calibrated for one-query runs and would truncate a fan-out gate loop mid-run, measuring the deadline instead of the gate
4. Stamp the decomposition settings on the emitted aggregate (outside aggregate_metrics, whose Dict[str, float] contract stands)
5. Live re-measure the repositories lane on the same question set/engine/endpoint as the recorded 0.42: fan-out only, then fan-out plus multi-hop
6. Record the comparison and the amended residual reading in the baseline doc
ADR required: no -- measurement instrumentation only; no change to shipped run behaviour (that is task-17371)
<!-- SECTION:PLAN:END -->
