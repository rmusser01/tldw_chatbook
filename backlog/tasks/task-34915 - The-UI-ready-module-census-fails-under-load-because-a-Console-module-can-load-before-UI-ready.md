---
id: TASK-34915
title: The UI-ready module census fails under load because a Console module can load
  before UI ready
status: Done
created_date: 2026-10-12 20:42
updated_date: '2026-10-20 14:39'
labels:
- tests
- performance
- flaky
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The UI-ready module census pins how many app modules are loaded when the app reports its UI ready, with no headroom (1033). On a loaded machine it counts 1034 about one run in four on clean dev (2 of 8 measured on 2026-10-12). The extra module is the Console config sync, which the Console control-bar sync imports lazily. That sync is queued to run after the Console screen's first refresh, and the app marks the UI ready right after it finishes pushing that screen, so which happens first depends on scheduling. The module arrived with TASK-34415, after the pin was set. The census runs in the performance workflow (Perf Guard), not the required check. CI has shown it once: TASK-34789's description records the Perf Guard census at 1034 against 1033; none of the required-check logs saved for TASK-34789 show it, as they would not. A timing-dependent count makes the pin unreliable as a ratchet. Found while investigating flaky tests for TASK-34789.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census gives the same count on every run of unchanged code, loaded machine or not
- [x] #2 The pin still fails when a change adds a module to the path before UI ready
- [x] #3 The decision of whether the Console config sync belongs before UI ready is recorded with its measured import cost
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

2026-10-20 UTC. No code change was needed: the race this task describes went away on 2026-10-16, when PR #3023 (merge 14e13bb82c, commit 3e618d7119) made the Console compose import config_sync. ContextSpendMixin._build_console_cost_state imports it while building the cost chip, so the module is resident at UI ready on every run instead of only when the control-bar sync wins the race.

AC #1: 15 of 15 local warm-boot censuses on 2026-10-19 gave the same count (1030 on dev then), including runs under a 4-way CPU load. In CI, the Perf Guard census printed 1031/1031 on all 11 runs from #3167's merge (2026-10-20 04:10Z) to 13:31Z: 4 push runs on dev and 7 pull_request runs on 4 branches. #3167 added one module, UI.Navigation.surface_swap_guard, which took the count from 1030 to the pin.

AC #2: the pin has no headroom (1031 of 1031). Negative control on dev 1bbc145ffc: an empty module imported from config_sync made the census 1032 and failed the test ("1032 tldw_chatbook modules resident at _ui_ready (ratchet limit 1031)"). Reverted, the census passes again.

AC #3: decision is to keep config_sync resident at UI ready. Its marginal import cost with the app loaded is 0.046 ms median. The cost chip needs it at compose, so deferring it would only move the import into the first sync. The pin stays at 1031 (count == pin), so it needs no change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

The census is deterministic since PR #3023 made the Console compose load config_sync; CI showed 1031/1031 on 11 of 11 runs after #3167, a negative control fails at 1032, and config_sync stays resident (0.046 ms).
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
