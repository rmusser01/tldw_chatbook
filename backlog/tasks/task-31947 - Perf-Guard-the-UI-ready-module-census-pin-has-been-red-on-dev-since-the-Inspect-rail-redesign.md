---
id: TASK-31947
title: >-
  Perf Guard - the UI-ready module census pin has been red on dev since the
  Inspect rail redesign
status: Done
assignee:
  - '@claude'
created_date: '2026-09-07 08:25'
updated_date: '2026-09-07 15:59'
labels:
  - library
  - media-ux
  - test-debt
  - perf
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seen while landing media wave-5 PR F (2026-09-05): the 'UI latency guardrails' workflow's Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size has been red on dev since 7e904737c (Inspect rail Environment redesign) and 5f12507c1 (#2414 library reuse). It is not a Library media failure - the boot import set grew and nobody re-pinned or trimmed it, so every PR since reads a red Perf Guard it did not cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census pin is green on dev, either by trimming the boot imports back or by re-pinning with the reason recorded in the pin
- [x] #2 The change(s) that grew the boot import set are named in this task or in the pin's comment
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish the current state: run the pin locally on dev and read the Perf Guard workflow's conclusions on dev. 2. If green, name the commits that grew the boot import set and the ones that recovered the headroom; if red, trim or re-pin with the reason recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by evidence, no change in this PR. The pin (MAX_TLDW_MODULES_AT_UI_READY = 972, 'measured 969 macOS / 971 linux CI, +/-1 wobble headroom') is green again on dev: locally on 801216bb0 the census reports 971/972 modules (headroom 1, snapshot drift +32/-29), and the Perf Guard workflow succeeded on dev at 290b2d77f (2026-09-07 08:34 UTC) and 801216bb0 (09:05 UTC) after failing on 761416317, a9e13f4e3 and 68eef5bab (2026-09-06). AC#2: the growth came from 7e904737c (Inspect rail Environment redesign) and 5f12507c1 (#2414 library screen reuse); the headroom was recovered by the Console first-use deferrals 84695063c (defer the Environment stack until Inspect first opens), 86c77a4fb (defer first-use environment and vLLM imports) and eec6db562 (preserve startup headroom with first-use helpers). Residual risk: headroom is 1 module on Linux CI, so the next eager import flips it red again; the pin's own comment documents the refresh script (scripts/update_boot_budget_snapshots.py --only ui-ready) and the wobble.

Re-measured at PR M's final review (branch merged with dev 7ac39f4e8, macOS): 972/972 modules, headroom 0 (snapshot drift +33/-29); the pin still passes and dev's Perf Guard is green through 53813b3b9. Headroom is now zero on macOS, so the next module-scope import that reaches UI-ready flips this pin red again; the branch itself adds none (its only new module-scope imports are textual.*, already counted). The task stays Done on its ACs (green on dev, growth named); the zero headroom is the standing risk.
<!-- SECTION:NOTES:END -->
