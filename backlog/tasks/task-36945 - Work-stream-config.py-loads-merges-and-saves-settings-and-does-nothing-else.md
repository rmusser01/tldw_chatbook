---
id: TASK-36945
title: 'Work stream: config.py loads, merges and saves settings and does nothing else'
status: Done
created_date: 2026-10-26 01:08
updated_date: '2026-11-20 14:55'
dependencies:
- TASK-36942.1
labels:
- architecture-review-2026-10-25
- config
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`config.py` is 12,258 lines: a 2,584-line inline TOML template, a 1,672-line loader, encryption, canvas and Console capture policy, and three lazily-created DB singletons that act as a global service locator. One consumer, the agent note tools, reads the bare global and fails unless something else created it first. Settings are read from 356 call sites, several at import time where hot reload can never reach them.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F9; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every subtask of this stream is Done
- [x] #2 config.py has a size ratchet row that this stream lowers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: all four subtasks (TASK-36945.1 to .4) are Done. AC #2: TASK-36942.1 (`2e0adfee71`) added `config.py`'s size-ratchet row at 12,258. TASK-36945.1 (`90f239f9c2`) lowered it from 12,285 to 9,668, and TASK-36945.2 (`b84fcd2b18`) from 9,668 to 8,979. It is 8,837 now. The guard files `test_layer_direction.py`, `test_module_size_ratchet.py`, `test_app_string_lookup_census.py`, `test_agent_runtime_import_footprint.py` and `test_screen_size_ratchet.py` passed serially on dev `d85e90bc9e` through the scratch wrapper: 314 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
