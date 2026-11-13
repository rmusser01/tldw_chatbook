---
id: TASK-40725
title: Avoid importing the application before the ingest worker crash probe
status: In Progress
created_date: 2026-11-13 20:16
assignee:
- '@codex'
updated_date: 2026-11-13 20:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The real pool crash regression can exhaust its short observation budget importing its test-module target before the intended worker exit occurs under loaded local CI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The regression still crashes a real production-created spawn worker and verifies owning-generation failure and physical teardown using unchanged deadlines.
- [x] #2 The crash payload no longer imports the test module or application solely to call the standard-library exit function.
- [ ] #3 The exact regression and adjacent generation-sentinel tests pass, with an independent review and live required-lane rerun.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test-only removal of an unnecessary import boundary; production behavior and deadlines stay unchanged.

1. Inspect the failed CI log, spawn payload and production sentinel path; run the exact regression locally.
2. Submit os._exit directly with code17 instead of the test-module wrapper; remove the unused helper.
3. Run the exact and neighboring generation tests, prove a disabled-monitor negative control fails, review independently, then verify the normal PR gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed the test-module wrapper and submitted the picklable standard-library os._exit with exit code17 directly to the same production-created spawn pool. No production code, initializer, generation/sentinel ownership, assertions, isolation or deadlines changed. A wrapper target imports the large test module/application before the intended fault; the direct builtin removes that unrelated startup cost. The CI log alone cannot establish when its child reached the original wrapper, so live loaded verification remains pending.

Exact test before edit:1passed in3.02s locally. Exact plus adjacent current-generation sentinel test after edit:2passed in1.19s. A temporary external pytest plugin disabled only the production sentinel monitor; the changed exact regression then failed as expected at the original10-second observation budget, proving it still detects missing crash reporting. Independent review found no blockers and confirmed native builtin pickle round-trip. Ruff comparison found no new issues; pre-existing ASYNC110 remains unchanged, and lint passes with only that existing rule excluded. Whitespace check passed. Added the incident to lessons-testing-evidence.md. ADR not required for this test-only simplification. Normal PR lane rerun remains pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
