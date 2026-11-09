---
id: TASK-39603
title: Keep UI14 style checks within the existing job limit
status: Done
created_date: 2026-11-09 00:24
labels:
- ci
- performance
- tests
priority: high
references:
- Tests/UI/test_python_style_inventory.py
- Tests/UI/python_style_inventory.py
- Tests/private_profile.py
- .github/workflows/derived-artifacts.yml
updated_date: 2026-11-09 00:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR’s UI14 job was cancelled at its existing limit after pytest reported 293 passes. Remove unnecessary per-case process startup for pure AST checks while keeping the full gate and private filesystem coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retain all 293 original UI14 case IDs and selectors, all style assertions and parameters, and the original job, shard, concurrency and case limits.
- [x] #2 Only the 49 pure AST cases use the existing bootstrap profile; the CLI filesystem case still runs in its real private pytest child, with original source and guard semantics.
- [x] #3 Measure the original whole shard after the change, review the scoped repair and leave exact-head hosted qualification as a separate gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test fixture scope only; no product interface, test authority, gate, or runtime limit changes.

1. Keep the original cancelled job, original 293 case IDs and phase timings. Independently verify the pure AST/CLI boundary.
2. Replace the private launcher only on the 15 literal AST definitions (49 cases) with the existing bootstrap profile marker. UI’s autouse real App import then keeps its already bound sandbox; network and restoration guards remain. Keep the existing CLI case/decorator, every assertion, parameter and original UI14 selector exact.
3. Run the original UI14 shard once under the same case cap, compare every JUnit case ID and retain the actual CLI child receipt and style timing. Treat local results as local only; the new commit needs its own hosted wall-limit gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The original UI14 job reached its existing 20-minute wall limit just after 293 tests passed. Its 49 pure style-AST cases now use the existing bootstrap profile, avoiding separate pytest launches for literal parsing. The filesystem/CLI test retains its original private pytest child, and all assertion bodies, parameterizations, original selectors and limits are unchanged. All 293 original cases passed locally in 568.88 seconds. The same 50 style IDs on this machine measured 147.319 seconds before and 23.252 seconds after, with the scanner unchanged; the actual private CLI child passed and was retained. The successor must still meet its own hosted wall limit. No ADR is needed for this test-only fixture scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retained every original UI14 case and the original job limits; removed private child launches only for pure syntax checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
