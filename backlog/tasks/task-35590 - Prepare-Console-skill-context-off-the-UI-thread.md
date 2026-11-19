---
id: TASK-35590
title: Prepare Console skill context off the UI thread
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 04:16'
updated_date: '2026-10-19 04:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console discovery responsive during local skill storage admission and catalog reads while retaining current permission and ownership checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stock Console context loading leaves the event loop responsive and returns the original checked catalog.
- [x] #2 Cancellation drains native work; stale sources reject results; custom callbacks retain their original execution path.
- [x] #3 Targeted regression tests and native Windows before-after profiling verify the improvement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; applies existing ADR-126/230 finite native ownership and ADR-231 authority separation. Reason: bounded scheduling correction retaining existing service and runtime contracts. 1. Finish the eight baseline failures in TASK-35589. 2. Capture a focused native baseline and hold-original regression. 3. Qualify stock discovery sources, execute the original decorated context call using existing finite preparation reads, preserve fallback affinity and current-owner publication. 4. Run focused safety and UI tests, reprofile the same workload, review and document results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Console discovery now recaptures the stock source after cold trust preparation and runs the original decorated context read through the existing finite preparation worker, tracked by runtime through cancellation/shutdown. Current-source/view checks gate publication; custom policy, context body, catalog, disabled-loader and trust-status callbacks retain direct affinity. Native controls hold the original callback after actual storage admission and verify loop progress, repeated cancellation custody, source replacement and custom fallback. Final 38 targeted tests pass; TASK-35589 repairs the original eight failures with 37 passing tests. Focused six-read Windows profile: median loop overshoot 142.08ms to 14.46ms; original context-admission main-thread samples 60 to 0. Underlying median read duration 151.79ms to 189.14ms: responsiveness improvement, not faster storage. One full tour worst typing 1167.21ms to 30.34ms, while idle retains a 144.84ms token-estimation/custom-tokenizer stall. No network attempts or real-profile refusals. Independent review approved final fixes. New tests lint/format clean; product retains same four baseline E721 diagnostics, changed ranges formatted, compilation/diff checks pass. Architecture documentation and evidence/limitations in Docs/Development/evidence/windows-performance-2026-10-18/CONSOLE_SKILL_RESULTS.md. ADR required: no; existing ADR-126/230/231 contracts retained. No full-suite claim; 14 other prompt failures independently reproduce with original fixture. Existing tiktoken edits untouched.
<!-- SECTION:NOTES:END -->
