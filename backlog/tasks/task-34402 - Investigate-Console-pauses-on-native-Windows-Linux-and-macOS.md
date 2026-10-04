---
id: TASK-34402
title: Investigate Console pauses on native Windows Linux and macOS
status: In Progress
created_date: 2026-10-04 19:17
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify the local causes of slow Console interaction and pre-provider send delays, and distinguish shared behavior from platform-specific amplification using native evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Retained native Windows Linux and macOS measurements identify the source revision and preserve storage safety checks.
- [ ] #2 Three durable captured sends complete with an immediate provider recorder on each platform, or platform failures are documented with evidence.
- [ ] #3 Report maps measured blocking work to source callers and distinguishes verified platform behavior from inference.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Observe guarded storage and native file calls without changing their behavior, using private profiles and an immediate provider recorder. 2. Run a bounded mounted Console probe on native Windows, Linux and macOS. 3. Correlate stage timings, event-loop delays and call counts with source paths and existing performance checks. 4. Retain results and document root causes and practical limits. ADR required: no. ADR path: N/A (existing ADR-126 applies). Reason: diagnostic measurements preserve runtime and security boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
