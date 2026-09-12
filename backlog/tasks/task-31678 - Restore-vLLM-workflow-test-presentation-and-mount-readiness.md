---
id: TASK-31678
title: Restore vLLM workflow test presentation and mount readiness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:19'
updated_date: '2026-09-05 18:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise vLLM click and recovery assertions against the actual styled, mounted controls rather than clipped or still-hydrating test views.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source-mode and forged numeric input tests deliver real clicks and preserve exact validation and draft assertions.
- [x] #2 The complete vLLM workflow file passes with scoped static checks and no relaxed readiness assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce click misses and compare bare App versus the existing consolidated CSS harness; root cause source buttons are clipped without their registered widget styles. 2. Use the standard styled harness, settle programmatic state layout before real clicks, and assert click delivery. 3. Investigate remaining mounted recovery/hydration races without weakening readiness checks. 4. Run the full workflow file and scoped static checks. ADR required: no. ADR path: N/A. Reason: restore test presentation and lifecycle preconditions using existing harness; no new runtime behavior or dependency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the standard consolidated CSS harness and real click delivery; waited for mounted vLLM view projection. Original six-case baseline had clipped source controls and a click before forged numeric state layout; a full run additionally exposed profile button visibility and pre-mount hydration. All eight targeted failures pass, and final complete vLLM workflow plus Console provider-apply files: 140 passed in 545.85s (/private/tmp/tldw-review-vllm-combined-verifiedpaths-20260905.xml). Exact draft, validation, lifecycle, readiness and event assertions retained. Scoped static and diff checks pass. No runtime or ADR change.
<!-- SECTION:NOTES:END -->
