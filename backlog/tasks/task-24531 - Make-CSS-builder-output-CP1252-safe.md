---
id: TASK-24531
title: Make CSS builder output CP1252 safe
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29'
updated_date: '2026-08-30 00:59'
labels:
  - css
  - windows
  - portability
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-29-windows-css-builder-output-portability-design.md
documentation:
  - Docs/superpowers/plans/2026-08-29-windows-css-builder-output-portability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the complete CSS generation entry point to run on strict Windows CP1252 standard output without weakening substantive build failures or changing generated stylesheets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every direct CSS-builder progress and completion message is encodable by a strict CP1252 output stream and remains readable
- [ ] #2 A full scratch-tree build succeeds through strict CP1252 output even when the checkout path contains a non-CP1252 character
- [ ] #3 The integration proof observes all four build phases, verifies every expected generated artifact and manifest exists, and proves distinctive source CSS reaches the generated bundle
- [ ] #4 Generated CSS ordering, hashing, manifest staleness semantics, source-race detection, missing-module failures, and output-preservation behavior remain unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Adapt the existing builder integration test to strict CP1252 output and a non-representable checkout path.
2. Replace direct builder output with ASCII-only numeric phase messages.
3. Verify manifest, staleness, fail-loud, and generated-artifact semantics.
4. Complete task evidence and self-review.

Detailed plan: Docs/superpowers/plans/2026-08-29-windows-css-builder-output-portability.md
ADR required: no
ADR path: N/A
Reason: portable build-script presentation only.
<!-- SECTION:PLAN:END -->
