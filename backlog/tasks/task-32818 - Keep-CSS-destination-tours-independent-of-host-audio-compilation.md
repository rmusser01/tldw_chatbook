---
id: TASK-32818
title: Keep CSS destination tours independent of host audio compilation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 20:02'
updated_date: '2026-09-18 20:07'
labels:
  - testing
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32816 validation reached all fifteen destinations and passed the CSS source assertion twice, but the private test process timed out during executor shutdown. Child thread dumps identify Meetings preparation compiling the macOS audio helper. A stylesheet budget test should finish without depending on host audio tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both existing CSS destination-tour tests complete with all fifteen route and body assertions and the original source limits.
- [x] #2 Only the external system-audio probe is isolated in the shared CSS test app; the real Meetings screen, owner, and preparation flow remain exercised.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the two timeout receipts and the child executor/compiler stack. 2. Give the shared CSS-tour app a deterministic unavailable system-audio probe before mount. 3. Run both unchanged source-budget journeys serially with their original 180-second limit, inspect lint and the diff, and document the narrower evidence boundary. ADR required: no. ADR path: N/A. Reason: Test-only dependency isolation; no production, platform, storage, or audio contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Only the shared CSS-tour app instance tap probe returns unavailable; the real Meetings owner, prepare worker, fifteen real screen/body assertions and original source thresholds remain. Two unchanged tests now pass serially and terminate under their original 180-second limit (private teardown 1.04s/1.62s). The original 180-second failures, diagnostic plugin and honestly labelled observed stack excerpt are retained in the Appearance QA receipt. Ruff/format and independent read-only review pass. No production audio, worker, timeout or source-budget contract changed. ADR not required: test-only external dependency isolation.
<!-- SECTION:NOTES:END -->
