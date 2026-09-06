---
id: TASK-31821
title: Close remaining inventory UI fixture-owned database resources
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 06:10'
updated_date: '2026-09-06 06:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native per-test attribution finds retained database handles in passing attachment, prompt, skill and thinking inventory tests; qualify exact ownership teardown without mutating production or relaxing resource thresholds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every repaired complete UI file passes with no retained test SQLite descriptors after its exact owner teardown.
- [x] #2 Only explicitly imported module-local builder products and tmp_path database owners are finalized; foreign/global owners remain untouched.
- [x] #3 Behavioral assertions, cleanup controls, scoped static checks and independent review remain qualified.
- [x] #4 The complete thinking integration file retains no tmp_path SQLite handles under Darwin F_GETPATH attribution.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only use of existing lifecycle owners. 1. Preserve native descriptor baselines (59 attachment/prompt tests pass but retain own ChaChaNotes and auxiliary DBs; skill cohort retains additional handles). 2. Explicitly import TASK31818 exact module-local test-app and tmp_path DB cleanup only into affected files after confirming their builder ownership. 3. Run complete changed UI files with native attribution; do not waive Stop geometry or runtime failures. 4. Run all shared cleanup importers and fault controls, static checks and independent review before closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Explicitly imported existing exact owner fixtures into attachment/prompt/skill files and the controller/tmp_path cleanup into thinking integration. All identified own SQLite retention is closed under validated Darwin F_GETPATH; the initial readlink(/dev/fd) observer was invalid on macOS and its zero was rejected. Root final combined cohort: 93 passed / 1 failed in 100.36s, three dependency warnings, no retained SQLite lines or FD-growth warning (/private/tmp/tldw-31821-resource-final.xml and .log). Five other repaired UI owner files pass74; thinking/regenerate pass25 separately. The sole combined failure is unchanged Stop clipping, tracked TASK31822; retain In Progress until full-file acceptance. Scoped lint/diff checks pass; changed-region formatting is preserved. No production/GC/threshold/global-owner changes. ADR required: no, existing lifecycle APIs.
<!-- SECTION:NOTES:END -->
