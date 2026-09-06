---
id: TASK-31821
title: Close remaining inventory UI fixture-owned database resources
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 06:10'
updated_date: '2026-09-06 15:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native per-test attribution finds retained database handles in passing attachment, prompt, skill and thinking inventory tests; qualify exact ownership teardown without mutating production or relaxing resource thresholds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every repaired complete UI file passes with no retained test SQLite descriptors after its exact owner teardown.
- [x] #2 Only explicitly imported module-local builder products and tmp_path database owners are finalized; foreign/global owners remain untouched.
- [x] #3 Behavioral assertions, cleanup controls, scoped static checks and independent review remain qualified.
- [x] #4 The complete thinking integration file retains no tmp_path SQLite handles under Darwin F_GETPATH attribution.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only use of existing lifecycle owners. 1. Preserve native descriptor baselines (59 attachment/prompt tests pass but retain own ChaChaNotes and auxiliary DBs; skill cohort retains additional handles). 2. Explicitly import TASK31818 exact module-local test-app and tmp_path DB cleanup only into affected files after confirming their builder ownership. 3. Run complete changed UI files with native attribution; do not waive Stop geometry or runtime failures. 4. Run all shared cleanup importers and fault controls, static checks and independent review before closure. 5. The approved-fix adjacent-file runs additionally attribute owned auxiliary handles to UAT, direct real-app screen reuse, roleplay resume, and dictation harness fixtures. Apply existing exact-owner cleanup to these test factories without changing production or finalizing foreign/global owners; verify the entire affected files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed exact fixture ownership for the remaining inventory and approved-repair adjacent files. Existing controller/tmp_path/database ExitStack APIs remain unchanged; explicit module-local builder adapters capture dictation helpers and direct real-app reuse factories, with only each real app notification DB added to its exact auxiliary callbacks. No production/global-owner/GC/resource-threshold changes. Initial own native descriptor baselines are preserved. Final original cohort that was93passed1Stopfailure now passes94 in105.93s; final seven adjacent dictation/reuse/command files253passed374.99s; root owner-fault controls plus handoff/UAT/attachment/width56passed71.95s. All three native Darwin F_GETPATH runs have zero retained SQLite lines and no descriptor-growth warning. Logs /private/tmp/tldw-31821-original-cohort-complete, tldw-31821-dictation-reuse-resources, tldw-approved-integration-final (.xml/.log). Prior shared fixture importer/control576pass evidence remains; shared fixture internals did not change. Independent ownership review and changed-file lint/region-format/diff checks pass. ADR required:no; existing test lifecycle APIs. Behavioral assertions remain unchanged except strengthened approved regressions.
<!-- SECTION:NOTES:END -->
