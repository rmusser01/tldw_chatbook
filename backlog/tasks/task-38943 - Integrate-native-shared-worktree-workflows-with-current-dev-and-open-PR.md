---
id: TASK-38943
title: Integrate native shared-worktree workflows with current dev and open PR
status: Done
created_date: 2026-11-05 16:47
assignee:
- '@codex'
labels:
- console
- workflows
- integration
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3411
documentation:
- backlog/decisions/285-native-phased-agent-workflows.md
- Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md
updated_date: 2026-11-05 23:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish the reviewed native phased shared-worktree workflow feature and follow-up coroutine correction as a scoped pull request against current dev while preserving the shared checkout and unrelated changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PR branch contains the intended workflow feature and warning fix while retaining upstream changes and excluding unrelated radio/mobile/prompt-mixin work.
- [x] #2 Current upstream migrations, native runtime ownership and recovery remain valid; affected integration behavior passes targeted clean-source validation with documented limits.
- [x] #3 Independent integration review has no unresolved blocking findings, the PR is pushed/opened and attached, and the original shared checkout/index/unrelated file bytes remain preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR expected
ADR path: backlog/decisions/285-native-phased-agent-workflows.md
Reason: integrate the existing approved feature and amend its schema allocation/legacy compatibility mapping before changing those boundaries.
1. Freeze the shared checkout/index baseline; identify exact feature scope and current-upstream dependencies using read-only subagent audits.
2. Build a scoped synthetic feature tree using a temporary Git index; integrate with current origin/dev and resolve only intended changes in an ignored in-repository source directory.
3. Preserve upstream review-boundary migration25/runtime extractions, allocate workflow stages26–30 and use exact existing migration/recovery owners for recognized legacy compatibility, after ADR285 amendment.
4. Run targeted clean-source import/schema/recovery/native runtime and Console checks with source-root/hash receipts; fix actual integration regressions.
5. Obtain independent integration review and scoped fixes, create/push a new codex topic branch, open and attach the PR against dev.
6. Verify the original HEAD/index/protected file bytes are unchanged; document exact validation and CI limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Integrated the native phased shared-worktree feature and coroutine correction onto dev29592, preserving shipped review-boundary schema25, extracted runtime/UI owners and exact legacy recovery. ADR285 records the existing architecture and integration decisions; no new ADR was needed.
Targeted qualified checks passed: eight repository guards on unchanged 3,228 inputs; cold direct UI; four MCP controls; six legacy handoff cases; 34 hook-read controls; coroutine/workflow controls; qualified navigation; 31 style/fold cases; documented storage/runtime cohorts. All 9,000 Python files parse; 267 changed Python files have no introduced F821/F811/F823 diagnostics. Independent SPEC PASS / QUALITY APPROVE WITH EXPLICIT SCOPE LIMITS addresses both Important integration findings. Exact receipts: Docs/superpowers/qa/2026-11-05-native-workflow-dev-integration/README.md.
The original shared dev/HEAD, staged index and all 292 protected files are preserved. Full local suite and fresh live-provider replay were not run. Eighteen inherited lint findings, three upstream SQLite CI failures, mixed-source navigation qualification and retention-only process-exit controls remain explicit. Final reviewed corrections are pushed to PR3411, which is OPEN and ready for review at code head01d3ba76394c0a90d46013709eeb177ea1d28ee6. GitHub reports conflict-free; CI remains a separate merge gate. No merge or auto-merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR3411 is pushed and ready for review. Independent integration gates and targeted checks pass within documented source/platform limits. The original shared checkout/index and all292 protected files are preserved. CI and known baseline findings prevent a merge-ready claim; no merge or auto-merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
