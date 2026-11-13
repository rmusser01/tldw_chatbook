---
id: TASK-40724
title: Bound self-hosted regression lanes without dropping coverage
status: Done
created_date: 2026-11-13 19:39
assignee:
- '@codex'
updated_date: 2026-11-13 22:12
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first sustained local gate reaches the Console P0 and Dreams job time limits before completing their existing tests. Keep the required coverage while allowing the measured workloads to finish within bounded jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console P0 executes each existing sandboxed and admission-sensitive target exactly once across required matrix jobs.
- [x] #2 Console P0 and Dreams retain per-test deadlines and required verdicts with bounded 35-minute job budgets below the controller deadline.
- [x] #3 Focused workflow contracts pass and a new normal PR gate verifies the revised execution under the local worker pool.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/359-isolated-self-hosted-ci-runners.md
Reason: Partition existing required tests within the accepted runner boundary.

1. Preserve attempt4 evidence: Dreams and P0 admission pass, but sandboxed reaches35min; its three response-rule restore selectors alone took25m46s and completed successfully.
2. Move exactly those three selectors into a third required P0 matrix profile named restore. Keep the other sandboxed selectors, flags, profiles, required aggregation and35-minute limits unchanged; update all exact coverage/count contracts.
3. Run focused workflow contracts and an exact before/after target multiset assertion; independently review. Publish with the prepared TASK40725 ingest fix, then verify the normal full PR gate and measure it separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Console P0 now has three required matrix profiles: sandboxed, restore and admission-sensitive, with fail-fast disabled. The restore profile contains exactly the three existing response-rule restore selectors moved from sandboxed; their parameterizations and180-second deadlines are unchanged. Each profile and Dreams uses35minutes, below the controller45-minute deadline. Routing, required aggregation and all other jobs remain unchanged. Exact coverage, profile-selection, overlap/resolution and runner-count contracts updated; ADR-103 records48 eligible application jobs. Existing ADR-359 applies.

Evidence: attempt43487945840 hit20-minute P0/Dreams limits. Attempt43496035687 passed Dreams21m08s and P0 admission13m27s, but sandboxed reached35minutes after the restore selectors alone completed in25m46s. This motivated the third profile instead of another timeout increase. The selector multiset is identical before/after and every selector occurs once. The revised profile contract failed before the edit;839 focused checks passed under --noconftest, and the single harness-dependent storage-lease test passed normally. Ruff, small-file formatting and whitespace checks pass; independent review found no blockers. README and incident evidence updated. Normal PR run 43510161110 qualified all final lanes: sandboxed job132180869051 passed in6m45s, restore132180869141 in3m02s, admission132180869046 in13m18s, and Dreams132180869002 in23m52s. No lane exceeded its35-minute budget. Separate Node dependency and snapshot-settings failures keep the full gate unqualified; they do not invalidate this lane-layout verification. Failed attempt timings are stored separately under /tmp/tldw-ci-{third,fourth}-attempt-final-*.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
