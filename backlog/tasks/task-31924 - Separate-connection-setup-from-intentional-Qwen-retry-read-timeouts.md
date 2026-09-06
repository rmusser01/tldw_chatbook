---
id: TASK-31924
title: Separate connection setup from intentional Qwen retry read timeouts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:08'
updated_date: '2026-09-06 05:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the real-HTTP retry and response cleanup regression deterministic when loopback connection setup exceeds its intentionally short body-read deadline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real timeout-status-connection retry sequence preserves exact attempt counts and response closure while connection setup cannot consume the intended read-timeout attempts.
- [x] #2 Intentional body-read timeouts remain 50ms and a negative control detects raising them or losing the intended timeout phase.
- [x] #3 Complete Qwen test file and lint pass with no production code, retry budget or broad timeout changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only transport fixture stabilization. 1. Preserve reproduced full-file and instrumented RED: first ConnectTimeout before scripted action, then two real read stalls exhaust three attempts. 2. Add an opt-in connect/read split to the real resource-tracking wrapper only for the combined retry test; keep scalar/read timeout0.05 and real Requests/HTTP server/errors. Preserve existing three-value helper return and other callers. 3. Assert the actual read-timeout phase and unchanged deadline, plus existing10 total attempts and resource closure; use a bounded negative control that raises the read deadline or bypasses intended retries. 4. Run complete Qwen file, repeat exact case, lint/format and independent review, then record and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept real requests transport and retry policy, giving only the combined retry test a1s connection allowance while preserving its50ms read deadline. Records actual retry errors and requires both stalled attempts to be ReadTimeoutError; budgets/status/response-close assertions unchanged. Reproduced ConnectTimeout consuming a scripted attempt; widened-read negative control fails phase assertion. Ten fresh exactcase runs pass. Independent completefile150passed8.86s with2 dependency warnings; full Ruff/format/diff checks and independent review pass. Only test file plus formatter-only indentation cleanup; production unchanged. Checkpoint and lesson record evidence. ADR required:no, test-only scheduling stabilization.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31813 was renumbered to TASK-31924 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
