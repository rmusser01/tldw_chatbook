---
id: TASK-32917
title: Gate server audio diagnostics by effective administrator capability
status: In Progress
assignee: []
created_date: '2026-09-23 20:15'
updated_date: '2026-09-26 01:02'
labels: []
dependencies: []
references:
  - backlog/decisions/178-server-audio-diagnostic-admin-boundary.md
  - 'https://github.com/rmusser01/tldw_server/pull/2968'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Match Chatbook connected-mode audio diagnostics to the server administrator authorization contract so ordinary users see an explicit admin-required outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Passive STT health remains usable for ordinary connected users.
- [x] #2 A server 403 for either diagnostic becomes an explicit admin-required result.
- [x] #3 Targeted connected-mode tests cover admin and nonadmin outcomes.
- [x] #4 Warm STT health and streaming diagnostics honor the server effective administrator capability before dispatch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-09-23-server-audio-diagnostic-admin-parity.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_chatbook/pull/2822. Targeted service/scope tests: 15 passed, 1 existing from_config bootstrap failure excluded; Ruff/compileall/diff checks passed.

Qodo wildcard review resolved by server can_run_audio_diagnostics capability. Revised targeted tests: 22 passed, 1 unrelated from_config bootstrap test deselected; Ruff/format/compileall/diff checks passed on touched small files. Older servers without this capability defer to protected diagnostic endpoints.

2026-09-25: rebased PR #2822 onto Chatbook dev 3b11bf972 after OmniVoice merge. All three commits unchanged by range-diff; 22 focused tests passed, one pre-existing from_config bootstrap test deselected; diff check clean. Final required derived-artifact CI must rerun on this head.

2026-09-25: prior PR Fast Lane failed in two unchanged MCP Workbench UI tests; both passed locally and their files were identical across old/new dev and PR. A failed-job rerun was queued when dev advanced to 461668df0 (ingestion-only merge). Rebased onto that dev; all four prior commits unchanged by range-diff, 22 focused audio/API tests passed with one pre-existing from_config bootstrap test deselected, diff check clean. New required CI will decide merge readiness.

2026-09-25: rebased PR #2822 onto Chatbook dev 50f6096db after the tier-2 P2 Notes/Library/Media merge. All five prior PR commits are patch-identical by range-diff. The 22 focused audio/API tests passed (one pre-existing from_config bootstrap failure deselected), and git diff --check is clean. Await the required latest-head derived-artifact gate.

2026-09-25: rebased PR #2822 onto Chatbook dev 1b61ee2c0 after the off-by-default Tamagotchi merge. All six prior PR commits are patch-identical by range-diff; 22 focused tests passed with one pre-existing from_config bootstrap failure deselected; git diff --check is clean. Latest-head required CI must rerun.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatbook now consults the server effective audio-diagnostic capability before privileged connected-mode diagnostics, preserves passive health, supports older servers through endpoint authorization, and maps 403 to admin_required. PR #2822 carries the compatibility change.
<!-- SECTION:FINAL_SUMMARY:END -->
