---
id: TASK-32917
title: Gate server audio diagnostics by effective administrator capability
status: In Progress
assignee: []
created_date: '2026-09-23 20:15'
updated_date: '2026-09-26 21:39'
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

2026-09-26 CI repair: PR Fast Lane and UI Fast Lane passed; Derived Artifacts failed on diagnostic inventory summary drift. Rebased onto dev 0053a0a40; all seven prior patches are identical. Local red checker reports only aggregate owner_files 596->597 and task_494_calls 7539->7541. The committed pin already contains 597 unique owner rows totaling 7541 TASK-494 calls; owner rows, diagnostics digests, and sink topology show no drift. Regenerate with the official script and verify that only these stale summary fields change.

Inventory red/green complete: official --write changed only owner_files and task_494_calls; a fresh --diff checker passes with 597 owners, 7541 TASK-494 calls and unchanged 14 sink files. The 22 focused audio/API tests passed (one known unrelated bootstrap test deselected); git diff --check passes. No Python source changed, so prior touched-source Bandit remains applicable. Commit the generated JSON plus plan/task records and require latest-head CI before merging.

2026-09-26 integration: dev c4225b5d3 merged tier-2 P3c and re-pinned the inventory. Rebase conflict was only our superseded two summary counters. Kept the newer upstream inventory unchanged, including its source_readiness owner addition. The first seven PR patches are identical by range-diff; the eighth now retains only plan/task evidence. Official inventory checker passes with 598 owners and 7542 TASK-494 calls; 22 focused audio/API tests pass (one known bootstrap test deselected). No Python source changed and diff check passes. Require fresh CI on this current-dev head before merge.

2026-09-26 late integration: required Derived Artifacts passed on 85426dd9f, but dev advanced to a3001f0ee with theme UX and MCP character authoring. Rebase completed cleanly and all eight prior PR patches are identical by range-diff. Official inventory checker passes with 598 owners and 7543 TASK-494 calls, retaining the upstream pin unchanged. The 22 focused audio/API tests pass with the same known bootstrap test deselected; git diff --check passes. No Python source changed and prior security qualification remains applicable. Fresh required CI must qualify the new head before merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Chatbook now consults the server effective audio-diagnostic capability before privileged connected-mode diagnostics, preserves passive health, supports older servers through endpoint authorization, and maps 403 to admin_required. PR #2822 carries the compatibility change.
<!-- SECTION:FINAL_SUMMARY:END -->
