---
id: TASK-38414
title: Restore layer direction for six lazy shutdown imports
status: To Do
created_date: 2026-11-01 20:12
labels:
- architecture
- bug
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the existing layer-direction contract for the six lazy shutdown imports below while preserving shutdown behavior. The exact test Tests/Architecture/test_layer_direction.py::test_no_new_upward_import reports the same six edges in current integration receipt r-b21b7ee0 and pinned clean-dev a9c1fa9d577f623befe0cc65f94d2a003ac6c8f1 receipt r-ad2461d6. app_shutdown.py, the test and allowlist are byte-identical across those sources; paired receipt hashes are stable. This is an observed inherited base defect, not a claim about which change introduced it. No allowlist workaround or unrelated layer redesign is authorized by this task.

tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.app_lifecycle (line 561)
tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.Scheduling.scheduler.handlers.dreams_handler (line 643)
tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.Scheduling.scheduler.handlers.dream_track_handler (line 661)
tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.Scheduling.scheduler.handlers.guardian_trend_handler (line 677)
tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.Tools.remote_session_registry (line 718)
tldw_chatbook.Utils.app_shutdown -> tldw_chatbook.Tools.remote_workspace_transport (line 721)

Receipts: C:/Users/GDesktop-1/.codex/pr3323-private-fixtures-20261101. The stale-allowlist check passes in r-cf89449e; the existing architecture rule correctly exposes these six violations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing layer-direction test passes for all six named shutdown edges without adding allowlist entries or weakening the rule.
- [ ] #2 Targeted shutdown controls preserve the six affected cleanup paths and their ownership/order requirements.
- [ ] #3 The repair records its architectural boundary decision and exact current-versus-base regression evidence before closure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
