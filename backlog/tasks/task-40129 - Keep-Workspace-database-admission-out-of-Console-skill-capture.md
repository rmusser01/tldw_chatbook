---
id: TASK-40129
title: Keep Workspace database admission out of Console skill capture
status: In Progress
created_date: 2026-11-11 21:06
assignee:
- '@codex'
priority: high
references:
- https://github.com/rmusser01/tldw_chatbook/issues/3347
updated_date: 2026-11-11 21:18
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Console sends when an installed skill requires a trust lookup during turn configuration capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configuration capture can read Workspace policy and classify an installed local skill without a cross-path admission refusal.
- [x] #2 The local skill trust decision and existing native read and storage admission checks still apply.
- [x] #3 A targeted file-backed regression fails at the reported admission boundary before the fix and passes afterward.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md and backlog/decisions/353-sqlite-store-kernel.md
Reason: Limit a caller's Workspace operation to Workspace reads within the existing finite capture and SQLite ownership contracts.

1. Reproduce the refusal with a real file-backed WorkspaceDB, installed local skill and SkillTrustService through checked Console configuration capture.
2. Limit the WorkspaceDB storage operation to its reads. Keep native connection retirement and individual admissions intact.
3. Run the red-to-green regression, neighboring Console capture checks, and scoped static review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The Workspace connection holds path-scoped storage admission. Checked Console configuration closes that operation after reading project bindings, before entering RAG profile and skill trust capture. Tool and persona policy keep the earlier finite read introduced on dev. The outer finite read still retires newly opened database handles. The existing trust service continues to decide whether each installed skill is available.

At dev ae9f9e8d21 the file-backed regression failed in SkillTrustService.status_for_skill with RecoveryRequired(operation_path_outside_scope). With the fix, the regression and two neighboring stock worker and adapter parity tests passed on Windows in 156.73s. The regression asserts that the selected Workspace policy survives and that the installed skill keeps its original trust block reason. Scope follows ADR-230 and ADR-353.

The three targeted tests also passed after rebase onto dev 7775edb567. In the stock Windows serve CLI, a browser Send in a disposable profile with an installed local skill reached the loopback llama.cpp-compatible test provider and completed. The send log recorded successful trace reservation and provider entry. SQLite contained exactly one user prompt and one matching assistant reply. The browser frame assertion did not confirm the painted reply. Merge qualification is pending.

Files: console_configuration_preparation.py, test_console_configuration_preparation.py and Docs/Architecture/console.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
