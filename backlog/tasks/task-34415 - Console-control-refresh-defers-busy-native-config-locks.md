---
id: TASK-34415
title: Console control refresh defers busy native config locks
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 15:47'
updated_date: '2026-10-07 03:03'
labels:
  - console
  - performance
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep Console control refresh responsive when another thread holds the existing config rebuild or file lock, without weakening configuration source checks, recovery admission, or current-state rendering. Owner approved the bounded deferral after forwarding observations identified native rebuild contention; full activation latency qualification remains separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both shared refresh callers return without rendering or blocking on either held native config lock, coalesce one installed retry, and release every partial acquisition.
- [x] #2 The installed retry reads current config and selection after release; teardown, maintenance refusal, source drift, callback error, cleanup error, and recovery ownership retain their existing behavior.
- [x] #3 Deterministic real-lock RED and GREEN, affected lifetime and lock-order regressions, lint, formatting, artifact guards, and independent scoped review are retained without claiming full native or 50ms qualification.
- [x] #4 The PR MCP mount smoke observes completed rail replacement before asserting or shutting down, retaining its existing ten-second bound and server/canvas assertions; delayed-mount RED/GREEN is retained without changing production behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md. Reason: routine local scheduling repair preserves source/admission, native lock ordering, lifetime and activation contracts; no new config API, cache, policy or schema.
1. Add a deterministic real-thread REBUILD/FILE contention regression to Tests/Backup_Recovery/test_console_config_sync_lifetime.py for both existing refresh callers. Keep native locks unchanged; a finite watchdog frees only the test holder on baseline failure. Assert return while still held, no reads/publication, one retry and released partial acquisition; retain the expected baseline RED.
2. Move the existing shared checked owner into UI/Console_Modules/config_sync.py, retaining the ChatScreen delegator. With stdlib ExitStack, acquire the same native locks using acquire(blocking=False) in REBUILD then FILE order; if busy defer and return False, otherwise enter unchanged operation(config). Retain UI/body/cleanup exception classification verbatim. No await while held and no screen budget increase.
3. Release the real holder and replay the installed callback after an actual config save and changed selection. Assert fresh state, checked lifetime retirement, native pause/source-drift refusal and teardown fences; rerun existing Console config lifetime, native lock-order and sync-maintenance files plus mounted activation/reduced-motion coverage.
4. Run scoped Ruff/format, whitespace, eleven artifact guards and paired screen ratchet measurements. Retain logs and accurately separate lock responsiveness from the still-open full 50ms/native/Windows/participant qualification. Request independent scoped review, fix scoped findings, commit and create a PR against dev; do not auto-merge or treat skipped bots as reviews.
5. Owner authorized latest-dev rebase, scoped corrections and merge. Preserve both appended lessons in the documentation conflict. The rebased screen exceeds the unchanged 25204 line ceiling by 27; remove only the two control helpers with no code/test/script callers (_get_shell_bar and _collapse_console_hidden_control_bar), then inline the exact query/QueryError behavior of the one-call-site _get_compact_model_bar. Remove the unused private _summary_row_value import (the module-owned implementation and its callers remain). Lower only the measured method ceiling. Verify the two ratchets and five existing mounted compact-control/model-profile controls, then rerun the affected lifetime/activation selection and artifact guards. Use a fresh independent review in place of credit-blocked Qodo, explicitly approved for PR3034; wait for current-head CI and verify the protected merge. No new behavior or ADR.
6. Exact-head CI found the unchanged MCP mount smoke ending during a rail replacement. Preserve the ten-second bound and all server/canvas assertions; wait for the rail's pending replacement and expected rows, prove the exact SelectOverlay failure with a delayed second Select composition, then rerun that probe and neighboring mount cases. Retain failed CI and local receipts, obtain fresh scoped review, and wait for new-head CI. Test-only correction; no production or ADR change.
Task allocation: CLI chose local 34413, which exists on another fetched branch; remote and registered-worktree prefix scans found max34414. Renumbered only this unpublished file to unused34415 before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded nonblocking entry using the exact native REBUILD then FILE RLocks retained across unchanged operation(config), with the installed coalesced fresh-state retry. Both callers invoke the module-owned helper directly; redundant screen wrapper removed. The stale worker fixture supplies the incumbent acknowledgement callback without changing assertions; exact base-owner counterfactual confirms the prior failure. Deterministic real-lock RED six cases then GREEN six. Existing ADR126/120 source/admission, cleanup and lifetime contracts are preserved; no new ADR.

Latest-dev replay preserves both appended lessons. Remove two unused private control helpers and inline the single-use compact lookup with its exact QueryError fallback; no behavior/layout change. Combined screen 25204/756, both size ratchets green, no ceiling raised. Final affected selection 61 passed in 235.40s, no pytest warnings; six additional unchanged compact-control/model-profile/provider-mirror cases pass in separate existing bootstrap-profile processes. All eleven artifact guards pass. Four Python paths format clean; helper and two tests Ruff clean. Latest-dev-relative screen lint 192→189, no new diagnostics; removals belong to deleted code. Fresh independent review found no code regression; its historical-count documentation finding was corrected. Initial overlay timeout and inherited Splash/housekeeping warnings are retained, never suppressed. QA receipt Docs/QA/task-34415/README.md; native-lock/TOCTOU lesson in backlog/docs/lessons-console-wiring.md. Owner explicitly accepted fresh independent review instead of credit-blocked Qodo for PR3034 and authorized protected merge after current-head CI. Parent 50ms/native/Windows/participant/lifetime qualification remains open and unwaived. This bounded task stays In Progress pending publication and current-head PR CI closeout.

Exact-head Derived37562090298 failed the unchanged MCP mount smoke after loading cleared before rail replacement completion. A 200ms second-Select composition hold reproduced its exact SelectOverlay shutdown error; the test-only bounded readiness correction passed that same hold and three original mount/compact-viewport cases (3 in7.35s, no pytest warnings). Ten-second bound, assertions and all production/script/package/workflow inputs are unchanged. Five inherited file Ruff findings unchanged, format clean. Failed CI/admission warnings and ordinary-probe housekeeping warnings are retained in QA receipts; no qualification waiver. Fresh scoped review and new-head CI remain required; task stays In Progress.
<!-- SECTION:NOTES:END -->
