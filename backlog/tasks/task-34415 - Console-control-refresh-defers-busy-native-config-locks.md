---
id: TASK-34415
title: Console control refresh defers busy native config locks
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 15:47'
updated_date: '2026-10-06 16:10'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md. Reason: routine local scheduling repair preserves source/admission, native lock ordering, lifetime and activation contracts; no new config API, cache, policy or schema.
1. Add a deterministic real-thread REBUILD/FILE contention regression to Tests/Backup_Recovery/test_console_config_sync_lifetime.py for both existing refresh callers. Keep native locks unchanged; a finite watchdog frees only the test holder on baseline failure. Assert return while still held, no reads/publication, one retry and released partial acquisition; retain the expected baseline RED.
2. Move the existing shared checked owner into UI/Console_Modules/config_sync.py, retaining the ChatScreen delegator. With stdlib ExitStack, acquire the same native locks using acquire(blocking=False) in REBUILD then FILE order; if busy defer and return False, otherwise enter unchanged operation(config). Retain UI/body/cleanup exception classification verbatim. No await while held and no screen budget increase.
3. Release the real holder and replay the installed callback after an actual config save and changed selection. Assert fresh state, checked lifetime retirement, native pause/source-drift refusal and teardown fences; rerun existing Console config lifetime, native lock-order and sync-maintenance files plus mounted activation/reduced-motion coverage.
4. Run scoped Ruff/format, whitespace, eleven artifact guards and paired screen ratchet measurements. Retain logs and accurately separate lock responsiveness from the still-open full 50ms/native/Windows/participant qualification. Request independent scoped review, fix scoped findings, commit and create a PR against dev; do not auto-merge or treat skipped bots as reviews.
Task allocation: CLI chose local 34413, which exists on another fetched branch; remote and registered-worktree prefix scans found max34414. Renumbered only this unpublished file to unused34415 before implementation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented bounded nonblocking entry using the exact native REBUILD then FILE RLocks retained across unchanged operation(config), with the installed coalesced fresh-state retry. Both callers invoke the module-owned helper directly; redundant screen wrapper removed. Screen 25231/760 becomes 25204/759; published line ceiling lowered 25218 to 25204, method budget unchanged 759. The stale worker fixture now supplies the incumbent acknowledgement callback without changing assertions; exact base-owner counterfactual confirms the prior failure. Deterministic RED six cases then GREEN six; final affected run 61 passed in 113.84s, no pytest warnings. All eleven final artifact guards pass. Four Python paths format clean; helper and two tests Ruff clean; full-screen 192 inherited diagnostics unchanged after normalized shifted positions. Independent scoped review found no Critical/Important/Minor findings. Earlier Splash and pytest housekeeping warnings retained, never suppressed. QA receipt Docs/QA/task-34415/README.md; native-lock/TOCTOU lesson recorded in backlog/docs/lessons-console-wiring.md. ADR required: no; existing ADR126/120 contracts preserved. No full 50ms/native/Windows/participant/lifetime qualification or Qodo waiver; parent tasks remain In Progress. This bounded task stays In Progress pending publication and current-head PR CI closeout.
<!-- SECTION:NOTES:END -->
