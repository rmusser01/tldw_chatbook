---
id: TASK-38060
title: Stabilize Notes link journey navigation and paint readiness
status: Done
created_date: 2026-10-29 23:53
assignee:
- '@codex'
updated_date: 2026-10-30 00:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required Notes preview-link UI journey intermittently times out entering a later note or samples a missing painted link. Keep its semantic link and OS-handler checks reliable while preparatory navigation and frame publication settle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The note-link harness awaits canonical Notes navigation and a current painted link before acting, without adding arbitrary sleeps or weakening link outcome assertions.
- [x] #2 The failing preview journey and its shared picker sibling pass targeted verification, and the UTF-8 regression coverage remains green.
- [x] #3 The CI failure attribution, targeted evidence and test-only ADR decision are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Attribute PR3308's two red checks: the required artifact aggregate only reports UI Fast Lane 13's failure in the note-preview journey. Retain the CI receipt and reproduce the unchanged journey with a fresh private profile.
2. Keep the repair test-only: await the canonical Notes route before looking at retained tree rows, and wait for a current painted click offset. Preserve timeouts and every link target, browser, deleted-note and open_url assertion.
3. Run the preview journey and shared picker sibling, rerun the original UTF-8/local-provider checks, then scoped lint/format/diff and independent review. Record intermittent baseline behavior and exact evidence limits before re-arming the PR.
ADR required: no
ADR path: N/A
Reason: test-harness readiness correction; no production behavior or architectural boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3308's original run 40952949986 failed UI Fast Lane 13 at _open_note(STALE) waiting for the editor to arm. The derived-artifact aggregate propagated that UI failure; its failed step was Require successful UI fast lane. Auto-merge was disabled before synchronization, and the UTF-8 commit was rebased onto dev ed7481a30c.
The shared test helper now awaits screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES) for preparatory navigation and polls for the current compositor-derived click offset before the real Pilot mouse click. Production guards, timeouts and all target/browser/deleted-note/open_url assertions are preserved. Ruff normalized two existing long-expression blocks in this same helper.
Baseline evidence: an unchanged local preview journey failed at the later Gone painted-offset precondition; a diagnostic run with extra compositor observations passed. This establishes timing sensitivity, not the exact hosted interleaving. The repair makes the missing setup/action preconditions explicit rather than increasing waits or changing production behavior.
GREEN: the original preview journey and shared picker sibling passed together (2 passed). The refreshed provider selection passed all 104 cases. An added hostile-link check in that mixed invocation failed earlier at the Library-source startup deadline; the same negative-control UI test passed in its own fresh process (1 passed). Thus 107 distinct targeted cases passed across the provider and isolated UI contexts; the mixed-process startup failure remains explicitly recorded, not claimed green. Existing Pydantic, pytest-asyncio and audioop warnings were observed.
Scoped Ruff lint and format checks and git diff --check pass. Independent read-only review found no actionable issues and confirmed that canonical navigation preserves flush/admission guards and that actual painted mouse actions and semantic assertions remain intact. No full suite or live-provider result is claimed.
Updated Tests/UI/library_note_links_support.py and the incident entry in backlog/docs/lessons-testing-evidence.md. ADR required: no; this is a test-harness readiness repair. The original UTF-8 production fix remains under TASK-37600.
Latest-dev rebase onto ff815f8559 retained upstream lessons and revealed that dev independently allocated TASK-37650. Renumbered only this Notes task and its own lesson to TASK-38060 after scanning task IDs across all reachable refs (maximum 37960); upstream Hands-free records remain intact.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
