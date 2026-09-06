---
id: TASK-31919
title: Retain fork ownership through display-name persistence
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 02:00'
labels:
  - tests
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings display-name preparation mutates live identity and character projections,
then returns an asynchronous persistence plan without the fork-transition lease
used by ordinary roleplay refresh. A fork can observe those changes before their
durable outcome. Cleanup must retain ownership through the actual writer, even
when cancellation or an independent settings operation fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source rejects forks during live name/projection changes and until its detached persistence result is accepted or abandoned.
- [x] #2 No-op, stale-binding and failed preparation release ownership; other sessions remain independent.
- [x] #3 Settings coordination releases the exact plan after success, stale results, errors and cancellation, including cancellation before task startup; cancellation and sibling failure never release while a writer remains active.
- [x] #4 Real SQLite durability, deterministic interleaving regressions, targeted complete-file tests, scoped checks and independent review qualify the change without exempting other census routes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add `Tests/Chat/test_console_display_name_fork_lifetime.py` using the real Console store, persistence service and SQLite. Observe fork eligibility inside the real materialization call and after preparation/persistence but before acceptance. Require failure before production edits. Cover no-op/stale binding and preparation failure.
2. In `prepare_session_user_display_name_override_for_commit`, wrap existing mutation and plan construction in `_fork_source_transition(session.id)`. Construct a token-bearing plan, register one additional existing roleplay lease before the outer context exits, and document acceptance/abandonment responsibility. Keep revisions, validation, writers and projection behavior unchanged.
3. Exercise the actual `ConsoleSettingsDurabilityController` with real store/SQLite and event-controlled persistence: pre-start cancellation, cancellation during a blocked writer, stale return, failure, and sibling failure while writing. In `_coordinate_console_settings_submission`, create an explicit display-name task, attach exact-plan abandonment to task completion, then pass that task to the existing gather. Do not release in an outer gather finally or add a new drain loop; the existing serialized writer already drains cancellation.
4. Register the repaired direct route only after the existing owner-aware census recognizes its boundary. Run complete lifetime, settings publication/apply, first-send, fork and provider-apply UI files; report the remaining census classification separately. Run scoped Ruff/format, inspect diagnostic statement delta, obtain independent review, and update the draft checkpoint.

ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: Restore the existing fork snapshot lifetime using existing roleplay leases and asyncio task completion; no new ownership or persistence contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Protected live display-name/projection updates with the existing transition and extended it using an existing roleplay lease until result acceptance or abandonment. The coordinator now attaches exact-plan cleanup to display-name task completion, covering pre-start cancellation and stale/error exits without releasing a running writer on sibling failure. Original real-SQLite live-gap regression: 1 failed/3 passed. Coordinator cleanup regression: 3 failed/5 passed. All 10 lifetime tests now pass, including event-controlled worker cancellation and sibling failure, other-session isolation, and durable reopen. Six complete affected files: 331 passed in 161.27s, three existing warnings; /private/tmp/tldw-name-lifetime-final.xml. Review then corrected fixture handle cleanup using same-file quiescence and zero registered-handle assertion; complete lifetime+census rerun: 36 passed/1 known classification failure in 18.49s; /private/tmp/tldw-name-lifetime-cleanup-census.xml. Six other routes remain unclassified; no scanner or exemption changes. Scoped static checks and independent re-review pass. All 81 store and 10 settings-controller diagnostics remain unchanged. Added the task-completion cleanup lesson and updated the checkpoint. ADR check: existing ADR-092, no new ADR required.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31800 was renumbered to TASK-31919 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
