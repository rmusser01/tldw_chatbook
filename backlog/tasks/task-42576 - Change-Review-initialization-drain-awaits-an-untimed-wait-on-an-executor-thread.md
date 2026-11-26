---
id: TASK-42576
title: "Change Review initialization drain awaits an untimed wait on an executor thread"
status: To Do
assignee: []
created_date: 2026-11-26 15:45
labels:
- follow-up
- console
- workspaces
- tests
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Workspaces/change_review_consent.py
- tldw_chatbook/Agents/legacy_hook_ownership.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-41061's review (round 27, 2026-11-26), on its branch at dev 6e41919880. `ConsoleRuntime._drain_change_review_initialization` (`Chat/console_runtime.py`, about line 2161) runs `asyncio.to_thread(service.wait_closed)`. That `ChangeReviewConsentService.wait_closed` (`Workspaces/change_review_consent.py`, about line 450) joins each initialization worker with `join_unless_retained`, with no timeout.

This is the class of wait TASK-41061 fixed for hook custody (`_EventCustodyRegistry.join_closed`, lesson "A summary followed by a hang is a thread interpreter exit joins" in `backlog/docs/lessons-testing-evidence.md`). Cancelling the awaiting task does not stop the executor thread. If a worker never returns, loop shutdown and interpreter exit join that thread forever: the app does not exit after Quit, and a pytest run hangs after its summary.

Constraint: `Workspaces/change_review_consent.py` is on another session's do-not-edit list in the follow-up briefs. Either the fix stays in `console_runtime.py` (for example, polling a timed join from there), or that file's owner signs off first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cancelled Change Review initialization drain whose worker never returns leaves no thread that loop shutdown or interpreter exit waits on
- [ ] #2 A test shows that red before the fix and green after, on a PR lane
- [ ] #3 A drain whose workers do return still joins them before it reports closed
<!-- AC:END -->
