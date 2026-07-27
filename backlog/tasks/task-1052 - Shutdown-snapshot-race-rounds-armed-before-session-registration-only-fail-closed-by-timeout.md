---
id: TASK-1052
title: 'Shutdown-snapshot race: rounds armed before session registration only fail closed by timeout'
status: To Do
assignee: []
created_date: '2026-07-27 14:32'
labels:
  - console
  - approvals
  - shutdown
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shutdown's per-session cancel fanout, and `close_session`'s deny path, both walk `_active_stream_tasks` to find which sessions have a live round to cancel/deny. A confirm or approval round that gets armed BEFORE its owning session id has been registered into `_active_stream_tasks` is invisible to both of those paths at the moment they run: `shutdown()` will not include it in the fanout, and `close_session` will not deny it either -- the round is simply not iterated.

This does not auto-approve or otherwise mis-decide anything -- the round still fails closed via its own `_MCP_APPROVAL_POLL_SECONDS`/deadline timeout loop (up to the full ~120s configured timeout), same as any other unresolved round. The gap is purely one of promptness: a round caught in this window sits until its own timeout elapses instead of observing the shutdown/close signal immediately like every other in-flight round does.

`Tests/UI/test_skill_install_concurrent_confirms.py::test_bare_shutdown_flag_alone_does_not_deny_a_real_session_round` currently pins this behavior as evidence (a bare `_shutdown_requested` flag with no corresponding `_active_stream_tasks` entry does not, by itself, deny a real armed round) -- that test is the reference point for reproducing and then closing this gap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A round armed for a session before that session appears in `_active_stream_tasks` observes `shutdown()`'s cancel fanout promptly (not only via its own timeout).
- [ ] #2 The same promptness holds for `close_session`'s deny path.
- [ ] #3 A regression test arms a round ahead of session registration, triggers shutdown/close, and asserts the round resolves before its timeout deadline.
<!-- AC:END -->
