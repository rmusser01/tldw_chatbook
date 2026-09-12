---
id: TASK-26839
title: Console settings path queries the Workspace DB for a title it discards
status: Done
assignee: []
created_date: '2026-09-01 15:09'
updated_date: '2026-09-01 15:20'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26834 cause 3, sampled on the main thread in three separate in-terminal probe runs (30-60ms buckets). _ensure_active_console_session_settings computes the workspace tab title eagerly -- a synchronous registry get_workspace SQLite query -- and passes it to store.ensure_session, which uses the title ONLY when creating a session. With an active session (every provider/model display rebuild, every click that refreshes the control bar) the query runs and its result is discarded. The caller already computes creating_blank_session two lines above, so the title becomes conditional on it. Same disease TASK-21118 gated on the keystroke path, different entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With an active session, _ensure_active_console_session_settings performs zero registry get_workspace calls, gated by a counter test on a non-default workspace
- [x] #2 Creating a session still derives its title from the workspace registry exactly as before, pinned by test
- [x] #3 The pathlib.stat recompute in build_console_workspace_state is recorded as a DELIBERATE non-fix (ADR-028 security posture), not silently left
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-line fix at the call site: `_ensure_active_console_session_settings`
already computes `creating_blank_session` two lines above `ensure_session`,
so the title argument becomes conditional on it -- the workspace-title seam
(and its synchronous `get_workspace` SQLite query) runs only when a session
is actually being created. With one active, `DEFAULT_CONSOLE_SESSION_TITLE`
is passed and discarded exactly as any title would be.

Tests (`Tests/UI/test_console_settings_title_laziness.py`, shape borrowed
from TASK-21118's keystroke gate): the failing gate measured **5
get_workspace round-trips for 5 settings reads** with an active session
pre-fix, now 0; the control pins the seam still consulting the registry
(and proves the counter is wired). Two fixture findings recorded in the
file's docstrings: the harness boots with an active session, and the store
coerces an UNKNOWN workspace id back to the default with a numbered title,
so the creation pin sits on the seam rather than the full ensure path.

**Deliberate non-fix (AC #3):** the `pathlib.stat` recompute in
`build_console_workspace_state` (~20ms sampled) stays. Its docstring and
ADR-028 record that stored status is NEVER trusted for local-filesystem
bindings -- a vanished folder or symlink swap would otherwise keep a widened
root reporting "ready". Removing a security recompute to save 20ms is a
design decision for the ADR's owner, not a perf sweep.

Sweep: 152 passed across 5 session-surface files; the 3
`test_console_command_composer.py` reds are identical on pristine dev
(verified in a detached worktree) and recorded on TASK-25715.

Files: `tldw_chatbook/UI/Console_Modules/session.py` (one call site),
`Tests/UI/test_console_settings_title_laziness.py` (new).
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-26837, colliding with the older
"Provider-setup-can-report-a-successful-connection-test-yet-write-no-
api_settings-block" task that arrived on dev first while this one was still
in flight on its branch. Per the owner rule decided 2026-08-21 in TASK-19601
(**older id keeps it; the younger task renumbers with a provenance note,
regardless of Done status**), it renumbered to TASK-26839. Citations to
TASK-26837 in this branch's commit message, in PR #2291's body, and in
`Tests/UI/test_console_settings_title_laziness.py` and `session.py` comments
written before the renumber refer to THIS task; the other TASK-26837 holder
is the older arrival and keeps the id.
