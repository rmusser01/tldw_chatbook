---
id: TASK-16314
title: Console launch and live tail-follow
status: Done
assignee: []
created_date: '2026-08-15 00:21'
updated_date: '2026-08-15 05:48'
labels: []
dependencies:
  - TASK-16313
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Launch trajectory from chat_screen with single-letter binding; refresh on store revision; follow tail unless scrolled up, f resumes. Plan task 5 in Docs/superpowers/plans/2026-08-14-console-trajectory-view.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trajectory opens from Console,Live updates arrive without user action,Scroll-up suspends follow,f resumes follow
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in commits 47905905b/14373a7ac: y-key Console launch, revision-polled live tail-follow with suspend/resume; 9 tests green; see Implementation Notes and ADR-066

- **Approach**: per ADR-066 (`backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`)
  the Console launches the screen in a worker and the screen live-follows via revision
  polling (there is no observer bus).
- **Key files**: `tldw_chatbook/Chat/console_chat_store.py` (public
  `get_payload_revision` accepting session or conversation id; `write_trajectory_rows`
  bumps the revision per conversation + live session; `variant_sets_for_conversation`);
  `tldw_chatbook/UI/Screens/chat_screen.py` (`Binding("y", "open_trajectory_view")`,
  `action_open_trajectory_view`, module-level `_build_trajectory_snapshot`);
  `tldw_chatbook/UI/Screens/trajectory_screen.py` (`revision_provider`/`snapshot_builder`
  ctor kwargs, `set_interval(0.5, _poll_revision)`, follow state machine, `f` binding);
  `Tests/UI/test_trajectory_live.py` (9 tests, green — revision bus, live follow,
  suspend/resume, launch registration, real-DB snapshot integration).
- **Decisions**: **launch key is `y`** — the initial `j` binding was a lie on the surface
  a trajectory reader comes from: `console_transcript.py`'s focused key handler consumes
  `j`/`k` with `event.stop()`; the binding test now asserts `"j" not in bindings`.
  Live worker ordering: `_poll_revision` schedules `exclusive=True` workers carrying
  their revision; `_apply_live_snapshot` drops results whose revision no longer matches,
  so a slow older rebuild can never regress the ledger. Follow is pull-based from scroll
  geometry (checked each tick), with `call_after_refresh` scroll landing and a 1s grace
  window so the geometry poll cannot cancel an in-flight `f`; rebuilds preserve collapsed
  turns, search query and the visible window, scrolling to end only while following.
  `_build_trajectory_snapshot` uses the REAL seams (review fix): compaction via
  `ConsoleContextRepository.list_auxiliary_attempts` (projection filters
  `purpose == "conversation_compaction"`), variants via
  `store.variant_sets_for_conversation`; every seam degrades to empty rather than failing
  launch. **Variants are in-memory-only for cold DB**: variant CONTENTS are process-local
  (only selection metadata persists), so a conversation restored purely from disk
  legitimately renders without superseded variants — documented at the store method.
- **Deviations**: Textual 8.2.8's `run_worker` does not forward extra positional args to
  the callable, so the live rebuild dispatches through a lambda wrapper
  (`lambda: self._live_rebuild_worker(revision)`) — recorded in
  `backlog/docs/lessons-textual.md`. The pinned footer-text test in
  `test_console_workbench_contract.py` was updated for the "Y trajectory" hint.
<!-- SECTION:NOTES:END -->
