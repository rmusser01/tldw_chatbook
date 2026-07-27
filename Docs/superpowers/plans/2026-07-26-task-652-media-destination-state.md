# Media Destination State Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `MediaWindow` the sole Media view/selection owner, stop handled messages at that destination, and guarantee one metadata event causes one scoped mutation.

**Architecture:** `MediaScreen` snapshots the mounted `MediaWindow`; the destination stops selection/search/mutation messages before any await. Durable mutations use the existing scoped service, and presentation updates are guarded by the selected record plus a local operation generation. The app-level duplicate handler and legacy Media root/search paths are deleted.

**Tech Stack:** Python 3.11+, Textual messages/reactives/workers, scoped media services, pytest/pytest-asyncio, AST ownership tests.

**Backlog:** [TASK-652](../../../backlog/tasks/task-652%20-%20Remove-duplicate-Media-root-state-and-stop-mutation-bubbling.md)

**Specification:** [TldwCli Reactive State Decomposition Design](../specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md)

**Depends on:** TASK-647

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`; `backlog/decisions/011-chatbook-workbench-ui-system.md`

**Reason:** Existing ADRs assign Media state/actions to the registered destination and require stale-safe async completion.

---

## Execution and Test Boundary

Mounted coverage goes in `Tests/ProductionApp/test_media_state_ownership.py`.
It navigates a normal `TldwCli` to the real `MediaScreen`, posts real Media
messages, and installs only a narrow recording/fault-injection scoped service
on the mounted production window. No Media screen/window or app substitute is
permitted.

## Exact Removal Set

Remove these `TldwCli` reactives and companions:

```text
media_active_view
_initial_media_view_slug
current_media_type_filter_slug
current_media_type_filter_display_name
media_current_page
current_loaded_media_item
_media_search_timers
_media_search_generation
_initial_media_view
```

Legacy Chat sidebar pagination/selection fields belong exclusively to
TASK-650 and are not part of this task.
`MediaWindow.media_active_view` is destination-owned and remains allowed.

## File Structure

- Modify `tldw_chatbook/UI/MediaWindow_v2.py`: stop handled messages before await and guard stale presentation.
- Modify `tldw_chatbook/UI/Screens/media_screen.py`: keep snapshot/restore limited to the mounted window.
- Modify `tldw_chatbook/Event_Handlers/media_events.py`: retain message contracts/live shared helpers, delete app-root mutation/search/list implementations.
- Modify `tldw_chatbook/Event_Handlers/collections_tag_events.py`: remove fallback refresh through deleted app Media fields.
- Modify `tldw_chatbook/Event_Handlers/tab_initializers/misc_tab_initializers.py` and `__init__.py`: remove the obsolete Media initializer.
- Modify `tldw_chatbook/app.py`: remove root fields/watchers/input handlers and duplicate metadata registration.
- Create `Tests/ProductionApp/test_media_state_ownership.py`.
- Modify `Tests/test_application_state_ownership.py`.

## Task 1: Start TASK-652 and Reproduce Double Dispatch

- [ ] Move the task In Progress and add its task-local plan:

```bash
backlog task edit 652 -s "In Progress"
backlog task edit 652 --plan $'ADR required: yes\nADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md\nReason: Existing ADRs make MediaWindow the production state/action owner.\n\n1. Reproduce one event reaching destination and app.\n2. Stop handled messages before await.\n3. Remove duplicate root Media paths.\n4. Guard stale presentation and verify snapshots.'
```

- [ ] On a real mounted `MediaWindow`, install a recording scoped media
  mutation collaborator, post one real `MediaMetadataUpdateEvent`, and record
  mutation/refresh counts. Add a temporary observation hook to the real app
  handler only to prove bubbling; do not call an unbound method.
- [ ] Add analogous propagation assertions for type, search, item selection,
  delete/undelete, read-later, reading-highlight, and analysis mutation
  messages that the destination handles.
- [ ] Run:

```bash
pytest Tests/ProductionApp/test_media_state_ownership.py -q -k "metadata or propagation"
```

Expected: the metadata regression fails by observing duplicate reach before the fix.

## Task 2: Stop Messages and Guard Async Presentation

- [ ] Call `event.stop()` at the start of every handled destination method,
  before its first await or state mutation.
- [ ] Add destination-local generations for metadata mutation, item-detail
  loading, and search completion. Each async path captures its record/query
  identity before awaiting; durable mutations always settle, but list/viewer
  presentation applies only when the same generation, active media type,
  search tuple, and selected record are still current.
- [ ] On failure, retain stopped propagation, show bounded recovery, and log
  only operation/record category and exception category—not media content,
  metadata values, or response bodies.
- [ ] Add tests for exactly one mutation/refresh, service failure, selection
  change while awaiting, screen navigation while awaiting, reverse-order
  item-detail completion, and reverse-order search completion. An older detail
  or search must never overwrite the newer selection/query.

## Task 3: Delete App-Root Media State and Legacy Handlers

- [ ] Remove the exact root descriptors/fields, assignments, watcher,
  initializer, input handlers, legacy list/search/page handlers, and
  `TldwCli.on_media_metadata_update()`.
- [ ] Keep Media message classes and live destination helpers in
  `media_events.py`; delete root-only functions after an import census.
- [ ] Remove `collections_tag_events` fallback refresh through
  `current_media_type_filter_slug`; the destination refreshes itself through
  its scoped owner.
- [ ] Keep `MediaScreen.save_state()`/`restore_state()` reading and applying
  only actual `MediaWindow` fields. Do not add screen/root mirrors.
- [ ] Extend AST guards so root `app.media_active_view` is rejected while the
  destination descriptor remains allowed.

## Task 4: Verify and Close TASK-652

- [ ] Run:

```bash
pytest Tests/ProductionApp/test_media_state_ownership.py Tests/test_application_state_ownership.py -q
python -m compileall -q tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Event_Handlers/collections_tag_events.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/app.py
python -m ruff check tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Event_Handlers/collections_tag_events.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/app.py Tests/ProductionApp/test_media_state_ownership.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Event_Handlers/collections_tag_events.py tldw_chatbook/Event_Handlers/tab_initializers Tests/ProductionApp/test_media_state_ownership.py Tests/test_application_state_ownership.py
git diff --check
```

- Do not mass-format the verified pre-task `app.py` baseline exception.

- [ ] Commit implementation:

```bash
git add tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Screens/media_screen.py tldw_chatbook/Event_Handlers/media_events.py tldw_chatbook/Event_Handlers/collections_tag_events.py tldw_chatbook/Event_Handlers/tab_initializers tldw_chatbook/app.py Tests/ProductionApp/test_media_state_ownership.py Tests/test_application_state_ownership.py
git commit -m "refactor(media): own state and stop mutations at destination (task-652)"
```

- [ ] Re-read TASK-652, add Implementation Notes containing actual commands,
  counts, durations, mutation/staleness evidence, modified files, ADRs, and
  deviations, check all acceptance criteria, then mark Done and commit its
  task file:

```bash
backlog task 652 --plain
backlog task edit 652 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 -s Done
git add 'backlog/tasks/task-652 - Remove-duplicate-Media-root-state-and-stop-mutation-bubbling.md'
git commit -m "docs(backlog): close Media state ownership (task-652)"
```
