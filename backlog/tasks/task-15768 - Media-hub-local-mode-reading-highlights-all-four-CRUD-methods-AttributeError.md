---
id: TASK-15768
title: 'Media hub local mode: reading-highlights CRUD all AttributeError against the real service'
status: Done
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - media
priority: high
---

## Description

Found and confirmed during task-15467 (input-latency burn-down), explicitly
left unfixed as out of scope for that task's off-loop-threading AC:
`MediaReadingScopeService` calls `list_reading_highlights`,
`create_reading_highlight`, `update_reading_highlight`, and
`delete_reading_highlight` in local mode, but `LocalMediaReadingService` only
implements the unprefixed `list_highlights`/`create_highlight`/
`update_highlight`/`delete_highlight`. All four calls `AttributeError`
against a real local service — confirmed directly
(`getattr(service, method_name)` raises for all four).

Every local-mode media-item click already hits this: loading highlights is
swallowed by `_load_media_item_detail`'s broad `except Exception` and
silently presents zero highlights, and every local-mode highlight
create/update/delete action hits the identical `AttributeError`. The
Library screen's own, separate call sites already use the correct unprefixed
names and work fine — this is specifically the Media hub's scope-service
bridge, which never matched `LocalMediaReadingService`'s actual method
names.

## Acceptance Criteria

- [x] `MediaReadingScopeService`'s local-mode calls for list/create/update/delete
      reading highlights reach `LocalMediaReadingService`'s real methods (no
      `AttributeError`), by renaming one side to match the other
- [x] A local-mode media item with existing highlights shows them in the
      Media hub item detail (regression test against a real
      `LocalMediaReadingService`, not a mock that hides the name mismatch)
- [x] Creating, updating, and deleting a highlight from the Media hub in
      local mode works end-to-end (tests)
- [x] The broad `except Exception` around the detail load in
      `_load_media_item_detail` is narrowed or logged so a future
      naming/contract drift is visible instead of silently presenting empty
      state

## Implementation Plan

1. Re-locate the mismatch at HEAD (`bb91fef73`): confirm
   `MediaReadingScopeService.{create,list,update,delete}_reading_highlight(s)`
   dispatch the *prefixed* leaf names via `_call_local_leaf`, while
   `LocalMediaReadingService` only implements the unprefixed
   `create_highlight`/`list_highlights`/`update_highlight`/`delete_highlight`.
2. Establish drift direction from git history (`git log -S`): the prefixed
   scope methods were born in the server-parity work calling
   `service.create_reading_highlight(...)` — written against
   `ServerMediaReadingService`'s back-compat aliases; both leaves' primary
   contract is the unprefixed names (server implements unprefixed as primary,
   prefixed only as thin aliases; local implements only unprefixed). So the
   scope-service dispatch is the drifted side: fix the four leaf-dispatch
   name strings there. No public API changes — the Media hub keeps calling
   the scope's `*_reading_highlight*` methods.
3. Born-red tests first (fail with the current `AttributeError` signature):
   - Scope-level: drive all four prefixed scope methods in local mode against
     a REAL `LocalMediaReadingService` + real `MediaDatabase` (no fake).
   - Media-hub level (`Tests/UI/test_media_window_v2_parity.py` harness):
     item-detail highlight load shows seeded highlights, and
     create/update/delete handlers round-trip end-to-end against the real
     local service.
4. AC3 completeness: the Media hub's update handler forwards
   `quote=event.quote` but `quote` is not an updatable field on EITHER leaf
   (server `ReadingHighlightUpdateRequest` = color/note/state; local
   `update_highlight` = color/note/state) — any non-None quote TypeErrors.
   Stop forwarding it.
5. AC4: in `MediaWindow_v2`, log contract drift (`AttributeError`/`TypeError`)
   from the item-detail highlight load and the detail load with a full
   traceback at error level instead of the current message-only/category-only
   swallow.
6. Align the fakes/tests that encoded the wrong contract:
   `FakeLocalMediaService`'s prefixed highlight methods (the mismatch-hiding
   mock) removed; server-mode routing test assertions moved to the unprefixed
   leaf calls; off-loop (task-15467) probe doubles renamed to the real leaf
   names; stale "always AttributeErrors" docstrings updated.
7. ruff check + format on touched files; targeted pytest for the touched test
   modules; baseline any unrelated reds against origin/dev.

## Implementation Notes

Confirmed at HEAD `bb91fef73` and fixed on the scope-service side — the side
git history shows drifted.

**The mismatch.** `MediaReadingScopeService.{create,list,update,delete}_
reading_highlight(s)` dispatched the *prefixed* leaf names
(`_call_local_leaf(..., "create_reading_highlight", ...)` etc.), but
`LocalMediaReadingService` implements only the unprefixed
`create_highlight`/`list_highlights`/`update_highlight`/`delete_highlight`
(`tldw_chatbook/Media/local_media_reading_service.py:1875-1983`).
`ServerMediaReadingService`'s primary methods are ALSO the unprefixed names —
the prefixed ones there are thin back-compat aliases
(`server_media_reading_service.py:1603-1613`), which is why server mode
worked and local mode `AttributeError`'d. History
(`git show 7a6129009:...media_reading_scope_service.py` /
`git show 435c9dac7 --`): the prefixed scope methods were born in the
server-parity work calling `service.create_reading_highlight(...)` — written
against the server aliases, never matching the local leaf; task-15467 later
converted the same names to `_call_local_leaf` and documented the pre-existing
AttributeError. So the scope dispatch was the drift; both leaves' documented
contract is unprefixed.

**Fix (production).**
- `tldw_chatbook/Media/media_reading_scope_service.py`: the four
  `_call_local_leaf` dispatch names changed to the unprefixed leaf contract
  (`create_highlight`/`list_highlights`/`update_highlight`/
  `delete_highlight`). Scope-service public API (what `MediaWindow_v2`
  calls) unchanged.
- `tldw_chatbook/UI/MediaWindow_v2.py`:
  - `load_reading_highlights`: new `(AttributeError, TypeError)` branch logs
    the full traceback at error level (contract drift can no longer
    masquerade as "no highlights"); generic runtime failures keep the prior
    degrade-to-empty behavior (AC4).
  - `_load_media_item_detail`'s broad except now logs
    `AttributeError`/`TypeError` with `logger.opt(exception=True).error`
    instead of only the exception category (AC4).
  - `_handle_reading_highlight_update_async` no longer forwards
    `quote=event.quote`: `quote` is not an updatable field on EITHER backend
    (server `ReadingHighlightUpdateRequest` = color/note/state;
    local `update_highlight` = color/note/state), so any quote-carrying
    update TypeError'd against both real services — required for AC3 to hold
    in general.

**Born-red evidence** (all three watched failing with the current defect's
signature before the fix, command:
`PYTHONPATH=<worktree> .venv/bin/python -m pytest <nodeids>`):
- `Tests/Media/test_media_reading_scope_service.py::
  test_scope_service_reading_highlight_crud_reaches_real_local_service` —
  failed `AttributeError: 'LocalMediaReadingService' object has no attribute
  'list_reading_highlights'` at `media_reading_scope_service.py:170`.
- `Tests/UI/test_media_window_v2_parity.py::
  test_media_window_local_highlight_crud_end_to_end_against_real_service` —
  failed `assert [] == ['1']` (the silent empty-highlights symptom).
- `Tests/UI/test_media_window_v2_parity.py::
  test_media_window_logs_highlight_contract_drift_with_traceback` — failed:
  captured log was the message-only swallow, no traceback/"contract" marker.
All three pass after the fix (`3 passed`).

**Test-suite alignment** (the fakes that hid the mismatch):
- `Tests/Media/test_media_reading_scope_service.py`:
  `FakeLocalMediaService`'s four prefixed highlight methods REMOVED (real
  local service never had them); server-mode routing test now asserts the
  unprefixed leaf call (`create_highlight`, ...) and the unprefixed fake
  payload shapes.
- `Tests/Media/test_media_reading_scope_service_off_loop.py` (task-15467
  probes): recording-double leaves renamed to the real contract; the real-DB
  item-click chain test now EXERCISES `list_reading_highlights` against the
  real `LocalMediaReadingService` (its previous exclusion note documented
  exactly this defect and is superseded).
- `Tests/Media/test_local_media_reading_service.py`: stale
  "prefixed methods always AttributeError" docstring updated.
- `Tests/UI/test_media_window_v2_parity.py`: server-mode mocked CRUD test no
  longer asserts a forwarded `quote` on update.

**Verification** (`PYTHONPATH=<worktree> .venv/bin/python -m pytest ...`):
the four touched modules `Tests/Media/test_media_reading_scope_service.py
test_media_reading_scope_service_off_loop.py
test_local_media_reading_service.py Tests/UI/test_media_window_v2_parity.py`
→ **216 passed**. Blast radius (13 modules referencing the touched services,
incl. `Tests/ProductionApp/test_media_state_ownership.py`,
`test_service_composition_lifecycle.py`, Library/MCP/RuntimePolicy suites) →
372 passed; the 14 failures + ~22 teardown errors were **reproduced
identically on a throwaway baseline worktree at base `bb91fef73`**
(`14 failed, 155 passed, 24 errors` — same FAILED set, same `_no_network_io`
teardown-error family), i.e. pre-existing dev reds, not this change.
`ruff check` clean on all touched files; `ruff format --check` clean on all
touched files except two test files whose drift **pre-exists at base**
(verified via `git show bb91fef73:<file> | ruff format --check -`); only the
one drift hunk this task introduced was hand-formatted.

**Not fixed here (adjacent, pre-existing):** the
`MediaReadingHighlightUpdateEvent.quote` field remains accepted-but-ignored
by the update path on both backends; no production widget posts these events
yet (grep: only tests construct them).
