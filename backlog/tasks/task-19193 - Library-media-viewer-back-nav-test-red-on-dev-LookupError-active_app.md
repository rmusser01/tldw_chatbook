---
id: TASK-19193
title: Library media-viewer back nav test red on dev (LookupError active_app)
status: Done
assignee: []
created_date: '2026-08-20'
labels:
  - test-health
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_screen_navigation.py::test_action_library_media_viewer_back_returns_to_list_and_refocuses_it`
is red on dev `7877defba` (re-confirmed 2026-08-20 in a pristine worktree,
isolated config), with the same signature TASK-19046's implementer AND
reviewer both reproduced at their pristine base:

    screen.action_library_media_viewer_back()   # test_screen_navigation.py:2619
    ...
    textual/message_pump.py:257: in app
        return active_app.get()
    LookupError: <ContextVar name='active_app' at 0x...>

The test is synchronous by design: it constructs `LibraryScreen` outside a
running app, stubs `refresh`/`call_after_refresh`/`set_timer`, and pins
task-2856's contract that Escape from the plain read-only media viewer reuses
the exact `_exit_library_media_viewer` reset sequence and re-focuses the
list's first row (one seam for both exits). Something on the
`action_library_media_viewer_back` path now reaches `self.app`, which raises
outside an app context. Diagnose which it is before fixing: a product change
that made the exit path app-dependent (fix or stub at the new seam without
weakening the one-seam contract) versus a test-harness gap (extend the
harness). Per the owner's standing ruling, prefer the durable fix over
whatever silences the test fastest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The test passes on dev without deleting it, skipping it, or weakening the task-2856 one-seam contract it pins (same reset sequence for Escape and the back button, list first-row refocus).
- [x] #2 Implementation Notes identify the commit/change that introduced the `self.app` access on this path and classify the failure (product regression vs harness gap), with the fix applied at the layer that classification names.
- [x] #3 The surrounding `Tests/UI/test_screen_navigation.py` suite passes.
<!-- AC:END -->

## Implementation Plan

1. Reproduce at base (origin/dev 63901c30d) and capture the full traceback;
   read which frame reaches `self.app` (`active_app.get()`).
2. `git log -S _load_library_media_list_if_needed` to find the change that
   made `_exit_library_media_viewer` app-dependent; read that commit's own
   test coverage for the new behavior.
3. Classify: product regression vs harness gap. Production always runs the
   exit inside an app context (button handler / key action); `run_worker`
   is the sanctioned mechanism, so out-of-context is test-only reachable.
4. Fix at the layer the classification names. If harness: make the sync
   test's fixture represent the browsed-list flow it pins (seed the browse
   controller's applied page) instead of the deep-link state that now
   legitimately schedules a worker load, and pin the applied-page exit's
   no-reload contract with a run_worker capture asserting zero requests.
5. Green after; full `Tests/UI/test_screen_navigation.py` run (expect
   129 passed); repo-wide `--collect-only -q`; grep for sibling tests
   sharing the mechanism and report.

## Implementation Notes

**Classification: harness gap, not a product regression.** Fixed at the
test layer only; no product code changed.

**Root cause, frame by frame** (base = dev `63901c30d`, reproduced in a
pristine worktree with the pytest-isolated config):
`action_library_media_viewer_back()` → `_exit_library_media_viewer()` →
`_load_library_media_list_if_needed()` → (controller has
`applied_result is None`) → `LibraryMediaBrowseController.request()` →
`self._run_worker(...)` which is `screen.run_worker` → Textual
`dom.py:527` reads `self.app._thread_id` → `active_app.get()` raises
`LookupError` / `NoActiveAppError` because the synchronous test constructs
an un-mounted `LibraryScreen` with no running app.

**Introducing change:** `f86c636af` ("fix(library): close media lifecycle
authority gaps", 2026-08-16) added `self._load_library_media_list_if_
needed()` to `_exit_library_media_viewer` so a DEEP-LINK viewer entry
(no Media page ever applied) loads the exact page + facets on exit. That
is deliberate, sanctioned product behavior (`run_worker` is the repo's
background-work mechanism and always has an app context in production —
the exit only ever runs from a button handler or key action). The commit
covers that flow itself with the live-pilot test
`Tests/UI/test_library_shell.py::test_library_media_deep_link_back_loads_
exact_page_and_facets`. The one-seam contract the red test pins (Escape
and the "‹ Back to list" button share `_exit_library_media_viewer`'s
reset sequence + first-row refocus) is intact — only the synchronous
harness predated the exit path's new applied-page dependency.

**Fix (harness):** the scenario this test pins — Escape from a viewer the
user opened FROM the browsed list — implies an applied page, so the test
now seeds `screen._library_media_browse_controller.applied_result` with a
real one-item `MediaBrowseResult` (validated by the product's own
`__post_init__`, no fakes). `_load_library_media_list_if_needed()` then
early-returns, and every original assert stands unchanged (reset
sequence, single recompose refresh, settle-window timer, first-row
refocus). Added a `run_worker` capture stub asserting
`worker_requests == []`: with a page applied the exit must NOT re-request
it (pins f86c636af's early-return, and turns any future app-dependent
work on this path into a clear assertion instead of a LookupError).

**Evidence:**
- Born-red: full `LookupError: <ContextVar name='active_app'>` traceback
  captured at base before any change.
- Mutation check (Edit-based, restored): with the applied-page seed
  disabled and only the worker stub in place, the test fails loudly —
  `refresh_calls == [True, True, True]` (the load's two canvas-sync
  fallback refreshes) — proving the new pin is not vacuous.
- `Tests/UI/test_screen_navigation.py`: 1 failed / 128 passed at base →
  **129 passed** after (299s → 177s wall).
- Repo-wide `pytest --collect-only -q`: 52113 tests collected, exit 0.
- Same-mechanism sweep: the only other test invoking
  `action_library_media_viewer_back` (`Tests/UI/test_library_canvas_
  sync_defects.py:160`) runs inside a live pilot (`host.run_test`) and is
  unaffected; the sub-state parametrized sibling never reaches the seam.

**Files changed:** `Tests/UI/test_screen_navigation.py` (one test), this
task file.
