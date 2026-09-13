---
id: TASK-20978
title: >-
  A watchlists export test fails under load inside the directory picker
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - testing
  - test-integrity
  - flake
  - watchlists
priority: low
dependencies:
  - TASK-20970
---

## Description

Source: observed during **TASK-19561**'s post-rebase verification run, and run
down rather than assumed.

`Tests/Watchlists/test_watchlists_artifacts_pane.py::
test_export_feed_press_survives_an_os_error_from_the_service`
(`:5382`) failed once inside a 1570-test run with:

> `KeyError("No 'directory-navigation--hidden' key in COMPONENT_CLASSES")`

raised from inside the third-party `SelectDirectory` picker, not from
application code.

Evidence gathered at the time, all pointing the same way:

- Green on an identical re-run of the same selection (1566 passed).
- Green with the five tests added by that branch deselected (1561 passed).
- Green 8/8 consecutively when run alone.
- Green in an isolated run of the module (131 passed).
- Unreachable from the change under test — the failing test's harness is a
  `DestinationHarness`, never `TldwCli`, so none of that branch's
  `on_unmount` / watchdog / signal code runs in it.
- `pytest-randomly` is not installed in this environment, so collection order is
  fixed. The variation therefore is **not** ordering; it is load or accumulated
  process state.
- A baseline of the same four-directory selection on clean `origin/dev` produced
  one failure that was a *different* test, and no artifacts-pane failure —
  consistent with a flake rather than a defect in either branch.

A `COMPONENT_CLASSES` lookup failing intermittently inside a widget suggests the
picker is being queried at a point in its lifecycle where its component classes
are not yet resolved, which under load is a timing question. Whether the right
answer is a fix in how the test drives the picker, or an upstream constraint, is
open.

**This cannot currently be re-measured on `dev`.** All 122 tests in that module
are red at `684c6aba4` from the unrelated Actor Pack construction failure
(TASK-20970) — including this one — so any observation taken now would be of
that error, not this one. The dependency is recorded for that reason.

## Acceptance Criteria

- [ ] The cause of the intermittent `KeyError` in the directory picker is
      identified, not merely made to stop appearing
- [ ] `test_export_feed_press_survives_an_os_error_from_the_service` passes
      reliably under a loaded full-suite run, not only in isolation
- [ ] The fix is shown to address timing or lifecycle rather than suppressing the
      symptom — a retry, a broad `except`, or a skip does not satisfy this
- [ ] If the cause is shared by sibling tests that drive the same picker, they
      are covered too
- [ ] If the conclusion is that the flake originates upstream and cannot be
      fixed here, that is recorded with the evidence, and the test is made
      deterministic in a way that still exercises the behaviour it names
- [ ] Verification is performed after TASK-20970 is fixed, so the module is
      otherwise green and the measurement is of this defect

## Notes

Filed with its negative evidence intact deliberately. "Flaky, re-run it" is how
a real load-dependent defect gets absorbed; the run-down above is what
distinguishes this from that, and it should not have to be repeated by whoever
picks this up.
