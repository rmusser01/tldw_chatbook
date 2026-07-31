---
id: TASK-1266
title: Retire TTSPlaygroundWidget by porting its tests to the rebuilt pane
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-31 02:55'
labels:
  - ui
  - speech
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SpeechPlaygroundPane` now carries the whole Playground: the axes populate
from the shared catalog, Generate reaches synthesis, results are delivered
to it, and a provider reconfiguration invalidates it. `TTSPlaygroundWidget`
is down to 6 methods and 586 lines, of which 372 are `compose` — the legacy
layout the rebuild exists to replace — plus `__init__` and three host-side
hooks.

What blocks deletion is not behaviour but coverage. `test_stts_playground_audio_cpp.py`
(43 tests) exercises the shared mixins *through* the legacy host. Deleting
the widget without porting them would delete the only tests covering catalog
staleness, request-generation ordering, and provider-switch control
restoration.

Measured, not estimated: swapping the test harness to mount the rebuilt pane
leaves 41 of 43 failing, clustered as

- **27** — the tests query `TTSPlaygroundWidget` by type selector. Mechanical.
- **8** — "Timed out waiting for Playground state". Needs investigation.
- **~6** — a model-select divergence (`Select.NULL` where `<opaque:model>` is
  expected, and an extra `second-model`). Needs investigation.

The 27 are a test-side rename. The other 14 are the real question: either the
rebuilt pane genuinely behaves differently, or its mount sequence differs
enough that the fixtures need adjusting. Find out which before changing
either side — a difference that turns out to be a real behavioural gap is a
bug in the pane, not a test to be re-pointed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 14 non-mechanical failures is classified as either a pane
      defect or a fixture difference, with the reason recorded
- [x] #2 Any pane defect found is fixed in the pane, not worked around in tests
- [x] #3 `test_stts_playground_audio_cpp.py` passes against `SpeechPlaygroundPane`
      with no loss of assertions
- [x] #4 `TTSPlaygroundWidget` is deleted, along with the `playground` branch in
      `STTSWindow.watch_current_view` and `_redesign_view` in `STTSScreen`
- [x] #5 The `TTSPlaygroundWidget` name no longer appears in the delivery or
      invalidation lookups in `stts_events.py`
- [x] #6 The Speech screen is driven on a live run after deletion, not only
      under pytest
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SetupStep.compose is now a final wrapper over per-step compose_step(): on exception it logs, flags compose_failed, and renders a one-line skip notice; __init_subclass__ guards each step's own on_mount/on_show against the gutted DOM; _refresh_active_ids drops failed steps from navigation; SummaryStep appends a reasoned ✗ row per skipped step. Two Pilot tests force RagStep.compose_step to raise and assert survival + summary row.
<!-- SECTION:NOTES:END -->
