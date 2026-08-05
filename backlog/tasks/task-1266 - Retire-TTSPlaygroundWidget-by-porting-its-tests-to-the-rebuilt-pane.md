---
id: TASK-1266
title: Retire TTSPlaygroundWidget by porting its tests to the rebuilt pane
status: Done
assignee: []
labels:
  - ui
  - speech
  - tech-debt
priority: medium
---

## Description

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

## Acceptance Criteria

- [x] Each of the 14 non-mechanical failures is classified as either a pane
      defect or a fixture difference, with the reason recorded
- [x] Any pane defect found is fixed in the pane, not worked around in tests
- [x] `test_stts_playground_audio_cpp.py` passes against `SpeechPlaygroundPane`
      with no loss of assertions
- [x] `TTSPlaygroundWidget` is deleted, along with the `playground` branch in
      `STTSWindow.watch_current_view` and `_redesign_view` in `STTSScreen`
- [x] The `TTSPlaygroundWidget` name no longer appears in the delivery or
      invalidation lookups in `stts_events.py`
- [x] The Speech screen is driven on a live run after deletion, not only
      under pytest

## Implementation Notes

Investigated on `feat/speech-console-redesign` (commit `b94ae9c48`). The
question this task existed to answer — are the 14 non-mechanical failures
pane defects or fixture differences — is answered: **all of them were pane
defects.** Zero tests needed re-pointing.

Porting the 43 tests onto `SpeechPlaygroundPane` went 41 → 18 → 12 → 8 → 6
→ 5 → 3 → 2 → 1 → 0 failures, each step a distinct fix:

1. Axis selects allowed a selectable BLANK option ("no provider").
2. Provider-status and audio.cpp-restriction lines mounted empty — the copy
   lived in the legacy `compose()`, and the shared code only toggles them.
3. The empty-axis sentinel was UNAVAILABLE where it should be LOADING —
   claiming nothing is available before any catalog was fetched.
4. Provider knobs had **no default values**, so `float('')` raised on every
   Chatterbox and Higgs generation. 19 defaults recovered from legacy.
5. The text area started empty, so Generate refused outright.
6. Nothing re-evaluated Generate on text change (`on_tts_text_changed`
   matched no Textual message, which is why it was dead).
7. `_show_provider_specific_controls` recomposed the pane, destroying the
   axis selects mid-catalog-load; then the naive swap hit deferred-`remove()`
   DuplicateIds and queued duplicate mounts.
8. `_compose_voice_source` used the compose-only `with` idiom.

A live run afterwards caught a ninth that no test could: the language axis
read "Waiting for provider…" forever once audio.cpp's catalog arrived
without languages.

### Retirement (commit `83be26a9a`)

`TTSPlaygroundWidget` is deleted — 587 lines — and both mount sites plus the
two `stts_events` lookups point at `SpeechPlaygroundPane`. Its 83 tests moved
rather than being lost: `test_stts_playground_audio_cpp.py` (43) and
`Tests/TTS/test_stts_audio_cpp_generation.py` (40) drive the pane through its
`_tts_service_factory` and `_cli_setting` hooks, assertions unchanged.

Two things the retarget surfaced, both silent:

- `Tests/TTS` patched `STTS_Window.get_tts_service` **inline** in two tests,
  not only via the shared fixture, so those kept hitting the real service
  after the fixture was ported.
- The delivery lookup had to go back to `query_one` per host. I had switched
  it to `query()`, which the handler tests' fake app does not implement; it
  raised AttributeError into a broad `except` and returned None, so four
  delivery tests failed with an empty completions list.

`test_the_inventory_matches_the_live_widget` is retired with a note in its
place — it parsed the legacy widget's source, which no longer exists, and
`test_speech_playground_completeness.py` guards the same property against
what the pane actually mounts.

295 speech/stts tests pass; the lone failure fails identically at the
merge-base. Driven live after deletion: axes populate from the catalog, the
text area is seeded, language reads "Not used by this provider".
