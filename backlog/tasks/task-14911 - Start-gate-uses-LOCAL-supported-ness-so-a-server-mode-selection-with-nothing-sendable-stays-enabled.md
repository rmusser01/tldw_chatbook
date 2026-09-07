---
id: TASK-14911
title: >-
  Start gate uses LOCAL supported-ness, so a server-mode selection with nothing
  sendable stays enabled
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 02:00'
updated_date: '2026-08-11 03:52'
labels:
  - library
  - ingest
  - server
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while closing task-14827, and out of that task's scope (which was the forecast, not the gate).

task-14823 gates Start on a selection with nothing importable, but the predicate is 'the pre-flight found no supported type group' -- a LOCAL verdict. Since task-14827 the forecast knows the server refuses a different set (images have no server media type at all), so a folder of nothing but images now correctly forecasts '0 will be sent to the server - N will fail (unsupported by the server)' while Start stays enabled and every row lands as a failure. That is precisely the guaranteed-failure submit task-14823 exists to prevent, one backend over.

The forecast already carries the answer (will_import == 0 and every staged file refused), so the gate should read it rather than re-deriving supported-ness from type groups -- the same 'one computation' move task-14820 made for the commit and consent lines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing Start in server mode on a selection the server will refuse entirely is blocked, with a gate line naming the reason
- [x] #2 The same selection in local mode is unaffected, because those files import fine on this machine
- [x] #3 The gate reads the existing IngestForecast rather than deriving a second notion of what is importable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED first: a state-level test that a server-mode images-only selection leaves Start enabled, and an end-to-end test that pressing Start queues doomed jobs.
2. Make the gate backend-aware in Library/library_ingest_state.py by READING the existing IngestForecast (will_import == 0, no predicted matches, staged_total > 0) rather than re-deriving supported-ness from type groups.
3. Give the server case its own sentence and its own recovery, distinct from the local 'nothing can be imported' wording, using the arc's established 'unsupported by the server' vocabulary.
4. Feed the same selection_has_nothing_importable flag the submit seam already refuses on, so no entry point can route around a disabled button; prove it end to end (no job created).
5. Mutation-check both halves separately: the state predicate and the submit-seam flag.
6. Update Docs/User_Guide/library/import-and-export.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The gate now asks the backend the run is aimed at, by reading the forecast that already knows the answer.

**The hole (AC#1).** `nothing_importable` is a LOCAL verdict -- "did the pre-flight find a supported type group" -- so a folder of nothing but images (a real local capability, deliberately left server-unmapped by task-3307) forecast `0 will be sent to the server · 3 will fail (unsupported by the server)` with Start still ENABLED. That is task-14823's guaranteed-failure submit, one backend over, and the RED run said so in those words: *"Start stayed live for a selection the server refuses entirely: '0 will be sent to the server · 3 will fail (unsupported by the server)'"*.

**One computation, not two (AC#3).** A new `nothing_sendable` term reads the existing `IngestForecast`: `targets_server and staged_total > 0 and will_import == 0 and will_match == 0`. Nothing about supported-ness is re-derived, so the gate line and the commit line cannot state different numbers -- the same move task-14820 made for the commit and consent lines. The `will_match` clause preserves the task-2223 ruling: zero imports plus predicted duplicate matches keeps Start enabled, because the duplicate probe is capped best-effort and never a blocker. `will_fail_tooling` cannot reach this predicate, since a server forecast zeroes local tooling gaps (task-14827) -- a missing local extra still arms the two-press consent locally rather than a hard gate, untouched.

**Two vocabularies, kept apart.** Ordered AFTER `nothing_importable` on purpose. A file nothing on this machine can read is diagnosed identically whichever target is selected, and switching to Local would not help it, so that case keeps "Nothing in this selection can be imported — N unsupported files." The new branch is the other case -- files this machine reads fine that this destination will not take -- and gets the arc's established server vocabulary plus its own recovery: "Nothing in this selection can be sent to the server — 3 files unsupported by the server. Switch to importing on this machine, or choose video, audio, document, PDF or e-book files." The recovery clause is appended only when the server's refusal is a blocker; a 0-byte file (task-14910) is refused on both backends, so switching target is not offered for it, and it is named separately ("... — 1 file unsupported by the server and 1 empty file.").

**Both halves of the gate (AC#1 again).** The predicate feeds `start_enabled` AND the existing `selection_has_nothing_importable` flag, which `_submit_library_ingest_form` already refuses on with the gate's own reason -- so no new screen logic was needed and no entry point (Enter in the path field, an accelerator, a future caller) can route around a disabled button. Proven end to end: `test_server_mode_start_creates_no_job_for_a_selection_the_server_refuses` drives the real pre-flight and the real submit seam and asserts the registry holds NO job. Mutation-checked separately: disabling the state predicate reddens 4 unit tests + the integration test; leaving the state gate closed but reverting only the submit-seam flag still queued 2 jobs ("a submit the server was certain to refuse reached the queue"), which is the whole reason the seam-side refusal exists.

**AC#2** is a guard at both levels: the same images-only selection in local mode keeps `start_enabled=True`, an empty gate line, and a submit that takes the CONSENT route (this venv has no OCR backend) rather than the refusal route.

**Process note.** The Implementation Plan was recorded after the RED tests were written rather than before -- the tests and the plan say the same thing, but the order deviated.

Modified: `Library/library_ingest_state.py`, `UI/Screens/library_screen.py` (comment only -- the seam already refuses on the shared flag), `Tests/Library/test_library_ingest_state.py`, `Tests/integration/test_library_ingest_flow.py`, `Docs/User_Guide/library/import-and-export.md`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
