---
id: TASK-2361
title: 'Realtime: idle ceiling can eject a speaker whose turn has not committed'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04'
updated_date: '2026-08-05 03:28'
labels:
  - realtime
  - voice
dependencies: []
priority: low
---

## Description (the why)

The idle ceiling counts activity as turn-commit or reply-end. A user who STARTS speaking
just before the deadline (speech_started, no commit yet) is cut off mid-utterance with
"idle for N minutes" (V4 final review M3). Matches the spec's letter, surprises an
attending user.

## Acceptance Criteria (the what)

- [x] `on_speech_started` while live refreshes the idle anchor (in both barge-in modes'
      reachable paths) so an in-progress utterance is never cut by the cost guard.
- [x] A genuinely silent session still exits at the ceiling.
- [x] FSM tests pin both directions.

## Implementation Plan (the how)

1. Read `RealtimeLoopController.on_speech_started` (`Chat/console_realtime_loop.py`)
   and the module docstring's "Idle ceiling" section to confirm the existing
   pending-anchor idiom (`_last_activity = None`, adopted by the next `tick(now)`)
   already used by `enter()`, `on_session_ready()`, and the barge-in path.
2. Add RED tests to `Tests/Chat/test_console_realtime_loop.py` pinning both
   directions: (a) `on_speech_started` while `live`, in EITHER barge-in mode, must
   refresh the anchor so a subsequent `tick()` does not fire `idle-timeout`; (b) a
   session that never sees `on_speech_started` still exits at the ceiling.
3. Apply the same pending-anchor idiom to `on_speech_started`, scoped to the `live`
   state, BEFORE the `acoustic_barge_in` gate that governs barge-in only -- so the
   refresh happens in both modes without changing barge-in semantics.
4. Run the FSM suite plus the contract trio to confirm no regressions.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed by refreshing `_last_activity` (the pending-anchor idiom already used by
`enter()`/`on_session_ready()`/the barge-in path) whenever `on_speech_started`
arrives while `live`, before the `acoustic_barge_in` mode gate that governs
barge-in only. Previously, default (keyboard-only) mode's `on_speech_started`
early-returned the instant it saw `acoustic_barge_in is False` -- correct for
barge-in (nothing to interrupt while `live`), but that same early return also
skipped the idle-anchor refresh, so a user who started an utterance just before
the ceiling but had not yet reached `on_turn_committed` could be ejected by the
next `tick()` mid-sentence. The refresh is scoped to `live` only: while
`thinking`/`speaking` (reachable only in acoustic mode, the only mode where the
mic stays hot then), the existing barge-in path already refreshes the anchor as
part of returning to `live`.

Two new tests pin both directions in `Tests/Chat/test_console_realtime_loop.py`:
`test_speech_started_while_live_refreshes_idle_anchor_both_modes` (RED against
the pre-fix code for `acoustic=False`, confirmed) and
`test_genuinely_silent_session_still_exits_at_idle_ceiling` (the ceiling itself
is unweakened). A third test,
`test_speech_started_does_not_refresh_anchor_outside_live`, pins that the
already-covered barge-in path is unaffected by this change.

Modified files:
- `tldw_chatbook/Chat/console_realtime_loop.py` -- `on_speech_started` now
  refreshes the idle anchor while `live`, in both barge-in modes; module
  docstring's "Idle ceiling" section updated to document the fourth use of the
  pending-anchor idiom.
- `Tests/Chat/test_console_realtime_loop.py` -- three new tests (see above).

Verification: `./.venv/bin/python -m pytest Tests/Chat/test_console_realtime_loop.py -p no:randomly -q` -- 35 passed. Full contract-trio + covering-suite run (`Tests/Chat/test_console_hands_free.py Tests/UI/test_console_hands_free_wiring.py Tests/UI/test_console_dictation.py Tests/UI/test_console_realtime_wiring.py Tests/Audio/test_realtime_mic_tap.py Tests/Chat/test_console_realtime_loop.py`) -- 212 passed, no regressions. No lesson filed: this was a straightforward application of the file's own established idiom, not a new trap.
<!-- SECTION:NOTES:END -->
