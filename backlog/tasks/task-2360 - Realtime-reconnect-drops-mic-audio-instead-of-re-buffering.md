---
id: TASK-2360
title: 'Realtime: reconnect drops mic audio instead of re-buffering'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-04'
updated_date: '2026-08-05 03:29'
labels:
  - realtime
  - voice
dependencies: []
priority: medium
---

## Description (the why)

The mic tap's pre-ready buffer guarantees first words spoken during the CONNECT handshake
are not lost, but `mark_ready()` is one-way: during a mid-loop RECONNECT the tap streams
into a session slot that is momentarily None and frames are dropped. The chip says
"reconnecting…" so it is not invisible, but the entry-time guarantee does not extend across
reconnects (V4 final review M2).

## Acceptance Criteria (the what)

- [x] Speech during a reconnect window is buffered (bounded) and flushed to the new session
      once ready, mirroring the entry-time guarantee.
- [x] A failed reconnect discards the buffer with the existing reasoned exit (no stale audio
      sent to a later session).
- [x] Pinned by a wiring test driving frames during the reconnect window.

## Implementation Plan (the how)

1. Read `RealtimeMicTap` (`Audio/realtime_mic_tap.py`) -- `start()`, `mark_ready()`,
   `set_gated()`, `stop()` -- and the reconnect wiring in `UI/Screens/chat_screen.py`
   (`_console_realtime_begin_reconnect`, `_on_console_realtime_ready`,
   `_on_console_realtime_frames`) to find the exact drop point, and confirm `stop()`
   is documented terminal (never a pause) by a prior review's binding ruling.
2. Trace why frames actually get lost: `session.session` is reassigned to the NEW
   (not-yet-connected) provider session early in `_connect_console_realtime`, well
   before `connect()` resolves -- a real `OpenAIRealtimeSession.append_audio` silently
   drops anything enqueued before `connect()` sets up its outbound queue
   (`_enqueue`'s `self._loop is None` guard). So the bug is not simply
   "`session.session is None`"; buffering must happen at the TAP, independent of
   whatever the downstream session does.
3. Add RED tests to `Tests/Audio/test_realtime_mic_tap.py` for a new re-armable
   `begin_buffering()` method: reroutes post-ready frames back into the same bounded,
   in-order pre-ready buffer; idempotent; a no-op before the first `mark_ready()`;
   `stop()` (already terminal) discards a re-buffered window exactly like the
   original pre-ready buffer; the recorder is never touched (not `stop()`).
4. Implement `begin_buffering()` on `RealtimeMicTap` -- flips `_ready` back to False
   under the existing lock, reusing `mark_ready()`'s existing flush machinery
   unchanged for the release side.
5. Add RED wiring tests to `Tests/UI/test_console_realtime_wiring.py` driving real
   frames (via the REAL tap + a fake recorder) through a reconnect window, both for
   a reconnect that succeeds (frames flush to the new session in order) and one that
   fails (frames never reach the failed session; the loop's existing teardown is
   what discards them).
6. Wire `session.tap.begin_buffering()` as the FIRST action in
   `_console_realtime_begin_reconnect`, before `session.session` is even cleared, to
   minimize the window where a frame could still reach a dying/not-yet-ready session.
   `_on_console_realtime_ready`'s existing `tap.mark_ready()` call already runs
   unconditionally for both first-connect and every reconnect, so no change needed
   on the release side.
7. Run the mic-tap suite, the wiring suite, and the contract trio to confirm no
   regressions. Check whether any new diagnostic log call requires the
   `check_persistent_diagnostic_inventory.py` regeneration.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `RealtimeMicTap.begin_buffering()`: a re-armable buffering state, deliberately
NOT `stop()` (which stays terminal per the prior review's binding ruling). It flips
`_ready` back to `False` under the tap's existing lock, so subsequent frames route
back into the SAME bounded, in-order pre-ready buffer `_on_recorder_frames` already
used before the very first `mark_ready()` call -- no new buffer, no new eviction or
ordering logic. The recorder itself is untouched (still running); the call is
instant, unlike `stop()`'s bounded quiescence wait. Release is "free": the existing
`_on_console_realtime_ready` handler already calls `tap.mark_ready()` unconditionally
on every ready (first connect or reconnect), so the SAME flush path used for the
entry-time guarantee now also drains the reconnect-window buffer, in order, into the
new session once it goes ready.

Wiring: `_console_realtime_begin_reconnect` (`UI/Screens/chat_screen.py`) now calls
`session.tap.begin_buffering()` as its FIRST action, before `session.session` is even
cleared -- minimizing the window where a frame could still reach the dying session.
Investigation found the drop is subtler than "`session.session is None`": that field
is reassigned to the NEW provider session quite early (inside `_connect_console_
realtime`, before `await provider_session.connect()`), so a naive fix gated only on
"is session.session set" would still leak frames into a session whose handshake has
not started -- a real `OpenAIRealtimeSession.append_audio`/`_enqueue` silently drops
those (its outbound queue does not exist until `connect()` runs). Buffering at the
tap sidesteps this entirely: those frames now never reach `on_frames` at all until
the new session is genuinely ready.

A failed reconnect needs no separate discard path: `_console_realtime_begin_
reconnect`'s buffer is either flushed by the next successful `mark_ready()`, or the
loop's EXISTING exit teardown (`_teardown_console_realtime_loop` ->
`_close_console_realtime_resources` -> `tap.stop()`) discards it -- `stop()` already
clears the buffer as part of its existing terminal contract, and the tap itself
(along with any leftover buffered audio) is never reused: a fresh loop entry always
builds a brand-new `RealtimeMicTap`, so there is structurally no "later session" a
stale reconnect-window buffer could ever leak into.

Tests: `Tests/Audio/test_realtime_mic_tap.py` gained six tests for
`begin_buffering()` (reroute + in-order flush, no-op before first ready, idempotent,
discard-on-stop, not-`stop()`/recorder-keeps-running). `Tests/UI/test_console_
realtime_wiring.py` gained two wiring tests using the REAL tap + a fake recorder + a
fake session: one drives frames through a reconnect that succeeds and asserts they
land on the new session in order; the other drives frames through a reconnect that
fails and asserts they never reach the failed session. `Tests/Chat/test_console_
realtime_loop.py` is untouched by this task (that suite belongs to task-2361).

Modified files:
- `tldw_chatbook/Audio/realtime_mic_tap.py` -- new `begin_buffering()` method;
  module docstring's "Guarantees this module makes" list updated.
- `tldw_chatbook/UI/Screens/chat_screen.py` -- `_console_realtime_begin_reconnect`
  now re-arms the tap's buffer before doing anything else.
- `Tests/Audio/test_realtime_mic_tap.py` -- six new tests (see above).
- `Tests/UI/test_console_realtime_wiring.py` -- two new wiring tests (see above).

Verification: `./.venv/bin/python -m pytest Tests/Audio/test_realtime_mic_tap.py -p no:randomly -q` -- 21 passed. Full contract-trio + covering-suite run (`Tests/Chat/test_console_hands_free.py Tests/UI/test_console_hands_free_wiring.py Tests/UI/test_console_dictation.py Tests/UI/test_console_realtime_wiring.py Tests/Audio/test_realtime_mic_tap.py Tests/Chat/test_console_realtime_loop.py`) -- 212 passed, no regressions.

Diagnostics inventory: this task adds one new `logger.opt(exception=True).debug(...)`
call (the `begin_buffering()` try/except in `_console_realtime_begin_reconnect`,
matching the existing style of every other tap call in that method), which does
change `chat_screen.py`'s entry in `Docs/security/production-diagnostic-inventory.json`.
Regeneration was investigated but deliberately NOT committed here: running
`scripts/check_persistent_diagnostic_inventory.py` against a clean worktree of this
branch's own base commit (`a8ef42bff`, no changes from this task at all) already
fails the same way, with a much larger diff spanning unrelated files (e.g.
`UI/Console_Modules/dictation.py`, `Utils/instance_lock.py`,
`Widgets/Persona_Widgets/personas_library_pane.py` -- none touched by this task).
That drift pre-dates this branch, and a worktree already exists on this machine for
a dedicated fix (`codex/fix-diagnostic-inventory`). Regenerating the inventory here
would bundle unrelated, already-owned drift into this task's diff. Filed as a
concern for the dispatcher rather than silently absorbed.
<!-- SECTION:NOTES:END -->
