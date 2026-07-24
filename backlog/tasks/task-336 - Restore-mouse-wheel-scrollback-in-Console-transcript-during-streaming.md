---
id: TASK-336
title: Restore mouse-wheel scrollback in Console transcript during streaming
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 07:29'
labels:
  - console
  - ux
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With the same overflowing conversation loaded: wheel-up over the transcript scrolls normally when the app is idle (verified: view moved, j4-30a), but during active streaming the identical wheel gesture produced no view change at all - the viewport stayed pinned to the bottom (j4-34; buffer compare 400ms after wheel showed the top region unchanged). Round 1 corroborates: two wheel-up attempts mid-run left the buffer byte-identical (j4-06/j4-07 are identical files). The user cannot read history with the primary terminal scroll gesture while a long reply streams; whether the event is swallowed or instantly yanked back, the observable outcome is a locked viewport.

**Repro:** Load a conversation whose transcript overflows -> confirm wheel-up scrolls while idle -> send a long prompt -> once [streaming] text is growing, hover the transcript and wheel-up -> view stays pinned to bottom.

**Verifier note:** Contradicts the shipped anchor contract 'released on scroll-up (never yanked)'. Before task-298 there was no auto-follow at all, so wheel scrollback during streams trivially worked — the lock is introduced by the anchor work. task-298's own notes concede AC#2 was pinned only by harness tests because 'xterm can't drive scroll reliably', i.e. the wheel path was never live-verified. Textual 8.2.7 code says pointer scroll-up SHOULD release the anchor (_scroll_up_for_pointer defaults release_anchor=True, unlike _scroll_down_for_pointer), so the live lock's mechanism is unexplained (possible re-anchor via a mid-stream code path or event-ordering under tick load) — but two independent rounds observed it and idle wheel works in the same harness, ruling out a pure tool artifact. Keeping P1: primary terminal scroll gesture dead for minutes-long streams; the click+PageUp workaround is undiscoverable.

**Source:** Console UX expert review 2026-07-20 (finding j4-wheel-scroll-locked-during-stream; P1, verdict REGRESSION, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-34-midstream-wheel.png`, `j4-30a-idle-wheel.png`, `j4-06-midstream.png`, `j4-07-scrolled-up.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wheel-up during streaming should detach auto-follow and scroll back (standard terminal behavior), with auto-follow re-engaging only when the user returns to the bottom
- [x] #2 A regression test pins the restored behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: understand Textual 8.2.7 anchor mechanics vs the transcript's 0.2s reconcile churn (row mounts/moves may re-engage or never release the anchor on wheel-up)
2. Fix so wheel-up during an active stream detaches follow and scrolls; returning to bottom re-engages (task-298 contract)
3. TDD at the widget level; live-verify against the real llama-server stream
<!-- SECTION:PLAN:END -->

## Implementation Notes

TWO real mechanisms, both fixed:

1. Late-arriving follow intent: the send-time ``anchor()`` travels through
   the coalesced sync pass and can land AFTER the user has already
   wheel-scrolled — yanking them back. The transcript now stamps
   ``note_follow_intent()`` at every programmatic jump-to-tail site
   (send dispatch, session activation, resume) and ``release_anchor()``
   stamps user scrolls; ``set_messages`` only honors a new-user-send anchor
   when the intent is newer than the last user scroll. The task-298
   contract boundary is pinned: a send AFTER scrollback still jumps.

2. The live byte-identical symptom (diagnosed with an instrumented served
   app against the real llama-server): during heavy row churn (sub-agent
   runs) the arrangement transiently collapses — ``max_scroll_y`` reads 0,
   ``scroll_y`` can go negative via the compositor anchor path — so the
   base ``allow_vertical_scroll`` gate is False at the moment the wheel
   event arrives and the gesture is silently dropped (no scroll, no
   ``release_anchor``). ``ConsoleTranscript.allow_vertical_scroll`` now
   accepts gestures whenever messages exist; a clamped scroll is a no-op
   but the reader's intent registers.

Verified: 3 UI tests (yank reproduced RED first; contract boundary;
collapsed-layout gate) and live against llama.cpp under an active
sub-agent run: wheel moves the view and it stays (previously: no movement
at all, matching j4-06/07 byte-identical evidence). Files:
`Widgets/Console/console_transcript.py`, `UI/Screens/chat_screen.py`.
