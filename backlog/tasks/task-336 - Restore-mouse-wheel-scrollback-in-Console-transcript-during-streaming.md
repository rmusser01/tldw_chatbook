---
id: TASK-336
title: Restore mouse-wheel scrollback in Console transcript during streaming
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, regression]
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
- [ ] #1 Wheel-up during streaming should detach auto-follow and scroll back (standard terminal behavior), with auto-follow re-engaging only when the user returns to the bottom
- [ ] #2 A regression test pins the restored behavior
<!-- AC:END -->
