---
id: TASK-1890
title: Row-scope the script and audio interrupted sweeps
status: To Do
assignee: []
created_date: '2026-08-02'
labels:
  - watchlists
  - briefings
  - audio
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed during the whole-branch review fix wave for `chore/briefings-residuals-1810-1812`
(verdict: `.superpowers/sdd/briefings-residuals/whole-branch-verdict.md`, adjudication (b)),
which fixed `fail_interrupted_briefings`'s `exclude` to be row-scoped rather than
watchlist-scoped (task-1812, AC #3) and then closed the residual claim-registration
window that fix left open (this branch's own fix wave, Important 1). Both sibling sweeps
in the same family were deliberately left out of that scope and still exclude by the
coarser key:

- `fail_interrupted_scripts` (`tldw_chatbook/Subscriptions/briefing_cast.py`) excludes by
  `script_id`, not by the claimed row's own id.
- `fail_interrupted_audio` (`tldw_chatbook/Subscriptions/briefing_audio.py:1342`, `AND
  script_id NOT IN (...)`) excludes by `script_id` too.

This is the same bug class task-1812 fixed for briefings: a `script_id` (or, for audio, a
script whose scripts share one id) can have more than one `generating` row over its
lifetime -- a crash-zombie row left by a prior process, coexisting with a freshly-claimed
live row for the SAME key. A coarse, key-scoped `exclude` cannot tell them apart, so it
shields both rather than only the live one.

Unlike the briefings case, this errs toward *over*-protection, never over-sweeping: a
row-scoped exclude only ever narrows which rows survive, so leaving the coarse exclude in
place cannot cause a live row to be falsely marked `interrupted`. There is no correctness
hole here -- a zombie merely survives longer than it needs to, until a sweep runs while
its key is entirely unclaimed.

But task-1811 (this same branch) gave the coarse audio exclude a user-visible surface it
did not have before: `WatchlistsCollectionsScreen`'s Synthesize refusal toast
(`tldw_chatbook/UI/Screens/watchlists_collections_screen.py:5784`, "is already being
synthesized for this script") can now name a row that is not actually live -- a
crash-zombie audio row shielded by an unrelated live claim on the same `script_id`,
surfaced as if it were the thing blocking Synthesize. The *decision* to refuse is still
correct (something IS claimed for this script); the *row named in the message* can be
dishonest.

Reference: task-1812 (the briefings-side fix this generalizes) and this branch's
whole-branch verdict file for the adjudication that filed this task rather than folding
the fix into 1811/1812's own scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `fail_interrupted_scripts`'s `exclude` is scoped to the claimed script's own row id, not merely its `script_id`, mirroring `fail_interrupted_briefings`'s row-scoped shape (task-1812)
- [ ] #2 `fail_interrupted_audio`'s `exclude` is scoped to the claimed audio row's own id, not merely its `script_id`, the same way
- [ ] #3 Both row-scoped sweeps additionally handle the unrecorded-claim window the same way this branch's briefings fix does (a claim taken before its row id is recorded must still spare that row from a concurrent sweep) -- reference `chore/briefings-residuals-1810-1812`'s `pending_briefing_claim_watchlist_ids()` shape
- [ ] #4 A same-`script_id` crash-zombie script row and a live claim coexist in one sweep: the zombie is failed as interrupted, the live row is untouched (script sweep coexistence test)
- [ ] #5 A same-`script_id` crash-zombie audio row and a live claim coexist in one sweep: the zombie is failed as interrupted, the live row is untouched (audio sweep coexistence test)
- [ ] #6 A claim taken but not yet row-recorded survives a sweep run inside that exact window, for both the script and audio sweeps (window regression tests, mirroring this branch's briefings window test)
- [ ] #7 The Synthesize blocking toast (`watchlists_collections_screen.py`) never names a crash-zombie audio row as "already being synthesized" -- once the audio sweep is row-scoped, a zombie sharing a `script_id` with a live claim is swept before the toast is composed, so only the genuinely live row's label can appear
<!-- AC:END -->
