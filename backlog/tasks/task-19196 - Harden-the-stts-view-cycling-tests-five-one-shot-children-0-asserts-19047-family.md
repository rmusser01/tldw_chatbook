---
id: TASK-19196
title: Harden the stts view-cycling test's five one-shot children[0] asserts (19047 family)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20'
labels:
  - test-health
  - flake
  - stts
dependencies:
  - TASK-19047
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Conditional hardening candidate, filed from the third-wave close-out. At dev
`7877defba`,
`Tests/UI/test_stts_profile_library.py::test_voice_profiles_view_mounts_focused_library_without_hiding_other_views`
(:1085, body :1104-1139) carries five one-shot
`app.query_one(".stts-content").children[0]` isinstance asserts (playground →
settings → audiobook → dictation → playground) — the exact
raising-predicate/empty-window class TASK-19047 fixed elsewhere in the same
file: `STTSWindow.watch_current_view` swaps the body in a `speech-view-mount`
worker, and `.children[0]` sampled between `remove_children()` and `mount()`
raises `IndexError` (see the 2026-08-20 "settle whose predicate can RAISE"
entry in `backlog/docs/lessons-testing-evidence.md`, and the
`_stts_content_first_child` helper 19047 added at :1053, used with
`_wait_until` at e.g. :2910).

Important honesty note: these five asserts NEVER fired across all of 19047's
and its reviewer's CPU-burner load runs — this is a structural sibling of a
proven flake class, not an observed flake. Hence the conditional shape: either
prove it can fire (load reproduction) or convert it mechanically to the
already-shipped helpers and re-prove the whole file under load. Do not
half-convert.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Route A first: run the target test standalone 12-15x under 20 CPU
   burners (19047's recipe; 20 is the count that reproduced the
   children[0]/IndexError shape there), outputs to files, and record the
   outcome honestly — a bounded negative result is acceptable per AC #1.
2. If a failure fires: fix exactly what fired, born-red style. If quiet:
   Route B — mechanically convert the five one-shot
   `app.query_one(".stts-content").children[0]` isinstance asserts to
   `_wait_until` + `_stts_content_first_child` settles (both helpers
   already shipped by 19047 in this file). Wall-clock-bounded polls only;
   helpers return None mid-swap; no fixed pauses, no attempt-count waits.
   For the two playground steps, fold `#tts-generate-btn` existence into
   the settled condition (non-raising `app.query`) so the conversion
   cannot unmask the adjacent one-shot button asserts — 19047's
   catalogue-shapes lesson: the first raise masks everything behind it.
   Do not weaken anything: each settle ends asserting at least what the
   one-shot asserted, and the plain button asserts stay.
3. Evidence per AC #2: post-change target test standalone 10x under the
   same 20-burner load; full file once under load and once unloaded, all
   green with counts read from output files. ruff check + format on the
   touched file. Kill all burners, verify 0 remaining via pgrep.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A load-reproduction attempt (19047's CPU-burner loop methodology) against the current asserts is run FIRST and its outcome recorded — either a reproduced failure justifying the fix, or a bounded negative result; OR the five asserts are mechanically converted to `_wait_until` + `_stts_content_first_child` settles matching the file's existing pattern.
- [x] #2 If converted: the full file's load-loop evidence is re-run and recorded (not just the touched test), per 19047's catalogue-shapes-by-re-running lesson.
- [x] #3 The test's contract is unchanged: it still pins that each view switch mounts the expected pane type as the content's first child and that cycling back to playground restores `#tts-generate-btn`.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Route A first (AC #1): bounded negative result.** 15/15 standalone runs
of the target test passed under 20 CPU burners (busy-loop processes; 14
logical cores, ambient load avg ~7-8; run times 7.0-11.4s) at the
pre-change state — the five one-shots did not fire, matching 19047's
observation across all of its runs. No reproduced failure, so the fix is
the AC's sanctioned mechanical conversion, not born-red.

**Route B conversion.** The five one-shot
`app.query_one(".stts-content").children[0]` isinstance asserts
(initial playground → settings → audiobook → dictation → playground)
became `_wait_until(pilot, lambda: isinstance(_stts_content_first_child(
app), <Pane>))` settles — the exact helpers and pattern 19047 shipped in
this file (used at the dismissal test's settings-pane settle). All polls
are wall-clock-bounded (`_wait_until`'s 15s monotonic deadline); no fixed
pauses, no attempt-count waits; the helper returns None mid-swap instead
of raising. Nothing weakened: each settle ends asserting the same
first-child pane type the one-shot asserted (`_wait_until` raises
AssertionError on timeout).

**One deliberate strengthening, not a weakening:** the two playground
settles also fold `bool(app.query("#tts-generate-btn"))` (non-raising
query) into the settled condition. A bare first-child settle can return
at the instant the pane registers as a child but before its own compose
mounts descendants — which would leave the adjacent, in-contract
`assert app.query_one("#tts-generate-btn", Button)` one-shots sampling a
narrower version of the same window (19047's the-first-raise-masks-
what's-behind-it lesson). The plain button asserts stay, satisfying
AC #3's pinned contract (each view mounts its pane type as first child;
cycling back restores the generate button).

**Evidence (AC #2), final file state:** target standalone 10/10 green @
20 burners (7.7-9.1s); full file 163 passed @ 20 burners (181s); full
file 163 passed unloaded (70s). Counts read from captured output files.
ruff check + format clean on the touched file. All burners killed and
verified (pgrep: 0 remaining).

**No lessons entry:** the raising-predicate/empty-window class is already
recorded in `lessons-testing-evidence.md` (19047, 2026-08-20); nothing
new surfaced here.

**Files:** `Tests/UI/test_stts_profile_library.py` (only code change),
this task file.
<!-- SECTION:NOTES:END -->
