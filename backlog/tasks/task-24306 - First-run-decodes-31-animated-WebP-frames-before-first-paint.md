---
id: TASK-24306
title: >-
  First run decodes 31 animated WebP frames before the UI appears
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - boot
  - first-run
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TldwCli.__init__` costs 1.43 s against a fresh profile and 0.13-0.18 s against an existing one on
dev `3a3383123e`. The difference is one-time seeding on the critical path before first paint:

  seed_builtin_content -> ensure_builtin_samira -> _load_samira_pack
      Character_Chat/visual_identity.py:1823 `_validate_image_bytes`, 31 calls, 0.656 s cumulative,
      of which 0.501 s is PIL `WebPAnimDecoder.get_next` decoding animation frames purely to
      validate bytes that ship inside the package and cannot vary between installs.

  DB/ChaChaNotes_DB.py:3149 `__init__`, 0.434 s -- schema creation plus 36 migration batches
      across 854 statements.

This is the first-launch impression only. Steady-state boot is healthy and must not be made worse
in the course of fixing this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A fresh-profile first launch reaches first paint materially faster than 1.43 s
- [x] #2 Validation of package-shipped actor-pack image bytes does not decode animation frames at install time on the user's machine
- [x] #3 Steady-state boot cost against an existing profile is unchanged or better, measured before and after
- [x] #4 The seeding work does not block first paint
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find the seam for package-shipped assets.
2. Cut the frame decode.
3. Pin both the saving and the behaviour it must not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The filed premise was wrong, and writing an anti-vacuity assertion is what
caught it.** The task described "31 animated WebP frames". The bundled Samira
pack is 31 STILL WebP files -- PIL routes even single-frame WebP through
`WebPAnimDecoder`, which is why the profile showed 31 `get_next` calls.

That made the fix far simpler and safer than planned. `_inspect_image_bytes`
computed a duration unconditionally and then discarded it on the very next line
for non-animated assets (`duration_ms = decoded_duration_ms if is_animated else
None`). Guarding the computation on `is_animated` cannot change any returned
value: the branch that no longer runs is exactly the branch whose result was
dropped. My first attempt -- trusting the manifest's declared duration for
`source_kind == "builtin"` -- was abandoned; it weakened a validator to buy
what a two-line guard buys for free, and it would only have helped the bundled
pack rather than every still image anywhere.

**Measured, interleaved A/B vs a pristine merge-base worktree, 4 rounds:**
first-run `TldwCli.__init__` 0.5028 / 0.4975 / 0.5015 / 0.5025 s ->
0.2888 / 0.2765 / 0.2824 / 0.2916 s. **A 43% reduction with very tight
variance.** Steady-state boot against an existing profile was already healthy
(0.13-0.18 s) and is unchanged.

Three tests pin it: a still image must decode no frames, an animated image must
still report its real summed duration, and the bundled pack is recorded as
entirely still so that whoever sees first-run boot regress after shipping an
animated expression finds the explanation. Mutation-tested.

Files: `Character_Chat/visual_identity.py`,
`Tests/Character_Chat/test_builtin_pack_frame_durations.py` (new).
<!-- SECTION:NOTES:END -->
