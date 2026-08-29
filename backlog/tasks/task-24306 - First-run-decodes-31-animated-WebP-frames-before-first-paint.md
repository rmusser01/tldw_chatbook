---
id: TASK-24306
title: >-
  First run decodes 31 animated WebP frames before the UI appears
status: To Do
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
- [ ] #1 A fresh-profile first launch reaches first paint materially faster than 1.43 s
- [ ] #2 Validation of package-shipped actor-pack image bytes does not decode animation frames at install time on the user's machine
- [ ] #3 Steady-state boot cost against an existing profile is unchanged or better, measured before and after
- [ ] #4 The seeding work does not block first paint
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Find the seam for package-shipped assets.
2. Cut the frame decode.
3. Pin both the saving and the behaviour it must not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

**NOT DONE -- the fix was implemented, found to be a regression, and REVERTED.**
The finding stands; the approach was wrong. Recorded here so the next attempt
does not repeat it.

**The filed premise was wrong first.** The task said "31 animated WebP frames".
The bundled pack is 31 STILL WebP files; PIL routes even single-frame WebP
through `WebPAnimDecoder`, which is what the profile's 31 `get_next` calls
showed. An anti-vacuity assertion in the first test (`assert animated_seen`)
caught that.

**Then the fix was wrong.** `_inspect_image_bytes` computed a duration
unconditionally and discarded it for still images on the next line, so guarding
it on `is_animated` looked free. It is not: **`_image_duration_ms` is the only
caller of `image.load()` on that path, and `Image.open()` reads the header
without decoding the payload.** With the guard in place, `_inspect_image_bytes`
ACCEPTS bytes whose real decode raises `OSError` -- verified directly with
interior payload corruption at an intact container header and length:

    intact        : accepted   | real decode: OK
    mid-payload   : ACCEPTED   | real decode: FAILS -> OSError
    late-payload  : ACCEPTED   | real decode: FAILS -> OSError

Truncation is still caught by `Image.open`, which is why this was not obvious.

`Tests/Character_Chat/test_visual_identity_assets.py::test_complete_validation_
rechecks_cumulative_actual_decoded_work` is the repository's existing guard for
exactly this, and it went red. The word "actual" in its name is load-bearing.
**My own new test passed** -- it pinned the optimisation (no frames decoded)
without checking the invariant the decode was upholding, which is the more
useful half of this lesson.

The measured saving was real (first-run `__init__` 0.50 s -> 0.29 s, four
interleaved pairs) and came *from* removing that validation, so there is no
version of this approach that keeps both. A single `image.load()` per still
asset is what the old code already did.

**What the next attempt should do instead:** move the built-in pack seeding off
the first-paint critical path rather than weakening what it checks. There is
precedent -- `deferred_actor_pack_recovery` and `deferred_actor_pack_staging_
sweep` are already on the boot worker allowlist (task-21106). `seed_builtin_
content` is currently called synchronously from `get_chachanotes_db_lazy`, so
decoupling it from the DB accessor is the real work, and it is a larger change
than this cycle should bolt on.

An in-code comment at the guard site records why it must not be "optimised"
again without keeping a decode.

Files touched then reverted: `Character_Chat/visual_identity.py` (comment
retained), `Tests/Character_Chat/test_builtin_pack_frame_durations.py` (deleted).
<!-- SECTION:NOTES:END -->
