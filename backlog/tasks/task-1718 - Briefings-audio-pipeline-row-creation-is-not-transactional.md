---
id: TASK-1718
title: 'Briefings audio: pipeline row creation is not transactional'
status: In Progress
assignee: []
created_date: '2026-07-31 23:59'
labels:
  - watchlists
  - briefings
  - tts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`generate_script_audio` (`tldw_chatbook/Subscriptions/briefing_audio.py`) creates a
`briefing_audio` row via `db.create_briefing_audio` (its own `transaction()` block) and, on the
first in-band failure path (a voice that fails to resolve), finalizes that same row via a
separate `db.update_briefing_audio` call (its own, later `transaction()` block). The two writes
are not one transaction: a hard crash between them -- the process dying after the row is
inserted but before the first update lands -- leaves a `briefing_audio` row stuck in
`generating` with no worker left to finish it.

This is not a silent wedge: `fail_interrupted_audio` (the same zombie-sweep shape as
`briefing_cast.fail_interrupted_scripts`, TASK-1090's pattern) already sweeps orphaned
`generating` rows on both the Artifacts-load path and the next Synthesize attempt, flipping them
to `failed`/`"interrupted"` so the row surfaces honestly rather than staying invisible forever.
That self-healing is why this is filed as a low-priority follow-up rather than a blocking defect
of phase 2b (task-1630): the window is a narrow one (process death, not an in-band failure), and
its only consequence is a row that reads `failed: interrupted` instead of never having existed,
which is the same honest-degradation posture the rest of this pipeline already takes for every
other failure mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either `create_briefing_audio` and the row's first `update_briefing_audio` share one
      transaction (so a crash between them can no longer happen), or the non-atomicity is
      documented as an accepted trade-off in `briefing_audio.py`'s module docstring, naming
      `fail_interrupted_audio` as the self-healing path that recovers from it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both arms of AC#1, applied where each is correct. `create_briefing_audio` gained an optional `error`
param written in the SAME insert, so the two create-and-immediately-fail preflights
(`_record_voice_resolution_failure`, `_record_missing_pydub_failure`) now write their finished
`failed` row ATOMICALLY in one transaction — the create-then-separate-update window is gone for them
(they refuse before any synthesis, so there is nothing to do between the two writes). The MAIN
pipeline's create-`generating` → synthesize → update-`complete`/`failed` sequence is INHERENTLY two
transactions — a DB transaction cannot be held across the multi-second TTS synthesis, and the
`generating` row must be visible during it — so that non-atomicity is documented as the accepted
trade-off in the module docstring, naming `fail_interrupted_audio` (row-scoped per TASK-1890) as the
self-healing sweep that recovers an orphaned `generating` row. New DB test pins the atomic
create-with-error; mutation (drop `error` from the insert) reds both it AND the existing
voice-resolution helper test, proving the helpers now depend on the atomic path.

Files: `tldw_chatbook/DB/Subscriptions_DB.py`, `tldw_chatbook/Subscriptions/briefing_audio.py`,
`Tests/Subscriptions/test_briefing_audio_db.py`.
<!-- SECTION:NOTES:END -->
