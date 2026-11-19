---
id: TASK-21109
title: >-
  Generated-video store retention runs in __init__ behind a 5s interprocess lease - move it to deferred startup
status: Done
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - startup
  - video-generation
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21109).

`_build_generated_video_store()` (app.py:5676 -> 5322-5333) runs `VideoStore.enforce_retention()`
during `TldwCli.__init__`: mkdir, lock-file create, an EXCLUSIVE interprocess portalocker lease
with a 5.0 s poll-timeout (video_store.py:56, 191-233), then a scan+delete pass - all pre-paint.
A held lease from a concurrent instance (a deliberate, supported usage pattern) blocks boot for
up to 5 s.

## Acceptance Criteria

- [ ] App construction only resolves paths; `enforce_retention()` runs from `_schedule_deferred_startup_work` after first paint
- [ ] A probe holding the lease from a second process shows boot is no longer blocked
- [ ] Retention semantics (including the session default) unchanged; existing video-store tests green

## Re-verification against dev 2be18842a (2026-08-23)

An independent read-only pass re-checked this finding. **Mechanism still true; severity badly
over-billed; and the prescribed fix introduces a data-loss race unless guarded.**

**Confirmed**: `app.py:5737` (in `TldwCli.__init__`) → `app.py:5365-5375` constructs `VideoStore()`
and calls `enforce_retention()`. `video_store.py:57` sets a 5.0 s lease timeout;
`_root_lease()` (`:190-233`) spins a non-blocking exclusive lock at 10 ms intervals to that
deadline. Boot-only — nothing else calls `enforce_retention`.

**Cost corrected**: for a user who has never generated a video, `_snapshot()` returns empty
immediately and the whole pass is one `mkdir`, one lock-file create, one flock, one failed
`scandir`, one unlock — **sub-millisecond**. The 5 s stall requires a second instance holding the
lease at that exact moment, which only happens during its own sub-ms boot pass or an in-flight
save. `app.py:5369` already catches `VideoStoreBusyError`, so boot does not fail either way.

**The fix as written is unsafe**: the default retention mode is `"session"`, which means *delete
everything* (`video_store.py:795-800`). Moving the pass after first paint opens a window in which
a video published during this session is wiped by the session sweep, because the sweep has no
notion of process start. Any deferral MUST either capture a process-start timestamp and skip
files newer than it, or run before the video adapter can publish. Secondary: deferring means a
prior-session video marker can briefly resolve to a file that is about to vanish.

**Revised severity: low, bordering on not-worth-doing.** If something here is worth shipping, it
is the smaller honest win — stop creating `.generated_videos.capacity.lock` in every profile's
data dir at every boot for users who have never generated a video.

## Closure recommendation (2026-08-24, burn-down close-out)

**Recommend closing without work.** Sub-millisecond in practice: for a user with no videos the whole pass is one mkdir, one lock-file create, one flock, one failed scandir. The 5 s stall needs a second instance mid-save, and boot already catches VideoStoreBusyError. Worse, the prescribed deferral is unsafe -- the default retention mode is "session", which deletes everything, so moving the pass past first paint opens a window where a video published this session is wiped.

Left open rather than closed unilaterally: retiring a filed finding is the owner's call. The
evidence above is what a re-verification pass measured against dev before dispatch; if it is
accepted, close this as "retired on evidence" rather than "won't fix", because the mechanism was
real and only the cost or the prescribed fix was wrong.


## Implementation Plan (2026-10-11 dispatch)

The 2026-08-24 re-verification stands: the prescribed deferral past first
paint is UNSAFE (default retention mode "session" deletes everything, so a
deferred sweep can wipe a video published this session before the sweep's
process-start notion exists), and the billed 5 s stall needs a second
instance mid-save -- boot already catches VideoStoreBusyError. The board
kept the task LIVE past that recommendation, so what ships is the
re-verification's own "smaller honest win":

1. `enforce_retention` returns an empty report BEFORE taking the
   interprocess lease when the store root has never existed -- no
   `.generated_videos.capacity.lock` created in every profile's data dir
   at every boot for never-video users, no lease cycle, no dir creation.
2. Retention semantics for any profile that HAS a store are unchanged
   (boot still calls the pass; existing suite green).
3. Test: never-used store -> empty report, no lock file, no root dir.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Fix**: `Video_Generation/video_store.py` `enforce_retention` early-outs
  on `_ensure_safe_root() is None` (root absent). Safe by construction:
  orphan staging files live INSIDE the root (`_cleanup_orphan_stages_
  unlocked` already returns when the root is absent), `_snapshot` returns
  `()` for an absent root, and the save path `mkdir`s on demand -- nothing
  downstream depends on the boot pass creating anything.
- **Evidence**: `Tests/Video_Generation/test_video_store.py` (110 passed,
  2 skipped) including the new
  `test_boot_retention_creates_no_lock_file_for_a_never_used_store`.
  AC1/AC2 as originally written (deferral past first paint; probe-held
  lease) are deliberately NOT implemented -- the re-verification's
  data-loss race against session-mode retention stands, and this dispatch
  records that ruling in the task file rather than shipping the unsafe
  shape. If the owner wants full deferral later, it must carry a
  process-start timestamp guard or run before the video adapter can
  publish.
