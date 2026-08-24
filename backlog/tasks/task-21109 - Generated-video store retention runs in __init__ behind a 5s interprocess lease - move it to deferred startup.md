---
id: TASK-21109
title: >-
  Generated-video store retention runs in __init__ behind a 5s interprocess lease - move it to deferred startup
status: To Do
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
