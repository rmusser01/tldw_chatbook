# TASK-3401.18 Generated-Video Preview Crash Design

## Goal

Make explicit activation of a ready generated-video preview safe across decode,
pause/resume, end-of-file, unmount, and modal-player playback. A playback failure
must degrade inside the Console instead of escaping through Textual's worker and
terminating the app.

## Observed Failure and Root Cause

Live H3 UAT clicked the ready inline preview. The isolated application log then
recorded an unhandled `AttributeError` and app shutdown. The media and scratch log
were removed after recording sanitized evidence, so this design does not depend on
retaining user media, prompt text, paths, or host identity.

The source trace reproduces the structural cause without private evidence:

- `ConsoleVideoPreview._decode_loop()` calls `self.call_from_thread(...)` for
  frames and natural EOF.
- `VideoPlayerScreen._pump_loop()` and `_finish()` use the same call shape.
- Runtime inspection confirms neither `ConsoleVideoPreview` nor
  `VideoPlayerScreen` defines `call_from_thread`; Textual exposes the thread bridge
  on the owning application as `self.app.call_from_thread(...)`.
- In the inline preview, the first missing-method exception is caught by the
  decode-loop handler, but the `finally` block calls the same missing method again.
  That second `AttributeError` escapes the worker, matching the app-level UAT
  failure.
- In the modal player, the bad frame-dispatch call is caught and returns, silently
  dropping playback. Its EOF fallback then invokes a UI update directly from the
  worker thread, which is not a safe replacement for the missing bridge.

Existing unit tests replace `run_worker` with a no-op and exercise only direct state
transitions. They therefore never enter the thread-to-UI boundary that failed in
UAT.

## Decision

Repair both playback surfaces at the shared root-cause boundary:

1. Worker-thread UI callbacks use the owning Textual application's supported
   `self.app.call_from_thread(...)` bridge.
2. Inline source creation, open/probe, and decode all run in the activation worker,
   never in the click handler. The worker publishes source attachment, frames,
   failures, and EOF through the application bridge only when its widget generation
   is still current. If the widget has already paused, replayed, or unmounted, the
   worker closes its private source and exits without publishing.
   The successful UI-thread attachment creates the off-screen timer before returning
   permission for the worker to decode, so an immediate EOF cannot pause first and
   then leave a newly installed timer behind.
   The worker is the exclusive owner and closer of its PyAV source. UI-thread pause
   or unmount invalidates the generation, detaches the source reference, clears
   timer/active ownership, and signals cancellation; it never calls native decoder
   close across threads. The worker observes cancellation between frames and closes
   its own source in `finally`.
3. `AvFrameSource.iter_frames()` continues treating `GeneratorExit` as normal
   pause/stop, but unexpected decoder errors propagate to the preview owner. It no
   longer consumes those errors or logs raw exception text at the source seam; the
   preview owns the user transition and bounded diagnostic record.
   `AvFrameSource.close()` detaches/clears its internal container and stream
   references before closing the captured container, and propagates an unexpected
   close error to the worker owner for sanitized cleanup logging. A real native-close
   failure can therefore never leave a reusable half-closed source or disappear
   below the privacy boundary.
4. Each inline decode worker carries the play generation and source instance that
   created it. A late worker may pause/degrade only when both still match the
   widget's current generation/source. Pausing, replaying, or unmounting invalidates
   the old generation before closing it, so an old worker's `finally` block cannot
   pause a newly resumed preview.
5. Modal `ffprobe` and `PlayerPipeline.start()` run in a background activation
   worker, so the probe's 30-second timeout and process startup cannot freeze the UI.
   The worker publishes the started pipeline only when the modal activation token is
   still current; otherwise it stops/reaps the private pipeline and exits.
6. The modal player uses the application bridge for source attachment, frames,
   failures, and EOF. Every queued callback carries both pipeline identity and the
   pipeline generation captured by its worker and re-checks them on the UI thread.
   A seek, restart, dismissal, or unmount therefore invalidates already queued frames
   and EOF/failure callbacks before they mutate the screen.
   Seek restart and stop/reap are lifecycle-worker operations, not UI-thread calls.
   UI actions first invalidate/detach current playback state, then schedule the
   blocking control operation. Only one seek is allowed in flight for a modal; its
   completion publishes a restarted generation through the app bridge when still
   current and launches a new pump dedicated to that generation. The prior pump may
   exit or report a stale failure, but it cannot be the only pump for the restarted
   subprocesses. Unmount/failure cleanup is owned by an app-level worker so removing
   the screen cannot cancel the required stop/reap.
7. `PlayerPipeline.start()` itself owns transactional startup. Parent pipe descriptors
   close exactly once on every success/failure branch; a first- or second-spawn
   failure stops every child already created before re-raising. `stop()` terminates
   and waits, then kills and waits again when graceful termination times out, so no
   child remains unreaped. Silent media does not allocate an audio pipe at all.
   Each successful start/restart creates one per-generation run-state object holding
   that run's stdout, offset, clock, frame index, EOF, and render/drop statistics.
   The pump passes that exact object to `iter_frames()` before the generator's first
   advancement; the iterator never discovers stdout or counters from mutable
   pipeline-wide fields. Restart atomically replaces the pipeline's current-run
   reference but does not repurpose the old object. Frame timing/stat helpers also
   receive and update the originating run object. An old iterator may therefore
   finish a blocking read or update its own counters after restart, but it cannot
   consume the replacement pipe or mutate the new run's clock, frame index, EOF,
   offset interpretation, statistics, or first-frame PTS. No check-then-commit guard
   protects shared generation state because generation-owned state is not shared.
   The run object also owns an idempotent, thread-safe `close_stdout_once()` operation.
   Natural iterator completion closes it in `finally`; lifecycle stop/restart/unmount
   closes it after child termination, and startup failure closes it before the private
   run can be abandoned. Competing cleanup paths may call the operation, but the
   parent read descriptor closes exactly once, including when a created iterator was
   never advanced.
8. Expected capability absence keeps the existing install guidance. Unexpected
   playback failures show short actionable guidance that the inline preview stopped
   and that the dedicated Play action/system player remains available.
9. Diagnostic logs contain only a stable component (`frame_source`,
   `inline_preview`, or `modal_player`), phase (`open`, `decode`,
   `frame_dispatch`, `render`, `eof`, or `cleanup`), and exception class. They do not
   interpolate exception text, source paths, prompts, filenames, URLs, media bytes,
   or traceback locals.
10. A failed `app.call_from_thread` attempt is terminal for that callback. The worker
   emits one sanitized diagnostic and returns; it never retries the same broken bridge
   and never mutates a widget directly. UI-side failure handlers independently
   contain source/timer/pipeline cleanup plus notification/dismissal failures so an
   exception cannot travel back through the synchronous bridge and fail the worker.

This is a lifecycle correction, not a new playback architecture. PyAV remains the
inline decoder; the existing `PlayerPipeline` remains the modal audio/video path;
render modes, caps, one-active-preview policy, storage, and retention are unchanged.

## State and Lifecycle Behavior

### Inline preview

- `poster -> playing`: explicit click advances the generation and starts one
  activation/decode worker immediately; source open/probe remains off the UI thread.
- `playing -> paused`: explicit click, off-screen policy, or natural EOF stops the
  source and retains the last frame. The UI transition only invalidates/signals; the
  decode worker closes its source.
- `paused -> playing`: explicit click creates a fresh source using the existing
  behavior and resumes without blocking the UI.
- `playing -> unavailable`: source-open/probe or decode failure stops the source,
  clears active-preview ownership, and renders actionable guidance.
- `any -> unmounted`: timer/source cleanup is idempotent. Late worker callbacks may
  be refused by Textual but cannot escape or resurrect the widget.
- A generation/source match gates every frame, EOF, and failure callback. A stale
  worker exits without changing the state, source, timer, active registry, or frame
  owned by a later Play action.
- Timer stop and source close are independent, fail-contained cleanup steps. Each
  owned reference is cleared even if its cleanup call raises; timer stop runs on the
  UI thread, source close runs only in the owning worker, and only a sanitized
  `component=inline_preview phase=cleanup` diagnostic is emitted.
- The timer is installed as part of accepting the source generation on the UI thread,
  before decode begins. Immediate EOF therefore traverses normal cleanup and leaves
  no timer, source, or active-preview owner.

### Modal player

- Mount preflight and pipeline construction/start are guarded as one activation
  boundary in a background worker. A failure notifies and dismisses without blocking
  another UI action while probe/start is pending.
- Frames are scheduled onto the UI thread through the application bridge and render
  only when their pipeline identity/generation is still current.
- Pause/resume remain short UI-thread signal actions. Seek invalidates current
  callback identity immediately, then performs stop/restart in one background
  lifecycle worker; repeated seek input is ignored while that worker is pending.
  Successful seek completion attaches the restarted generation and starts its new
  pump before clearing the single-flight flag.
- Natural EOF schedules both the finished-state mutation and status refresh through
  the application bridge; no widget state is mutated directly from the worker.
- Unmount detaches/invalidates the pipeline immediately and schedules stop/reap on an
  app-owned lifecycle worker; late callbacks terminate quietly.
- An unexpected frame-iteration, clock, or dispatch failure is contained by the
  pump's outer boundary. When the modal is still mounted, one UI-thread failure
  handler stops/reaps the pipeline, shows generic recovery guidance, and dismisses;
  when the modal has gone away, the worker records only a sanitized diagnostic and
  returns.
- If pipeline `start()` raises after partially launching a child, activation calls
  `stop()` before clearing the pipeline and dismissing. Cleanup failure is contained
  and logged by component/phase/type without replacing the activation failure.
- A render conversion/widget-update failure is a playback failure, not a silently
  skipped frame. Inline and modal render handlers transition to their existing
  unavailable/recovery UI and emit only a sanitized `phase=render` diagnostic.
- If the application bridge itself refuses a late callback, the worker logs the
  bounded component/phase/type record once and exits. It does not attempt a second
  bridge call or direct UI fallback.

## Error and Privacy Contract

The user sees capability guidance for missing optional/system playback dependencies
and a generic recovery message for unexpected decoder/player failures. User-visible
copy does not include raw exception strings or local source paths.

Logs use a bounded shape such as:

```text
Console video playback failed: component=inline_preview phase=decode error_type=RuntimeError
```

The exception class is useful for diagnosis without retaining media identity. Full
tracebacks are intentionally omitted because they can contain local paths and
argument representations.

## Verification Design

Focused tests must exercise the boundaries the current tests bypass:

1. A mounted Textual/Pilot regression activates a ready inline preview through its
   real click path with a deterministic fake frame source. It observes one rendered
   frame, natural EOF, a responsive app, and no escaped worker exception. It uses the
   real Textual worker and `App.call_from_thread`; only the decoder source is replaced.
2. A deliberately blocked inline open/probe proves `pilot.click()` returns and a
   second UI action remains responsive before activation is released.
3. The same mounted path covers click-to-pause and click-to-resume without starting
   duplicate active workers or leaving stale active-preview ownership. A controlled
   old-worker/new-worker race proves the old generation cannot pause the resumed
   generation.
   A barrier-blocked decoder proves pause returns and a second UI action remains
   responsive while the worker still owns the source; after release, close runs on
   that worker and no stale state is published.
4. A focused `AvFrameSource` test makes the real decoder iterator raise an exception
   carrying a sensitive sentinel. It proves the exception reaches the preview owner
   and the source seam emits no raw exception text.
5. A real frame-source close test makes the captured container's `close()` raise a
   sensitive sentinel. It proves internal references clear before the exception
   propagates to the owner and that the source seam emits no raw exception text.
6. Source-open/probe and decode failures render guidance, release the source, and
   produce sanitized component/phase/error-type logs. Assertions reject prompt,
   path, filename, URL, and media content.
7. Unmount during decode proves a late frame/EOF callback cannot escape or update a
   removed widget.
8. Timer `stop()` and source `close()` are each made to raise a sensitive-sentinel
   exception. Focused tests prove neither escapes click/EOF/unmount, the sibling
   cleanup still runs in its designated thread, references are cleared, and logs
   contain only component, phase, and error type.
9. Deterministic immediate EOF proves the timer is installed before decode and that
   EOF clears timer/source/active ownership. A bridge-refusal test proves only one
   dispatch attempt occurs and no worker-thread UI fallback is called.
10. Inline and modal render conversion/update are each made to raise a
   sensitive-sentinel exception. Tests prove playback degrades, the exception does
   not escape, and logs contain `phase=render` plus error type but not the sentinel.
11. A blocked modal probe/start test proves the mount returns and another UI action
   remains responsive before activation completes.
12. Blocked seek and blocked stop/reap Pilot tests prove the initiating key/unmount
   transition returns immediately and another UI action remains responsive. They
   assert seek is single-flight, unmount cleanup is app-owned, and the required
   pipeline operation completes after its barrier is released.
13. A focused modal-player test supplies a deterministic fake pipeline, drives the
   real pump/EOF path, and asserts frames and finished status cross through the
   application's thread bridge. Controlled queued callbacks are drained only after
   a seek/restart and after unmount; old frames/EOF must be ignored. Removing either
   the production bridge or UI-thread identity check must make the test fail.
14. A controlled old-generation iterator failure is released after seek completion;
   the stale failure is ignored and the replacement pump renders a new-generation
   frame. Removing replacement-pump launch must make this test fail.
15. Real `PlayerPipeline` race tests pause an old pump after it captures its run but
   before the generator's first advancement, then restart playback. Advancing the old
   iterator must read only the old run's stdout; the replacement pipe remains unread
   until its new pump advances. Additional barriers release an old frame, its
   render/drop-stat update, and old EOF after restart. Those events may update only
   the old run object: the current run remains non-EOF until its own stream ends, its
   statistics remain unchanged, and its first frame has index zero and the new
   offset-derived PTS. Successful stop, restart, unmount, natural EOF, and a captured
   but never-advanced iterator each close their old run's stdout exactly once.
   Mutating the iterator or timing/stat helpers back to mutable pipeline-wide fields,
   or removing idempotent run cleanup, must fail these tests.
16. Real `PlayerPipeline` first-spawn and second-spawn failure tests prove every
   parent pipe descriptor closes and every successfully spawned child is terminated
   and reaped, process fields clear, and silent media never allocates an audio pipe.
   A forced terminate-timeout test proves kill is followed by wait. Failure after a
   private run/stdout exists but before publication proves that stdout also closes
   exactly once.
17. Screen-level pipeline-start failure proves the modal stops/reaps the private
   pipeline,
   dismisses with guidance, and launches no worker.
18. A mid-pump iterator/clock failure carrying a sensitive sentinel is contained by
   the real pump boundary. The test proves stop/reap, notification/dismissal when
   mounted, no escaped worker exception, and sanitized logging without the sentinel.

The mounted activation tests may patch the decoder/pipeline seams but must not stub
`run_worker`, call worker bodies directly, or replace `App.call_from_thread` on the
GREEN path. Mutating away the production bridge, generation checks, or timer-before-
decode ordering must make the corresponding test fail.

Only tests directly related to the touched frame-source/preview/player files are
run. No broad UI suite, full repository suite, live ComfyUI generation, or retained
media fixture is required.

## Alternatives Rejected

### Inline-only one-line replacement

It fixes the observed crash but leaves the modal player using the identical invalid
API. The modal currently catches that error and silently stops producing frames, so
the next full-player UAT would fail without an obvious exception.

### Catch and swallow `AttributeError`

This prevents shutdown but leaves playback nonfunctional and converts a deterministic
API misuse into a silent failure. It also does not address EOF's unsafe direct UI
fallback.

### New playback coordinator or shared abstraction

Both surfaces already have separate, appropriate pipelines and need only the same
supported Textual bridge plus lifecycle guards. A new coordinator would add state and
dependency surface without improving this repair.

## ADR Check

ADR required: no

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: ADR-044 already owns inline preview, modal playback, optional dependencies,
and ephemeral media lifecycle. This task corrects invalid Textual API usage and
failure containment without changing those decisions.
