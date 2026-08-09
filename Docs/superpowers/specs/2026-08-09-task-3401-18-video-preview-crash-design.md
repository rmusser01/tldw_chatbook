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
2. The inline preview handles source-open/probe, decode, EOF, and cleanup failures
   without allowing an exception to escape the worker. If the widget remains
   mounted, it transitions to a stable paused or unavailable state; if it has
   already unmounted, cleanup becomes a safe no-op.
3. `AvFrameSource.iter_frames()` continues treating `GeneratorExit` as normal
   pause/stop, but unexpected decoder errors propagate to the preview owner. It no
   longer consumes those errors or logs raw exception text at the source seam; the
   preview owns the user transition and bounded diagnostic record.
4. Each inline decode worker carries the play generation and source instance that
   created it. A late worker may pause/degrade only when both still match the
   widget's current generation/source. Pausing, replaying, or unmounting invalidates
   the old generation before closing it, so an old worker's `finally` block cannot
   pause a newly resumed preview.
5. The modal player uses the same application bridge for frames and EOF status. It
   never performs a direct widget update from the worker as a fallback. Pipeline
   setup failures notify the user and dismiss the modal without leaving a worker or
   child process behind.
6. Expected capability absence keeps the existing install guidance. Unexpected
   playback failures show short actionable guidance that the inline preview stopped
   and that the dedicated Play action/system player remains available.
7. Diagnostic logs contain only a stable component (`frame_source`,
   `inline_preview`, or `modal_player`), phase (`open`, `decode`,
   `frame_dispatch`, `eof`, or `cleanup`), and exception class. They do not
   interpolate exception text, source paths, prompts, filenames, URLs, media bytes,
   or traceback locals.

This is a lifecycle correction, not a new playback architecture. PyAV remains the
inline decoder; the existing `PlayerPipeline` remains the modal audio/video path;
render modes, caps, one-active-preview policy, storage, and retention are unchanged.

## State and Lifecycle Behavior

### Inline preview

- `poster -> playing`: explicit click opens the source and starts one decode worker.
- `playing -> paused`: explicit click, off-screen policy, or natural EOF stops the
  source and retains the last frame.
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
  owned reference is cleared even if its cleanup call raises; the other cleanup step
  still runs, and only a sanitized `component=inline_preview phase=cleanup`
  diagnostic is emitted.

### Modal player

- Mount preflight and pipeline construction/start are guarded as one activation
  boundary. A failure notifies and dismisses.
- Frames are scheduled onto the UI thread through the application bridge.
- Pause/resume and seek remain UI-thread actions against the existing pipeline.
- Natural EOF schedules both the finished-state mutation and status refresh through
  the application bridge; no widget state is mutated directly from the worker.
- Unmount stops the pipeline, and late worker callbacks terminate quietly.
- An unexpected frame-iteration, clock, or dispatch failure is contained by the
  pump's outer boundary. When the modal is still mounted, one UI-thread failure
  handler stops/reaps the pipeline, shows generic recovery guidance, and dismisses;
  when the modal has gone away, the worker records only a sanitized diagnostic and
  returns.
- If pipeline `start()` raises after partially launching a child, activation calls
  `stop()` before clearing the pipeline and dismissing. Cleanup failure is contained
  and logged by component/phase/type without replacing the activation failure.

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
   frame, natural EOF, a responsive app, and no escaped worker exception.
2. The same mounted path covers click-to-pause and click-to-resume without starting
   duplicate active workers or leaving stale active-preview ownership. A controlled
   old-worker/new-worker race proves the old generation cannot pause the resumed
   generation.
3. A focused `AvFrameSource` test makes the real decoder iterator raise an exception
   carrying a sensitive sentinel. It proves the exception reaches the preview owner
   and the source seam emits no raw exception text.
4. Source-open/probe and decode failures render guidance, release the source, and
   produce sanitized component/phase/error-type logs. Assertions reject prompt,
   path, filename, URL, and media content.
5. Unmount during decode proves a late frame/EOF callback cannot escape or update a
   removed widget.
6. Timer `stop()` and source `close()` are each made to raise a sensitive-sentinel
   exception. Focused tests prove neither escapes click/EOF/unmount, the sibling
   cleanup still runs, references are cleared, and logs contain only component,
   phase, and error type.
7. A focused modal-player test supplies a deterministic fake pipeline, drives the
   real pump/EOF path, and asserts frames and finished status cross through the
   application's thread bridge. Removing the production bridge must make the test
   fail.
8. Pipeline-start failure proves the modal stops/reaps any partially started
   pipeline,
   dismisses with guidance, and launches no worker.
9. A mid-pump iterator/clock failure carrying a sensitive sentinel is contained by
   the real pump boundary. The test proves stop/reap, notification/dismissal when
   mounted, no escaped worker exception, and sanitized logging without the sentinel.

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
