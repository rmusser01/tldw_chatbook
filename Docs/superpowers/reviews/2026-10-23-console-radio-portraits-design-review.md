# Console radio portraits: source-anchored design review

Date: 2026-10-23
Task: [TASK-36509](../../../backlog/tasks/task-36509%20-%20Design-Console-radio-portraits-with-audio-reactive-character-performance.md)
Design: [Revised spec](../specs/2026-10-23-console-radio-portraits-design.md)
Decision: [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md)
Reviewed design baseline: `8447cb870d`

What this change does: The design adds optional six-pose performances to existing
character expressions, with image-edit generation/import and radio framing. Mouth
movement follows owned TTS output while blinking and emotional selection remain
independent. This review assumes the existing single-user desktop application,
multiple open conversations and editable local profiles, not a new hosted service.

All six findings below are corrected in the revised spec and ADR where relevant.
These are implementation requirements, not claims of fixed product code.

## Must fix

### 1. Capture a persisted local source, not just a saved character ID

**What this is:** Preparation captures a character and its source portrait, then
later publishes six frames. Original spec lines 117-124 required a saved character
but did not settle dirty artwork or local/server/profile authority.

**Problem:** `personas_character_editor_widget.py:955` stages an avatar until Save.
A saved character can therefore show a new portrait that editor Cancel will discard.
Preparing against it spends requests and produces an attachment that cannot match
the active source. Separately, `personas_screen.py:473` treats server characters as
read-only, and existing visual author snapshots capture the actual database and
binding. A numeric ID can coincide across server/local authorities or profiles.

**Fix:** Prepare persisted artwork only, require explicit Save/Apply for dirty
artwork/bindings, and revalidate existing local-action plus exact service/database
authority before submissions and publication. Unrelated text edits need not block.

**If skipped:** Artwork can be prepared for a discarded portrait or attached to a
different local record after an authority switch. Include dirty-source Cancel,
same-ID server/local and profile-replacement controls in implementation tests.

### 2. Define hands-free, preview and pause integration explicitly

**What this is:** Original spec lines 127-131 and 219-243 assumed a common owned
speech presentation path for message Speak, hands-free sentences and editor samples.

**Problem:** `hands_free.py:2153` calls `speak_utterance` with completion only;
`tts_events.py:1072` creates a synthetic ID and has no playback observer argument.
Manual Speak instead builds a trusted completed-message snapshot in
`message.py:1080`. The preview has no such message. `TTSPlaybackLifecycle` and
`message.py:1299` also accept no paused state. Reusing these paths as if their
interfaces already matched would miss starts or fail admission/state transitions.

**Fix:** Specify an optional hands-free presentation owner/observer while retaining
completion/cooldown/voice behavior. Reuse `speak_guarded_utterance` for editor
samples with destination authority and exact-owner cancellation. Report confirmed
pause in the presentation snapshot without changing closed admission lifecycles.

**If skipped:** Hands-free portraits remain idle, preview speech is refused or
unscoped, and late preview cleanup can silence newer audio. Qualify all three paths
and leave their existing callbacks and consumers compatible.

### 3. Applying or removing a set must invalidate same-source caches

**What this is:** Performance frames change independently of the original portrait.
Original spec lines 170-176 and 300-304 described publication/source invalidation
but omitted the performance revision from visible request/cache identity.

**Problem:** `character.py:581` returns early for an unchanged avatar request;
`:652` also reuses a spec by the original resolution's cache identity. Applying
new mouth frames without changing actor, source bytes, mood or motion preferences
can therefore keep the old presentation until another unrelated change occurs.

**Fix:** Include local authority and performance revision/absence in request,
prepared-frame and spec keys. Publish Apply/Remove invalidation after commit and
reject late work from replaced revisions, without database reads on the frame path.

**If skipped:** Apply appears ineffective, Remove can leave stale animation visible,
and a slow old decoder can overwrite the replacement. Test mounted same-source
Apply/Remove and an intentionally delayed predecessor worker.

### 4. Separate the source clock from the device-output clock

**What this is:** Original spec lines 246-257 offered one monotonic output position
and an audio envelope, with underrun accounting left implicit.

**Problem:** A 500 ms underrun at 48 kHz advances device output by 24,000 silent
frames while consuming zero source frames. Indexing the source envelope with that
single output position skips half a second of resumed mouth motion. The existing
`streaming_sink.py:750` callback ignores its backend time argument, and partial
blocks zero-pad their suffix without the full-empty-block underrun event.

**Fix:** Specify source-to-device range mapping, distinct frame counters, validated
output-DAC/monotonic timing, explicit partial-block silence and generation resets.
Bound metadata to queued audio plus a device-tail reserve, retain each window until
its scheduled output has actually passed, and coalesce latest-only reader snapshots.
If timing is invalid, use the status-only fallback; presentation must not block audio.

**If skipped:** Animation can lead audible speech or drift after buffering, and
event backlog can grow for hidden/slow views. Test delayed DAC output, complete and
partial underruns, resumed audio, invalid timestamps and a stalled UI observer.

### 5. Release smoothing conflicts with the stated silence-response target

**What this is:** Original spec lines 269-279 combined a 100 ms release filter,
-45 dBFS closing threshold, downward hysteresis and a 100 ms visible-response target.

**Problem:** The smoothing domain and silence override were unspecified except
for exact digital zero. An allowed exponential implementation decaying from -12
to -60 dBFS crosses the hysteretic -48 dBFS boundary after 138.6 ms when smoothing
dBFS, or 443.0 ms when smoothing amplitude. Windowing and painting add more delay.
Ordinary quiet background is not digital zero, so a zero-only test would miss it.

**Fix:** Close after two raw 20 ms windows below the silence floor, bypassing
smoothing; force close immediately for digital zero, underrun, pause and terminal
states. Specify smoothing in dBFS and small/open hysteresis separately. Test the
complete window-to-paint response on a nonzero background against the 100 ms target.

**If skipped:** The character keeps talking through natural pauses despite passing
a perfectly silent test clip. Numerical tuning still needs real quiet/loud samples.

## Should fix

### 6. The preview must reveal six-frame alignment without requiring TTS

**What this is:** Original spec lines 127-131 offered idle/blink playback, frame
inspection and a live TTS sample, but no offline mouth/blink transition preview.

**Problem:** Idle preview exercises only two of six roles. Without configured TTS,
an imported set may pass equal-dimension checks while its open-mouth/blink frames
visibly jump. Existing `expression_generation.py:24` supplies a speaking-image
direction, not a facial rig or an alignment guarantee for these new variants.

**Fix:** Require a deterministic six-pose preview at the actual Console image size
and source comparison. Label the generation batch as six requests, retain successful
slots, and keep selected-slot regeneration to one explicit request. This makes
quality and the amount of provider work reviewable before activation/retry.

**If skipped:** Users can pay for several rounds of artwork and discover alignment
problems only after Apply. Real-provider quality remains a qualification requirement.

## Verification and verdict

Source inspection covered editor staging/local authority, immutable visual
resolution, avatar request/spec/paint caches, message snapshots, manual Speak,
hands-free completion, guarded app-owned speech, lifecycle transition validation,
PCM output, partial silence and external player clock limitations. The review used
stdlib calculations to reproduce the two release delays and underrun frame counts;
these demonstrate spec ambiguity, not a measured defect in unimplemented code.

Verdict: The six corrections above are incorporated. The user acknowledged the
revised design on 2026-10-23 and requested the [implementation plan](../plans/2026-10-23-console-radio-portraits-implementation.md),
which records nine atomic Backlog tasks. This acknowledgement closes the design
review, not product implementation or live qualification.

Not checked: No image-edit provider, generated-frame consistency, real audio device,
audible timing, app launch or product test suite was exercised. Those remain future
implementation qualification; the existing unrelated Backlog ID collisions remain.
