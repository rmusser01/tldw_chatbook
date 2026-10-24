# Console radio portraits and audio-reactive character performance

Date: 2026-10-23
Status: Accepted design; revised spec acknowledged 2026-10-23
Revision: 2026-10-23 source-anchored review corrections; see the [review](../reviews/2026-10-23-console-radio-portraits-design-review.md)
Task: [TASK-36509](../../../backlog/tasks/task-36509%20-%20Design-Console-radio-portraits-with-audio-reactive-character-performance.md)
Decision: [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md)
Implementation: [Plan and nine atomic Backlog tasks](../plans/2026-10-23-console-radio-portraits-implementation.md)

## Purpose and first-version scope

Make the Console's left-sidebar character feel present through a living portrait
and restrained radio-screen presentation, inspired by the Metal Gear Solid Codec.
Combine audio-reactive mouth movement with pauses, blinking, subtle idle motion,
and the existing emotional expressions. Support both generating artwork from an
existing portrait and importing ready-made frames.

This is a character portrait feature in the existing Character section, not a new
destination. The first version includes the editor preparation workflow, durable
local artwork, TTS integration, radio framing, and verification of the complete
editor-to-Console path. Implementation may be split into independently testable
tasks during planning; the complete feature includes all these outcomes.

Excluded: phoneme/viseme alignment, facial landmarks or rigs, arbitrary-image
warping, 3D/video avatars, microphone-driven animation, realtime voice engines,
contact/call menus, paired user portraits, sound effects, and new image providers.
No copyrighted game artwork or audio is bundled.

## Existing foundations and boundaries

The source inspection for this design used local dev HEAD `b07fc04252`. These are
existing seams, not claims that the new behavior has been implemented:

| Concern | Existing owner |
| --- | --- |
| Expression/manual/historical precedence | `tldw_chatbook/Chat/console_expression_state.py`, `UI/Console_Modules/character.py` |
| Disposable portrait playback and bounded preparation | `Widgets/Console/character_expression_avatar.py`, `Chat/character_expression_playback.py` |
| Verified expression assets and immutable pack versions | `Character_Chat/visual_identity.py` |
| Character editing and expression generation | `UI/Screens/personas_screen.py`, `Widgets/Persona_Widgets/personas_character_editor_widget.py`, `Character_Chat/expression_generation.py` |
| Image-reference capability and request validation | `Image_Generation/capabilities.py`, `Image_Generation/worker.py` |
| Speech ownership and start/stop/failure | `Event_Handlers/TTS_Events/tts_events.py`, `Chat/console_speech.py` |
| Manual/automatic Speak integration | `UI/Console_Modules/message.py` |
| Hands-free utterance integration | `UI/Console_Modules/hands_free.py`, `Chat/reply_sentence_sequencer.py` |
| Streaming output and format eligibility | `Audio/streaming_sink.py`, `TTS/pcm_stream.py` |
| External file playback | `TTS/audio_player.py` |
| Character rail geometry and full-size viewer | `UI/Screens/chat_screen.py`, `Widgets/Console/console_image_viewer_modal.py` |

`speaking` currently follows text-stream status in the expression resolver. That
can remain a legacy expression-selection input, but it must not open the new
audio-reactive mouth. Only playback belonging to the displayed character does so.
Do not add a second sentiment classifier, conversation store, speech player, or
floating Buddy runtime.

Relevant existing decisions are ADR-074 (visual identities and actor packs),
ADR-083 (rail ownership), ADR-144 (expression playback), ADR-150 (design tokens),
and ADR-161 (component patterns). Their full paths are linked in ADR-242.

## Radio-screen presentation

Keep the existing Character section and its controls. Within its measured image
budget, present a stable recessed portrait frame, character identification, a
compact voice-status label, and one small horizontal audio-level meter. The
identification/status and meter rows reserve their space even at rest; speech must
not resize the panel, shift controls, or move the outer rail.

Use theme-derived surfaces, borders and status colors. Preserve the portrait's
authored colors; do not force a green tint. Transmitting may illuminate the
existing frame without changing its dimensions. A single brief, low-contrast
transmission effect may mark playback start or interruption, using the existing
fast motion token. Do not use continuous static, flashing, fake reception bars,
decorative frequencies, or a connection label implying an unobserved network state.

The meter represents the same measured audio envelope as the mouth. It is empty
when silent and suppressed when timing is unavailable. Text status remains the
accessible source of meaning; color and animation are supplemental. Compact
layouts may omit the meter before sacrificing portrait controls. Preserve the
existing aspect-preserving contain fit, rail scrolling, and full-size-on-click
viewer. The viewer snapshots the currently displayed frame without owning another
animation or speech clock.

In ordinary layouts suppression uses visibility hiding, preserving the meter's
row and image/control geometry. Only compact layout removes the row from layout;
voice state and timing availability cannot change that layout decision.

All geometry, color, state and motion values compose from the design language and
component patterns. Add any missing `$ds-*` values centrally before use, edit only
source stylesheets, and rebuild the generated TCSS bundle. New controls remain
keyboard accessible and follow the existing keybinding contract.

## Artwork: six poses for each expression

Mood selection and speech motion are independent. A performance set attaches to
one character, one resolved expression and the exact source-image digest; its six
full-canvas still frames are:

| Role | Eyes | Mouth |
| --- | --- | --- |
| `rest` | Open | Closed |
| `small` | Open | Slightly open |
| `open` | Open | Open |
| `blink_rest` | Closed | Closed |
| `blink_small` | Closed | Slightly open |
| `blink_open` | Closed | Open |

The frames share pixel dimensions, framing, background and pose. Full-frame
selection keeps the first version usable without a layering or landmark editor.
Equal dimensions are validated automatically; facial alignment and identity
consistency must be assessed in the animated preview.

Prepare only the selected expression, not an automatic batch of every mood.
Legacy plain portraits use the neutral expression identity. Legacy expression
images and Shared Visual Identity expressions are both eligible. A mood without a
matching performance set retains its existing artwork/playback; do not substitute
an unrelated neutral talking face. Radio status and a trustworthy meter can still
work. A matching performance set replaces that expression's encoded animation
only while this new performance is effectively enabled.

During owned speech, an expression without a usable set displays its existing
static poster rather than running an uncontrolled encoded talking loop. Such
artwork can itself depict an open mouth; without prepared frames no synthetic
mouth closure is promised. Silence/closure rules below describe prepared sets.

## Prepare portrait performance

Expose the action beside artwork controls in the existing character editor. This
first version prepares only saved, locally owned characters in the active local
profile. Reuse the existing local-action/authority capture and revalidation gates;
an integer character ID is not sufficient authority. Server-owned characters stay
read-only, and profile/service replacement invalidates the whole draft and its
Console cache. Revalidate authority when admitting each new provider request;
already admitted work follows the cancellation caveat below.

Preparation uses persisted, currently active artwork. If the editor has an unsaved
replacement/removal of that portrait or an unpublished visual binding, require
Save/Apply of that source first; never save it implicitly. Unrelated dirty text
fields need not block preparation. This avoids spending generation requests on a
portrait whose subsequent editor Cancel would discard the source. The preparation
view shows the captured source image and expression, six named pose slots, a live
preview, and Generate, Import/Replace, Regenerate selected, Apply and Cancel
actions using existing component conventions.

1. Capture the exact local profile/service/database authority, character/editor
   identity, resolved expression, persisted source bytes/hash, active visual
   binding/version where present, and current performance revision. Keep the
   original source digest as the attachment key. For an animated source, extract
   its first fully composited frame through existing bounded validation to obtain
   one still reference for generation; do not replace or rehash the original asset.
2. Generate six poses from the captured portrait, or assign individual ready-made
   images to the six slots. Both paths populate the same disposable draft.
3. Inspect individual frames and a deterministic six-pose cycle at the actual
   Console image-box size, with a toggle back to the source for comparison. This
   offline preview exercises mouth/blink combinations without TTS and exposes
   background or face-position jumps; equal pixel dimensions alone do not prove
   alignment. Also provide the normal idle/blink preview. An explicit
   Play TTS sample action speaks a short fixed preview phrase through the selected
   character's existing TTS resolution and playback owner. It does not create a
   conversation message or trigger automatic replies. Use the existing guarded
   app-owned utterance seam (`speak_guarded_utterance`), selected-character voice
   resolution and destination-authority checks; do not fabricate a trusted
   message snapshot. Expose only this preview's presentation owner. Closing,
   cancelling, changing source or superseding the preview stops that exact owner
   through the existing scoped stop path, never a bare global Stop that could
   silence a newer request. It does not speak on opening the view, and unavailable
   TTS does not prevent importing or applying artwork.
4. Apply validates and atomically publishes all six frames. The existing artwork
   stays active throughout preparation. Cancel discards the draft.

Generate uses the configured image-generation service with image-reference/edit
capability. Show the chosen provider/model and that the source portrait will be
sent there before the explicit Generate action. Do not fall back to unrelated
text-only generation or silently select another provider. Reuse engine request,
output, size, path and secret-handling validation. No API keys, source bytes or
reference paths enter routine logs.

The batch action says Generate 6 frames and discloses six image-edit requests,
plus the selected destination, before submission. Regenerate selected submits one
request. Do not silently regenerate successful draft slots or automatically retry
a paid request after an uncertain provider outcome.

Each request references the captured still source and asks to preserve identity,
framing, pose, background and lighting, changing only mouth and eyelids for the
named role. Dispatch sequentially and retain successful draft slots when another
fails. Regeneration replaces only the chosen draft slot; it never publishes early.
Check the editor/source identity after every asynchronous completion. Switching
characters or closing the draft cancels remaining submissions and discards late
results, even if an already submitted provider job cannot be cancelled remotely.

Imports accept six individually assigned still PNG, WebP or JPEG images through
the existing file picker/path validation. No new ZIP/sprite-sheet format is needed.
Reject animated slot inputs, missing/duplicate roles, incompatible dimensions,
malformed images and decompression bombs. Normalize accepted draft frames to still
lossless PNG before preview and Apply, and validate the normalized output too.
Do not silently crop or independently stretch images to hide a mismatch.

A complete set has exactly six roles, at most 1,048,576 pixels per frame, at most
4 MiB encoded per frame, and at most 16 MiB encoded in total. Stricter applicable
existing validator/configuration limits win. Image work runs off the UI thread
and participates in bounded preparation accounting. Failed conversion releases
decoded buffers. Apply is unavailable until the entire set validates.

Changing the source image or binding during review makes the draft stale. Apply
must revalidate captured identity and expected revisions; it cannot retarget the
draft to the new source or overwrite a newer edit. The user explicitly starts a
new preparation against the new source.

## Durable ownership and compatibility

Store performance attachments locally in the main character SQLite database,
using a small parent record plus six frame records. The parent owns character ID,
normalized expression identity, source SHA-256, schema version and optimistic
revision. Each child owns one unique role, validated PNG bytes, dimensions and
digest. One active set per character/expression is sufficient. Replacing it checks
the expected revision and publishes the parent and all six frames in one existing
database transaction. Do not write frame files beside immutable pack assets.

Apply and Remove publish a local performance-revision invalidation only after the
database transaction commits. Include the local authority, source identity/hash
and performance revision (including an explicit absent-set identity) in the
Console request/spec/preparation cache keys. A same-source replacement must become
visible without changing session, reaction or portrait bytes. A failed transaction
does not invalidate the visible set. Removed/replaced sets reject late preparation
results and paints from their predecessor; no frame path queries SQLite.

This is a schema change: allocate the next live schema version during
implementation, increment `_CURRENT_SCHEMA_VERSION`, and add an executed SQL
migration under `DB/migrations/`. Do not reserve a migration number in this spec.
Source/hash mismatch disables the attachment and falls back to the selected
ordinary expression; it never activates stale performance for another portrait.
Removal of a performance returns to existing artwork. Character soft deletion
preserves its performance for restoration; permanent deletion removes its rows.

First-version performance data is local to this character/profile. The preparation
view says so. Existing character-card, Actor Pack and Chatbook exports retain their
ordinary artwork and formats; they do not silently gain a new portable extension.
Importing or copying a character creates no cross-character performance binding.
Full-profile backup/restore must preserve the new SQLite data and safely invalidate
old in-memory playback owners after restore. Cross-device export of performance
sets is outside this version; neither UI nor documentation may imply otherwise.

This local attachment avoids changing the existing visual-pack manifest schema,
canonical mood catalog or immutable content digest. Reuse source resolution,
validation, optimistic locking and backup admission rather than creating another
asset authority or permission model.

## Playback state and expression precedence

| Observed state | Portrait | Radio status |
| --- | --- | --- |
| No owned speech | Mouth closed; blink and idle motion when enabled | Ready |
| Text streaming without a speech request | Existing expression selection; no audio-driven mouth | Ready |
| Owned TTS generating/queued | Current expression, closed mouth | Preparing voice |
| Owned audio actually playing | Mouth follows output envelope; blink/idle continue | Transmitting |
| Silence within playback | Mouth closed; no invented activity | Transmitting; empty meter |
| Output underrun | Mouth closed until real samples resume | Transmitting; empty meter |
| Confirmed pause | Mouth closed | Paused |
| Natural finish or deliberate stop | Immediately close mouth; retain selected mood | Ready |
| Generation/playback failure | Close mouth; retain selected mood | Voice unavailable, with existing error detail access |
| Playing without trustworthy timing | Existing portrait; no audio-driven mouth or meter | Transmitting, plus concise timing-unavailable explanation |

An empty audio buffer is not by itself an end-of-utterance signal. Terminal events
override smoothing; closing does not wait for another text chunk or a release ramp.
Only confirmed player pause/resume events change Paused status. Never offer resume
semantics more precise than the actual player supports.

The present `TTSPlaybackLifecycle` and Console message presentation accept
generating/playing/stopped/failed, not paused. Do not feed a new paused string into
those closed state machines. Carry confirmed transport pause/resume and progress
in the optional presentation snapshot while keeping the existing completion and
speech-admission state machines intact. No new pause/resume UI or transport
capability is required by this feature.

Manual Speak, automatic reply speech, and hands-free TTS use the same presentation
contract. A hands-free reply has request ownership plus sequential utterance
identity, so one sentence's late terminal signal cannot stop the next sentence's
portrait. Preparing the next sentence cannot hide an already audible sentence.

Hands-free `speak_utterance` currently accepts text plus `on_finished` and creates
its own synthetic ID; it does not receive the message-owned playback lifecycle.
Extend that entry and the existing generation/sink/file paths with an optional
presentation owner/observer, captured by the hands-free caller from the exact
reply, session, character authority and sequencer token. Report real start and
terminal/progress through it, preserving cooldown bypass, quiet/failure handling,
one-voice displacement, current voice/provider selection, and exactly-once
`on_finished`. Do not reroute partial sentences through Manual Speak, whose
trusted message snapshot requires a completed speakable message. Terminal cleanup
still settles the exact owner even after its visible-session validator is stale.

Preserve current manual and explicit/historical expression precedence. A manual
reaction changes the mood without stopping audio; use its matching performance if
available, otherwise ordinary artwork. Replaying an older message temporarily uses
its existing durable expression identity when valid and from the displayed
character, then restores the current mood. A current manual reaction still takes
precedence. Unknown historical actor provenance does not establish a past mood;
use the current resolved expression instead. Missing historical assets use
existing resolver fallback; playback must not resurrect a removed pack or invent
history.

Bind presentation updates to the existing speech request/message, session and
epoch, character identity, utterance identity where needed, and playback generation.
The portrait subscribes to the current owner; it does not own audio. Session or
character changes discard the previous visual subscription immediately. Old audio
may continue or stop according to existing speech policy, but it cannot animate
the new character. Replay of a message attributable to another character is
status-only. Preview playback is scoped to its preparation view and cannot animate
an unrelated Console portrait. Supersession, hide/show and unmount reject stale
updates and release visual resources without adding an independent audio-stop rule
for Console-owned speech. The disposable editor sample is the scoped-stop exception
described above.

## Audio timing and mouth selection

Derive the mouth from the audio being played, not text, estimated text duration,
network chunk arrival or time since synthesis started. Extend the existing playback
boundary with an ephemeral, coalesced presentation snapshot: owner identity,
playback generation, confirmed transport state, output-timed envelope level and
timing validity. Keep separate counters for source frames consumed and device
frames presented: an output underrun advances the device clock with inserted
silence but consumes no source audio. The producer maps source-envelope windows
to device-output ranges; the widget receives the level for the range actually
audible now, not a source position advanced by wall-clock underrun duration.
Include a monotonic sample/publication time for freshness, and invalidate the
timing map on transport restart, discontinuity or generation replacement. PCM
buffers and audio contents do not cross into widgets, message metadata, logs or
the database.

For the existing PCM sink, prepare bounded energy windows alongside queued audio
outside the realtime callback. The callback exposes only minimal consumed-frame
and output-timing state; it does no image, database, UI, blocking queue or provider
work. Presentation follows output consumption and accounts for inserted underrun
silence and available device timing. The callback currently ignores its backend
time argument; use the available output-DAC timestamp with a validated mapping
to the presentation monotonic clock instead of equating buffer fill with audible
output. Partial blocks must map their zero-padded suffix to silence even if no
public SinkUnderrun event was emitted. Missing/invalid device timing makes the
mouth timing unavailable, rather than silently substituting callback wall time.
Do not fire a UI event for every sample.

Use the sink's existing non-realtime notification thread to sample presentation
at 20 ms intervals while observation is active, independent of public sink events
or widget reads. Callback progress replaces bounded shared state; publication
coalesces into one latest snapshot. Periodic publication cannot manufacture timing
validity: evaluate only already-mapped device intervals, never extrapolate speech
beyond them. The last mapped interval may remain valid during native device drain;
once it has passed without new timing, close the mouth and mark timing unavailable.

Natural completion is distinct from interruption. After the final source block is
submitted, keep playback ownership and the sink alive until the backend confirms
all pending output has played. Reuse sounddevice's graceful callback completion
and output-finished acknowledgement, with teardown off the realtime callback.
Only then publish successful drain, release the voice owner and complete the
utterance or advance hands-free to its next sentence/listening. This corrects the
sink's existing queue-exhaustion/abort behavior for observed and unobserved callers.
Explicit Stop, failure and replacement still abort immediately with the existing
two-block silence contract; a late finished callback cannot turn an interruption
into successful completion or disturb a newer owner. Invalid DAC timing disables
mouth observation but does not replace native completion with a clock estimate.
Preserve the existing buffered-duration plus five-second drain watchdog so missing
backend acknowledgement fails with bounded cleanup rather than hanging.

Bound queued envelope metadata by the existing queued-audio capacity plus a
one-second device-tail reserve, and keep only the latest presentation snapshot
for readers. Retain a window until its mapped device-output range has passed the
DAC timeline, not merely until source samples leave the software queue. A missing
or larger unverified device horizon makes timing unavailable. Stop/replacement
clears the old generation. Slow or hidden UI consumers must not grow a notification
backlog, block audio, or cause samples to be dropped. Presentation failure disables
its observer, not speech playback.

For supported PCM16 WAV artifacts, reuse `sink_plan` and the existing PCM sink
when available so complete-file speech can also have an observed sample clock.
Preserve format validation, existing stop/one-voice ownership and fallback policy;
never launch sink and external player concurrently for the same audio. External
player paths qualify only if a validated audio envelope can be aligned to a
trustworthy output position. The current elapsed-wall-time `get_position()` alone
does not qualify. If the sink/decoder/position capability is absent, retain working
audio with the explicitly described visual fallback. Do not install a dependency
or change a user's speech provider to obtain animation.

Initial internal tuning uses 20 ms RMS windows converted to normalized dBFS.
Digital zero, inserted underrun silence, confirmed pause and terminal states force
closed immediately. Two consecutive raw windows below -45 dBFS (40 ms) close the
mouth and empty the meter regardless of the smoothed value. Speech reopens on a
raw window at or above -42 dBFS. This separate silence gate prevents the release
filter from holding the mouth open over a quiet background or a natural pause.
Missing trustworthy progress for more than 100 ms also forces closed.
Closing resets the filter; reopening seeds it from the first finite speech window,
so digital-zero negative infinity cannot enter its smoothing state.

While the speech gate is open, smooth dBFS with exponential attack/release time
constants of 40/100 ms using the window's actual duration. Select small or open at
-24 dBFS, returning from open to small below -27 dBFS. These are initial internal
signal-processing constants, not user controls. Calibrate against quiet, loud,
nonzero-background and pause-heavy real samples; record adjustments in
implementation notes and deterministic tests. A loud-to-quiet nonzero-background
test must prove closure within the same 100 ms output-timeline target, including
windowing and the next paint; a digital-zero-only test is insufficient.

The presentation ceiling remains 30 paints/second. Target visible onset and
silence response within 100 ms of the observed output timeline on a qualified
local path, reporting measured device latency separately. A simulated clock proves
selection behavior; it does not establish audible hardware synchronization.

## Blinking, idle motion and lifecycle

Mouth pose, blink phase and idle offset have independent clocks. A blink swaps
to the closed-eye counterpart of the current mouth pose; changing mouth level
does not restart blinking or idle motion. Initial blink spacing is irregular
between 3 and 7 seconds, with a closed-eye hold using `$ds-duration-fast`. Idle
movement is a smooth 6-second cycle with no more than one rendered-image pixel of
offset inside a fixed canvas. Avoid moving the frame, controls or portrait fit.
Declare missing interval/cycle/amplitude design tokens centrally; the audio
processing constants above are not stylesheet values.

Do not add a per-frame image decode or rail remount. Prepare frames off-thread,
reuse the avatar's disposable presentation timer and existing generation fencing,
and retain only bounded active frames/current renderables. Preserve ADR-144's
64 MiB active/in-flight preparation budget and existing independent shared-decode
caps. Editor preview and Console preparation must not each assume the whole shared
budget is free. Do not add an unbounded per-expression cache.

Hidden avatars pause visual work. On reveal, sample the current speech owner and
resume from its present state without replaying hidden time. Unmount, source
change, performance revision/removal, local authority replacement, geometry change,
session change or motion-preference change invalidates outstanding preparation.
Resizing may rebuild fitted renderables; it cannot restart
audio, change the selected expression or cause rail geometry oscillation.

Respect existing Dynamic/Static, animations-enabled, Reduce motion, automatic
reaction and show-avatar preferences; no separate motion settings page is added.
Static or Reduce motion suppresses performance movement, dynamic meter and
transmission effects, preserving current static-expression rules and text status.
Automatic reactions off with no manual override retains ADR-144's neutral/static
behavior. Explicit manual selection retains its existing effective-motion rules.
Hiding avatars removes portrait presentation without changing the speech owner.

## Failures and verification

Generation failure, incompatible providers or absent TTS leave imports and existing
artwork usable. Partial drafts cannot publish. Invalid or over-budget stored sets
fall back to ordinary artwork with one concise reason, without repeated per-frame
notifications. Presentation failure never fails an otherwise valid chat or voice
request. Validation and optimistic concurrency remain mandatory at every Apply.

Implementation verification must cover:

| Area | Required evidence |
| --- | --- |
| Artwork and migration | Six-role validation, normalized-size caps, transaction rollback, persisted-source/save gate, local/server/profile authority isolation, stale-source/revision rejection, legacy data unchanged, soft-delete/restore and permanent purge |
| Draft workflow | Generate/import use the same draft; disclosed six-call batch and one-slot retry; offline six-pose preview at Console size; partial failure; guarded/destination-checked TTS sample with exact-owner cancellation; Cancel; successful Apply; character switch during generation; fresh editor reopen |
| Motion logic | Known waveform with exact zero and nonzero quiet background, measured silence-gate response, loud/small hysteresis, partial/full underrun, stale progress, independent blinking and immediate terminal closure |
| Ownership | Manual Speak, automatic replies, optional hands-free observer plus unchanged exactly-once completion/voice selection, sentence handoff, older-message replay/restore, preview isolation, cross-character replay, session switching and late terminal acknowledgement |
| Output capability | Distinct source/device clocks, output-DAC mapping and retained device-tail metadata, resumed speech after inserted silence, partial zero padding, bounded metadata and stalled-reader behavior, supported WAV artifact path, external-player timing refusal, sink failure/fallback without double audio, confirmed transport pause/resume without changing legacy state machines |
| Mounted UI | Actual changed portrait pixels, mood retention, reduced/static motion, hide/show, resize, focus and narrow/tall rail stability with production CSS |
| Persistence | Restart retains applied sets; same-source Apply/Remove refreshes visible caches immediately after commit; stale predecessor workers cannot repaint; full-profile backup/restore retains frames and retires stale presentation; portable formats unchanged |
| Static checks | Targeted lint/format/type checks as applicable, token governance, CSS bundle sync, migration packaging and Backlog guards |

Live qualification uses an isolated profile, one real image-edit backend, imported
reference artwork, and real TTS output. Capture the source, staged six-frame preview,
applied Console portrait, audible start/pause/silence/end/interruption, and a second
conversation switch during speech. Verify the file path and streaming path
separately; list platforms/players not exercised. A screenshot, mocked provider or
passing parser test alone cannot qualify animation quality or audible timing.
Run only tests relevant to changed functionality unless a full sweep is explicitly
requested.

## Review status and next step

The user approved the artwork workflow, radio framing and playback behavior, then
reviewed the written spec and requested a source-anchored issue review before
continuing. The linked review records the original gaps and the targeted
corrections now incorporated here. Local-only performance storage, ordinary
portable exports, six roles and the agreed user-facing feature scope are retained.
The user acknowledged the revised authority, preview ownership, cache invalidation,
hands-free lifecycle, output timing and silence-gate details on 2026-10-23 and
requested implementation planning. The linked plan defines nine atomic Backlog
tasks with explicit interfaces, targeted verification and the ADR check. No product
code, migrations, generation requests or audio runs have been performed for this
feature. Planning and design completion do not mark the feature implemented.
