# Console radio portraits and audio-reactive character performance

Date: 2026-10-23
Status: Original design implemented; compositing direction approved 2026-10-24, detailed follow-up prepared for review
Revision: 2026-10-24 compositing direction, one-time manual selection and three follow-up review corrections; generation/storage refinement in [ADR-243](../../../backlog/decisions/243-source-bound-portrait-compositing.md). Original source review: [review](../reviews/2026-10-23-console-radio-portraits-design-review.md)
Task: [TASK-36509](../../../backlog/tasks/task-36509%20-%20Design-Console-radio-portraits-with-audio-reactive-character-performance.md)
Decision: [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md)
Implementation: [Original completed plan](../plans/2026-10-23-console-radio-portraits-implementation.md); [compositing follow-up plan and five atomic tasks](../plans/2026-10-24-portrait-compositing-implementation.md) (planning only)

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

The frames share pixel dimensions, framing, background and pose. Playback still
selects whole frames; it does not composite or detect faces at runtime. Ready-made
imports retain six-slot review. Generated frames are assembled locally from shared
components under the rules below, enforcing exact paired pixels. Facial correctness,
mask placement and seams still require full-resolution and animated review.

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
2. For generation, confirm that the prepared still has open eyes and a closed
   mouth; mark and explicitly save its eye/mouth regions once. Generate three edits
   and assemble the six poses locally. Alternatively, assign ready-made images to
   the six slots without region selection. Both paths use the disposable draft.
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

### One-time source regions

Add **Edit regions…** to Living portrait. A focused region editor shows the exact
prepared base with labeled image-left eye, image-right eye and mouth rectangles.
Coordinates are integer prepared-image pixels, not original unscaled-image pixels
or terminal cells. Two pointer clicks set opposite corners; labeled x/y/width/height
fields allow the entire task by keyboard and precise adjustments. Show the full
contain-fitted source, current coordinates and a detailed crop/mask preview. Pointer
mapping subtracts contain padding, accounts for terminal-cell aspect and refuses
clicks in letterboxing. Resize changes the view transform, never the stored boxes.
Use token-backed focus/outline/state styling and preserve global terminal shortcuts.
No face model, automatic guess, freehand editor or new graphics dependency is needed.

Require exactly three rectangles, each at least 3×3 pixels, inside the base and
pairwise non-overlapping. Generate mask support only inside those rectangles.
Explicitly confirm **Base has open eyes and a closed mouth** before **Save regions**.
The app cannot infer this semantic condition. A source with an open mouth/closed
eyes needs a different saved base or ready-made six-frame imports; it triggers no
extra generation or automatic modification of the source portrait.

Save regions is an explicit local preparation-settings write with source/profile/
editor authority and its own optimistic revision. Reopen on the exact same source,
prepared pixel digest/dimensions and algorithm version reuses it without marking again.
A different source/binding or unsupported algorithm requires new confirmed regions;
never silently scale coordinates. One current recipe per character/expression
bounds storage. Cancel region editing drops only unsaved changes. Saving an unchanged recipe with
the current revision is a no-op; stale revisions still conflict. Saving changed
regions stops only the current draft sample before invalidating generated draft components and requires a fresh batch while
keeping the applied portrait active. Manual imported frames remain independently
usable. No provider call occurs during selection, save or offline preview.

### Prepared-base identity

Keep the original encoded source SHA-256 and full PerformanceSource unchanged for
ownership/source fencing. A separate `base_pixel_sha256` identifies the prepared
base for reusable regions and composition provenance. It hashes canonical decoded
RGBA8 pixels, never PNG compression or metadata. The exact digest input is the ASCII
tag `tldw-portrait-base-rgba-v1` followed by one NUL byte, unsigned 32-bit big-endian
width and height, then unpremultiplied RGBA bytes in top-to-bottom, left-to-right
order. Include all four channels, including RGB under zero alpha. Validate decoded
limits before hashing and charge the pixel-byte buffer to the shared memory budget.

Equivalent PNG encodings of the same prepared pixels reuse the recipe without a
Save or revision bump. A change to dimensions, any RGBA byte, preparation semantics,
source identity or supported mask algorithm is still detected; encoding equivalence
does not permit retargeting to a different original source. Artifact/frame SHA-256
values continue to cover encoded bytes for file integrity. Freeze one exact prepared
reference object within each generation batch as before.

### Three generated components and exact six-pose assembly

The action says **Generate components (3 edits)…**, names the configured destination
and model/workflow, discloses three sequential image-edit requests and explains that
it replaces the disposable candidate. All three use the identical bounded prepared
base: **eyelids closed**, **mouth slightly open**, **mouth open**. Prompt for unchanged
pose/framing/lighting; enforcement comes from local masks, not prompt compliance.
The base itself supplies `rest`. No provider-native masking support is required.
Existing image-reference capability, request validation, output bounds and secret
handling remain mandatory. Refuse incompatible result dimensions, animation or
invalid output; do not independently crop/stretch results or silently resubmit.

Composition algorithm version 1 uses inward feathered rectangular masks. For a
local rectangle of width w and height h, f = min(3, max(1, (min(w,h)-1)//2)).
At local pixel (x,y), d = min(x, w-1-x, y, h-1-y); alpha = 255*min(d,f)//f.
Outside the rectangle alpha is zero. Union the two disjoint eye masks. Composite
raw RGB edit pixels against base RGB with the matching mask, then restore the
base alpha everywhere. The finished component plane equals the base outside its
mask. Pixel blending/encoding uses the existing Pillow stack, not a new library.

Each raw component is feathered onto the base exactly once. The finished small/open
mouth planes supply the two open-eye poses. Finished eyelids supply `blink_rest`;
copy their already-composited eye pixels over each mouth plane to create
`blink_small` and `blink_open`, using a binary mask for eye-support copying.
Never feather a finished plane a second time. Assert all six roles and matching
dimensions, exact base alpha, exact base pixels outside masks, identical mouth pixels
across eye pairs and identical eye pixels across mouth states. `rest` equals the
prepared base, whose normalization may differ from the original encoded source.
Original source bytes remain unchanged.

Sequential component results form one immutable candidate snapshot bound to the
prepared pixel identity, source, saved region revision and algorithm. Each snapshot
owns both its available frames and its validated component planes; install them as
one state change. A full batch, after consent and before its first provider request,
atomically replaces the disposable candidate with only base `rest` and zero ready
components. It never merges with previously loaded/manual/applied draft frames.
Preparing this initial snapshot or cancelling consent leaves the previous draft
unchanged if no new batch was admitted. The already applied database set stays active
throughout every batch and retry.

Retain successful components from this new batch when another fails. Composed Apply
and Play speech sample require all three current components and exactly six current
roles; six occupied slots alone are insufficient. The draft's publication accessor
must enforce the same readiness as its buttons. Mark failed/uncertain components
and permit explicit retry. A component retry replaces the entire derived candidate
snapshot atomically, retaining unrelated frame bytes and current components. If the
retry fails, its prior snapshot remains valid; a failed retry of a complete candidate
does not make that previous valid candidate incomplete. Changing saved regions
installs an empty composition candidate for the new recipe rather than relabeling
old frames as manual. Normal manual imports retain their separate six-role rule.

Validate all new frames and total bytes before any snapshot replacement. Check the
captured previous snapshot identity and editor/source/recipe epochs after worker
completion; stale results cannot replace a different candidate. A failure before
replacement leaves frames, components, provenance and readiness unchanged together.

A separate component selector offers **Eyelids**, **Small mouth**, **Open mouth**.
**Regenerate component (1 edit)…** discloses its affected poses: eyelids updates all
three blink frames; small/open mouth updates its two eye variants. Unaffected frames
stay byte-identical. There is no paid regeneration of `rest` or an independent blink
pose. Keep the six-role selector for imports/individual preview. A successful manual Import or explicit
Clear switches the draft to **Manual frames**, drops component lineage and preserves
saved regions and remaining frames. Cancelled/invalid imports retain the previous
frames and component lineage; a new composite batch requires explicit consent.

Revalidate editor, source, profile, region revision and registry/config after each
await and before every submission/publication. Saving changed settings, changing
source/character/profile or closing the draft cancels remaining work and rejects
late results, including uncancellable remote jobs. Never automatically retry an
uncertain paid request. Generation/retry alone does not publish active artwork.

### In-app pair review (TASK-36554)

Provide **Review pairs…** beside Apply for a complete composed candidate. This is
an in-app viewer implemented by TASK-36554, not only an offline QA artifact. It
snapshots the six original normalized PNG frames and recipe. Its labeled 2×3 layout
has columns closed/small/open mouth and rows eyes open/closed. A clearly labeled
scaled overview shows composition, while source-detail mode shows a common source-
pixel crop across all six poses. Select image-left eye, image-right eye or mouth and
pan the same crop coordinates by keyboard/numeric controls to inspect every edge.

Decode from the original frame bytes, never from `CharacterExpressionAvatar.current_image`
or another fitted mosaic. Detail tiles preserve the original RGBA crop pixels with
no pre-downsampling; final graphics/mosaic rendering retains the terminal's actual
capabilities and does not claim physical one-screen-pixel correspondence. Derive
crop size from available viewport/backend geometry; if the region is larger, pan
through it. Bound the assembled review raster to 1,048,576 pixels, shorten the crop
without resampling when necessary, and decode/crop/close one source pose at a time.
Do not allocate a full six-canvas 6-megapixel montage. Share decode reservations,
explicitly close all images on every exit, and cancel stale crop workers.

Review is available with Static/Reduce motion and keyboard-only operation. It starts
no audio, provider call or animation clock and never applies/saves artwork. Closing
it changes no draft state. The controller explicitly retires the displayed modal and pending crops on
candidate/source/profile/recipe changes, rather than relying only on a passive
worker predicate or showing an old snapshot as current. Opening the viewer is not a new
checkbox or acknowledgement gate; Apply readiness remains the actual candidate
validation rule. TASK-36555 must open this real viewer in its installed journey,
verify raw crop pixels and bounds, then inspect captures; generating a separate QA
sheet does not satisfy the in-app requirement.

Review these pairs, the mask/detail crop, deterministic six-pose cycle, idle/blink
and actual Console-sized preview before Apply. Small mosaics alone did not expose
the original exposure/pairing failure. The existing explicit scoped TTS sample
remains available; its owner and cancellation rules do not change. Masks guarantee
pixel relationships and preserved surroundings, not proper eyelids, registration,
local tone or plausible facial anatomy.

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

### Region settings and composition provenance (ADR-243)

The follow-up adds one bounded source-bound preparation recipe per local character/
expression with an independent monotonically increasing revision, in a character-FK
SQLite table. Store full PerformanceSource identity, canonical prepared-base RGBA pixel SHA-256 and
dimensions, three integer rectangles, neutral-base confirmation and algorithm
version. Canonical recipe JSON is at most 2048 UTF-8 bytes and contains no image
bytes, paths, provider secrets or device-local authority identifier. The profile/DB
object authority is captured and revalidated separately. Allocate the next live
schema version when implementing, not a reserved number in this design.

Save regions cannot change performance active status, frame bytes or revision.
Source and recipe CAS checks reject concurrent/late saves. Apply may attach optional
composition-recipe provenance to the existing performance parent in the same
transaction as the six frames, checking both expected performance and region
revisions. Imported/existing sets have absent provenance; existing callers default
to that behavior. Setting region metadata alone does not invalidate Console caches;
only committed performance publication/removal does so.

No extra component PNGs or raw generation outputs are persisted. Reopening an
applied composited set validates its stored recipe/base and six-frame invariants;
`small`, `open` and `blink_rest` are already-composited component planes and support
one-component retry without blending them again. Unchanged stored roles retain their
encoded bytes after content/hash/geometry validation and performance CAS, even if
re-encoding them with the current encoder would produce different bytes. New or
changed input still passes the existing normalization boundary. Missing/corrupt/unknown provenance
or a changed saved recipe disables component reuse and requires a new complete
composition batch. Valid ordinary six-frame playback/import remains available.
Saving a changed recipe never rewrites an applied set in the background.

Remove performance retains saved regions and clears applied provenance. Soft-delete/
restore preserves preparation data; permanent character deletion cascades it.
Full-profile backup/restore includes settings/provenance and must be verified through
a fresh open plus another component edit. Ordinary character/Actor Pack/Chatbook/sync
exports omit both. There is no portable recipe format, new sidecar store or schema
change to immutable mood assets.

### Existing performance storage

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

The original attachment tables shipped in schema 82. ADR-243 adds settings and
optional provenance in the next live schema version: increment `_CURRENT_SCHEMA_VERSION`
and add an executed SQL migration under `DB/migrations/` during implementation.
Do not reserve a migration number in this spec.
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
| No owned speech | Mouth closed; blinking when enabled | Ready |
| Text streaming without a speech request | Existing expression selection; no audio-driven mouth | Ready |
| Owned TTS generating/queued | Current expression, closed mouth | Preparing voice |
| Owned audio actually playing | Mouth follows output envelope; blinking continues | Transmitting |
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

Mouth pose and blink phase are independent. A blink swaps to the closed-eye
counterpart of the current mouth pose; changing mouth level does not restart
blinking. Initial blink spacing is irregular between 3 and 7 seconds, with a
closed-eye hold using `$ds-duration-fast`. Keep the whole portrait stationary:
`$ds-portrait-idle-offset-px` is zero. Idle life comes from blinking, without
bobbing, panning, zooming or changing portrait fit. The prior one-pixel translation
was rejected after the 2026-10-25 live-call preview (TASK-36699): one pixel at a
24px fallback size was conspicuous, and enlarged comparison magnified it further.
The existing cycle token/helper remains compatible but has zero default amplitude.
Frame, controls and portrait fit remain fixed.

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
| Artwork and migration | Encoding-independent base identity, fresh-candidate reset/readiness, region bounds/mask/alpha/pixel-pair invariants, one-time-setting CAS, optional provenance and no-double-feather reopen, legacy upgrade, six-role validation, normalized-size caps, transaction rollback, persisted-source/save gate, local/server/profile authority isolation, stale-source/revision rejection, legacy data unchanged, soft-delete/restore and permanent purge |
| Draft workflow | Generate/import use the same draft; reusable source-bound region settings; three-call component batch and one-component dependent-group retry; full-resolution paired matrix and offline six-pose preview at Console size; partial failure; guarded/destination-checked TTS sample with exact-owner cancellation; Cancel; successful Apply; character switch during generation; fresh editor reopen |
| Motion logic | Known waveform with exact zero and nonzero quiet background, measured silence-gate response, loud/small hysteresis, partial/full underrun, stale progress, independent blinking and immediate terminal closure |
| Ownership | Manual Speak, automatic replies, optional hands-free observer plus unchanged exactly-once completion/voice selection, sentence handoff, older-message replay/restore, preview isolation, cross-character replay, session switching and late terminal acknowledgement |
| Output capability | Distinct source/device clocks, output-DAC mapping and retained device-tail metadata, resumed speech after inserted silence, partial zero padding, bounded metadata and stalled-reader behavior, supported WAV artifact path, external-player timing refusal, sink failure/fallback without double audio, confirmed transport pause/resume without changing legacy state machines |
| Mounted UI | Actual in-app pair viewer with original-frame native-detail crops, bounded sequential decoding and stale-review rejection, actual changed portrait pixels, mood retention, reduced/static motion, hide/show, resize, focus and narrow/tall rail stability with production CSS |
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

Original tasks TASK-36542–36550 are implemented and locally reviewed with the
qualification limits recorded in their plan/review. The rejected v1 generated
artwork was corrected using authorized local compositing; this does not mean the
app already uses that generation workflow. On 2026-10-24 the user approved the
component approach and one-time manual selection, then requested updated design,
implementation planning and atomic Backlog tasks. ADR-243 refines generation and
preparation storage; playback/TTS/radio contracts above remain governed by ADR-242.

The [follow-up plan](../plans/2026-10-24-portrait-compositing-implementation.md) defines
TASK-36551–36555. Those tasks are To Do; no compositing product code, migration,
provider job or new audio qualification is implemented by this documentation update.
