# ADR-242: Console radio portraits and character performance attachments

Status: Proposed; written design awaiting user review
Date: 2026-10-23
Related Task: [TASK-36509](../tasks/task-36509%20-%20Design-Console-radio-portraits-with-audio-reactive-character-performance.md)
Spec: [Console radio portraits](../../Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md)
Review: [Source-anchored design review](../../Docs/superpowers/reviews/2026-10-23-console-radio-portraits-design-review.md)
Supersedes: N/A

## Context

The Console already resolves character expressions and plays animated image
assets, but its operational speaking state follows text streaming. Users want a
living portrait with mouth movement driven by audible TTS, independent blinking
and mood, and radio-screen framing. They also need to generate matching artwork
from an existing portrait or import their own, then review it before activation.

The existing immutable Visual Identity manifest describes expression assets, not
mouth/eye roles. TTS has request-owned playback lifecycle signals, but output-level
amplitude and timing are not yet a shared presentation contract. External player
elapsed-time estimates cannot establish accurate mouth timing.

## Decision

1. Keep mood resolution authoritative. Attach an optional six-pose performance to
   a character's resolved expression and exact source-image digest. Three mouth
   poses each have open-eye and closed-eye variants. These are rendering roles,
   not new mood keys, a second emotion model, or another Buddy runtime.
2. Generate and import into the same disposable character-editor draft, admitted
   only for saved local characters and persisted active artwork. Capture and
   revalidate exact profile/service/database authority; server-owned IDs are not
   local authority. Require source Save/Apply before preparation when artwork is
   dirty, without implicitly saving unrelated edits. Use only configured
   reference-image-capable image-generation backends. Show provider destination
   and six-request batch size before explicit submission. Preview all six frames
   offline at Console size; Apply alone activates them, with stale-source and
   revision checks. Existing images survive partial generation or cancelled review.
3. Store bounded normalized PNG frames in new local character-database parent and
   child records, atomically published through existing transactions. Add a real
   SQL migration at the next available schema version. Keep original portraits,
   legacy expression images and immutable visual-pack assets unchanged. Publish
   committed Apply/Remove invalidation and include performance revision/absence
   plus exact local authority in all request/preparation cache identities.
4. Treat first-version performances as character/profile-local attachments.
   Full-profile backup/restore includes them. Existing portable character cards,
   Actor Packs and Chatbooks retain their ordinary artwork/formats and omit the
   new data. Disclose that limit; do not introduce a native pack-format extension
   or cross-character binding implicitly.
5. Extend the existing speech boundary with transient owner-fenced presentation
   state: confirmed transport state, output-timed audio envelope and timing
   capability. Hands-free gets an optional owner/observer without changing its
   existing voice selection or exactly-once completion. Editor TTS samples reuse
   guarded app-owned utterance/destination admission and exact-owner cancellation,
   never fabricated message snapshots or global Stop. Confirmed transport pause
   does not add unsupported states to the existing speech-admission lifecycle.
   No audio content or envelope history is persisted in messages or exported.
   The portrait follows speech; hiding the Console avatar does not change audio
   policy, while closing a disposable editor preview stops only that preview.
6. Follow PCM output consumption and underrun silence, with energy preparation
   outside the realtime callback. Distinguish consumed source frames from device
   frames, including partial zero padding, and align to validated output-DAC time.
   Bound envelope metadata by queued audio and coalesce reader snapshots; a slow
   observer cannot block or fail audio. A raw silence gate closes the mouth
   independently of release smoothing. Reuse the existing PCM sink for eligible WAV
   artifacts. External playback gets mouth animation only with a trustworthy
   envelope/position pairing; otherwise retain working voice and show truthful
   status with the original portrait. Never simulate sync from text or network
   chunk timing, change providers silently, or play through two paths at once.
7. Keep one stable token-backed radio frame in the existing Character rail.
   Preserve expression precedence, geometry, keyboard access, current bounded
   preparation and stale-worker rejection. Existing motion/accessibility settings
   suppress animation without hiding voice status. No new graphics dependency,
   destination, contact/call interaction or sound pack is introduced.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Loop a talking image for the whole utterance | Mouth movement would continue through pauses and underruns. |
| Phoneme alignment or a facial rig | More artwork, analysis and dependencies than the requested first version needs. |
| Warp any source portrait automatically | No dependable facial geometry; risks distortion across art styles and poses. |
| Encode poses as custom expression moods | Conflates emotional selection with mouth/eye rendering and changes precedence/catalog behavior. |
| Extend immutable visual-pack manifests immediately | Adds a portability/server-format contract absent from the requested first version. Local attachments preserve current formats. |
| Private loose frame files plus database pointers | Requires another file/DB publication and backup lifetime; six bounded SQLite blobs allow atomic Apply. |
| Estimate file progress from elapsed wall time | Player startup, pause and buffering drift can produce visibly false synchronization. |
| A new audio/animation framework | Existing speech owners, PCM sink and disposable avatar provide the required boundaries. |

## Consequences

This adds bounded data to the main SQLite store and therefore requires migration,
rollback, purge and full-profile restore coverage. Performance sets do not travel
with existing portable character exports in this version. That limitation is
explicit rather than silently altering established pack validation/digests.

Generated frame alignment remains a human preview decision. Source changes or
missing/corrupt performance data fall back to existing expression artwork. Precise
animation availability depends on actual playback capability; unsupported external
players keep audio and truthful status. Hardware timing and artwork quality need
live verification, separately from deterministic waveform/ownership tests.

## Preserved decisions

- [ADR-074: Actor packs and local visual runtimes](074-portable-actor-packs-and-local-persona-visual-runtime.md)
- [ADR-083: Console rail ownership](083-console-edge-rails-and-workspace-tree-ownership.md)
- [ADR-144: Dynamic/Static expression playback](144-character-expression-playback.md)
- [ADR-150: Design-token system](150-design-token-system-and-design-language.md)
- [ADR-161: Component patterns](161-component-pattern-library.md)
