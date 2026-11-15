# Watchlist article listening and saved queues

Date: 2026-11-15
Status: Proposed for user approval
Issue: [#2864](https://github.com/rmusser01/tldw_chatbook/issues/2864)
Source baseline: dev ecad404bd4567b57139716a6deaa6b37fe833749

## Intent and agreed behavior

Watchlists readers should be able to hear saved article text and listen through a daily brief's cited articles without leaving the brief.

The user chose these behaviors.

- Listen speaks the stored excerpt. A feed excerpt is identified as such.
- A brief can queue all cited excerpts in first-citation order. Each article can also play individually.
- A single article interrupts the queue. After it finishes, the queue resumes at the interrupted item's saved position.
- Queue order and playback positions survive restarting Chatbook.
- Reopening restores a paused listening session. The user presses Resume to start playback.
- The Watchlists database owns the queue and playback state.

The global default voice, local-backend scope, player choice, cache limits, checkpoint interval, and detailed interruption rules below are author recommendations for approval.

## Approaches considered

| Approach | Benefit | Cost | Outcome |
| --- | --- | --- | --- |
| SubscriptionsDB owns queue, bookmarks, and audio metadata, with private audio files | Transactional state, item lifecycle, and recovery have one owner | Schema migration and backup-policy updates | Chosen by the user |
| A profile-local queue manifest beside audio files | Avoids a database migration | Separate atomic-write, deletion, backup, and recovery rules | Considered in the discussion |

Both approaches retain the audio associated with a bookmark. Re-synthesizing speech can change timing, making an old position unsuitable for new audio.

## Existing contracts

- [Watchlists architecture](../../Architecture/subscriptions-watchlists.md) describes the local/server split and durable SubscriptionsDB ownership.
- [WatchlistScopeService](../../../tldw_chatbook/Subscriptions/watchlist_scope_service.py) exposes get_item_content for the local backend.
- [ContentPane](../../../tldw_chatbook/UI/Watchlists_Modules/content_pane.py) renders saved bodies and missing-body guidance.
- [Briefing generation](../../../tldw_chatbook/Subscriptions/briefing_service.py) extracts cited IDs in first-seen order. [SubscriptionsDB](../../../tldw_chatbook/DB/Subscriptions_DB.py) preserves citation_position, live_item_id, and title/source provenance in briefing_items.
- [Briefing audio](../../../tldw_chatbook/Subscriptions/briefing_audio.py) provides long-text chunking and WAV synthesis. [Audio stitching](../../../tldw_chatbook/TTS/audio_stitch.py) owns concatenation and duration validation.
- [Global TTS ownership](../../../backlog/decisions/304-global-and-studio-tts-settings-ownership.md) keeps provider and default-voice settings in Settings.
- [SimpleAudioPlayer](../../../tldw_chatbook/TTS/audio_player.py) is the shared playback owner. Its stop-based pause fallback cannot reliably restore a position. Windows shell handoff supplies no owned completion or stop control.
- [The media pipeline](../../../tldw_chatbook/Media_Playback/player_pipeline.py) supplies process-lifetime and seek precedents. Its full video pipeline is larger than this audio-only feature needs.

## Release boundary

The first implementation covers local Watchlists articles and locally saved briefings. Server items report that listening needs a supported body-read capability. Server IDs must never resolve against coincidentally matching local IDs.

Listen uses the saved body at entry creation. Feed text may be a summary, while a URL-monitor article may contain a complete saved page. Speak the whole saved body after speech preparation, independent of the brief generator's 800-character prompt cap.

Change-report items have listening unavailable with an explanation. Missing bodies use existing recovery guidance. Full-page fetching, narrated diffs, article readings inside cast scripts, article podcast export, and per-watchlist voices are follow-up scope.

Playback uses the audio device on the machine running Chatbook. Serve-mode clients see that output location before initiating playback. Streaming audio to a remote browser requires a separate design.

## Listening behavior

### A brief's queue

Play cited excerpts resolves a completed brief's saved provenance in citation_position order, once per item. Current body reads use the local authority. These are text snapshots captured now, rather than reconstructed historical article bodies.

Legacy briefings use the existing citation extraction and membership-validation path. An unknown citation cannot become an arbitrary local item read. Missing items, empty bodies, and unsupported content remain visible unavailable entries. Show playable and unavailable counts before starting. Automatic advancement bypasses unavailable entries while retaining the reason.

One resumable brief queue exists per local profile. Starting the same saved queue offers Resume. Starting a different brief asks before replacing the saved queue and bookmark, and commits the replacement only after validating its sources.

### A single article

Save the queue's media position and suspend it before launching the selected article. Remember whether the queue was playing or already paused.

Successful completion returns to the suspended item at its saved position, continuing automatically only when the queue had been playing. A foreground article does not consume or reorder a matching queue entry. Both may share one audio artifact.

Retain one foreground interruption. Selecting another article replaces that foreground entry and preserves the original suspended bookmark. Repeated Listen on the currently playing entry is idempotent.

### Transport and errors

| Intent or event | Result |
| --- | --- |
| Pause | Halt audio and save foreground and suspended state |
| Resume | Continue the saved foreground state, then its queued return |
| Stop | Halt the whole listening session and preserve bookmarks for manual Resume |
| Skip | Advance the queue, or finish the foreground interruption and return to the queue under its previous play/pause state |
| Clear queue | Stop, retire outstanding work, and remove the session and artifact references |
| Generation or playback failure | Pause on the entry with Retry and Skip available |
| Application startup | Restore paused, with no synthesis or playback |
| Unrelated app audio takes ownership | Checkpoint and pause Watchlists until a manual Resume |

Only successful foreground completion and explicit Skip can automatically return to a previously playing queue. Stop and errors require manual resumption.

Listening from a brief does not mark articles read. Opening an article retains the reader's existing read-state behavior.

## Database and audio identity

Add purpose-specific records to SubscriptionsDB through its migration and transaction conventions.

| Record | Required state |
| --- | --- |
| Listening session | Stable ID, source brief when present, queue cursor, foreground entry, suspended queue's return flag, transport state, revision, timestamps |
| Listening entry | Session/order/foreground role, local item identity, title and source provenance, speech-text snapshot and hash, converter version, voice fingerprint, artifact reference, offset, availability/failure code |
| Article audio | Content/voice/converter identity, generation status and token, relative file key, digest, validated PCM format/frame count, duration, creation and last-use times |

Offsets belong to entries because an interruption and its queue can share audio. Store offsets as PCM frame positions with the artifact's sample rate and digest, clamped to validated duration.

Snapshot speech text at queue admission, including entries awaiting synthesis. Later article edits cannot change a saved entry. A newly queued version receives a different content identity.

Use the app-wide default voice or default profile through the existing TTS request resolver. Freeze its non-secret provider/model/voice/options fingerprint for the session. Pending synthesis pauses if that selection becomes unavailable. An explicit voice update or regeneration resets the affected entry's offset with an explanation.

Credentials, raw provider request objects, and clone-reference paths are excluded from queue records. Existing voice consent and generated-artifact dependency rules still apply. Authorized retained WAV playback requires no provider call.

## Checkpoints and restoration

Pause, Stop, Skip, interruption, replacement, profile switch, and graceful shutdown commit the latest observed media position before releasing playback. Coalesce an off-loop checkpoint at most once per second during playback.

After a hard crash, recovery uses the last durable checkpoint. Up to one checkpoint interval plus player observation latency may repeat. Ordinary interruption resumes the saved position in the same WAV.

Update foreground state, both bookmarks, queue cursor, and automatic-return flag together in one transaction. Commands and completions carry expected session revision and playback generation. An obsolete completion cannot advance a new queue.

Restore a session paused, including any single article interrupting a queue. Manual Resume continues that article before returning to the suspended queue. Opening the app, Watchlists, or a brief starts neither synthesis nor audio.

A second process must not reconcile away another process's live playback. Reuse existing profile-exclusive ownership where applicable. Otherwise hold one nonblocking profile-private playback lock for the listening service. Only its holder mutates transport or synthesizes for the session. Other instances can display saved state with an ownership explanation. Crash release permits recovery. Revision checks also fence delayed callbacks.

## Controlled playback proposal

Extend the shared playback owner with a controlled WAV path for Watchlists. Use an installed ffplay executable on Windows, macOS, and Linux, with an owned process handle, bounded shutdown, observed completion, and a start position. Keep the existing safe argument-list launch and process cleanup patterns.

ffplay documents position seeking through -ss and media-position statistics through -stats. Parse bounded position observations in memory, tag them with the process generation, and discard raw output. Checkpoint audio time, never elapsed synthesis or UI time. [ffplay reference](https://ffmpeg.org/ffplay.html)

Pause can checkpoint and terminate the player. Resume starts the retained PCM WAV at that bookmark. Qualify resumed position within 250 ms of the saved position using marked audio on each supported OS. Inability to observe progress or stop the process pauses with a remedy.

Check controlled playback capability before spending a synthesis request. Missing ffplay leaves the queue readable and offers setup guidance. Default-app handoff lacks the required stop/completion contract for queued listening. Existing ordinary file playback retains its accepted behavior.

Console, Speech Lab, cast-brief playback, and media playback participate in the same ownership transfer. Checkpoint and pause Watchlists before another surface acquires audio. No independent audio singleton is introduced.

A recorded external-app handoff needs special treatment because the app cannot observe its completion or stop it. Explain that external playback must be stopped and require an explicit takeover acknowledgement before controlled listening starts. Releasing an in-app slot cannot be presented as proof that the external application stopped.

## Synthesis and runtime

Compose one app-scoped listening service with SubscriptionsDB and the existing TTS service. It owns transport, player callbacks, and generation. Screens post intents and display snapshots. Listening may continue during in-app navigation, with command-palette Pause/Resume and Stop. Profile switching checkpoints and retires the old service first.

Convert saved text, HTML, or Markdown to speech-friendly text. Read visible link labels, omit URL targets and image descriptions, remove markup and fence markers, discard scripts/styles, and retain readable prose and code text. Store the conversion version with its output.

Reuse exact/native and legacy TTS dispatch, existing chunking, WAV concatenation, and artifact bounds. Long bodies produce complete article audio from multiple requests. An over-limit body fails with an explanation instead of an unannounced partial recording.

Generate the current entry on demand after explicit listening intent. Start with one synthesis in flight and no speculative prefetch. DB/file work, conversion, and synthesis stay off the message loop. Cancellation retires owned workers before transfer.

Partial files cannot become playable. Obsolete results may only be discarded or finalized as unreferenced cache artifacts after dependency checks. They cannot start playback or update the active session.

## Files, deletion, and recovery

Retain profile-private WAVs under an article_audio asset root. Metadata stores validated relative keys. Reuse private-path validation and storage admission so a database value cannot become an arbitrary filesystem or executable target.

Write and validate a private temporary file, atomically finalize it, then complete the artifact row. Recovery handles interrupted rows, missing files, and orphaned finalized files. Reconcile prior interrupted work only after exclusive playback ownership is established.

Referenced audio survives pauses and restarts. Proposed defaults retain unreferenced audio for seven days under a 512 MiB least-recently-used cache ceiling. Active queue references protect files from eviction, including the suspended entry. User-retained active audio can exceed the cache ceiling. Disk-full failure pauses with the entry intact. Clear releases references and makes files immediately eligible for cleanup.

Source deletion removes saved text snapshots and invalidates associated audio under the source's deletion policy. Stop affected active playback first. Keep an unavailable placeholder and its reason to preserve order. Subscription/watchlist deletion also covers snapshots and cached audio. Deleting the source brief releases its saved listening session.

Register article_audio with the existing subscriptions.assets backup owner. Update detached layout, staging, schema census, and recovery policy together. Restored sessions are paused. A backup without audio or a missing/corrupt file leaves the entry needing audio. Explicit regeneration starts it at zero, rather than applying an old offset to a new synthesis.

SubscriptionsDB is version 3 at this baseline. Allocate the next schema step against the implementation base and add its SQL artifact under DB/migrations. Update table/index allowlists and exact backup/recovery schema contracts. Old profiles begin without a listening session. Failed migration rolls back both schema changes and version.

## UI and privacy

The reader gains Listen and a free single-letter binding, rechecked against current keys. The brief's citation area gains Play cited excerpts and Listen for the selected citation, using existing message/handler patterns and preserving the brief view.

One compact transport shows current article, stored-feed-text status, known elapsed/duration, queue position, and any suspended queue. Expose Pause/Resume, Stop, Skip, and Clear queue. After restart the primary action is Resume saved listening. Unknown timing is omitted. Empty bodies, unsupported content, player readiness, storage failures, and TTS failures have distinct explanations and recovery.

Use ADR-150 tokens and ADR-161 component patterns. Verify 80x24 and wider layouts, keyboard reachability, focus during refresh, and truthful footer hints.

Saved excerpts and WAVs are private profile data. Existing TTS consent, readiness, provider disclosure, and credential rules apply. Listening introduces no LLM call. Diagnostics contain bounded states, counts, timings, and failure codes. Article/title text, source URLs, raw player output, text-bearing exception locals, credentials, and voice-reference payloads are excluded.

## ADR check

A new canonical ADR is required before implementation because the change introduces durable queue state, audio lifecycle, and a controlled playback contract. Allocate its number during planning to avoid a concurrent-ID collision. Link it from the approved design, implementation plan, and Backlog tasks.

Existing governing decisions include [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md), [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md), [ADR-040](../../../backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md), [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md), and [ADR-304](../../../backlog/decisions/304-global-and-studio-tts-settings-ownership.md).

## Acceptance and evidence

Focused regression tests and live playback must establish these outcomes.

1. Listen speaks the complete saved body after conversion, with excerpt identification and useful missing-body/change-item behavior.
2. A brief queues validated cited items once in first-citation order, preserving unavailable placeholders and the brief view.
3. Mid-excerpt interruption plays one article and returns to the original media position. Stop and failure keep both paused.
4. Restart during either queue or foreground playback restores paused. Manual Resume continues the foreground article and its queued return.
5. Graceful resume meets the 250 ms target on Windows, macOS, and Linux. Crash tests use the one-second checkpoint bound.
6. Repeated Listen, replacement foreground, Skip, Clear, profile changes, unrelated audio, and obsolete callbacks never overlap audio or mutate a newer queue.
7. Long bodies synthesize completely in chunks. Reusing valid retained audio spends no synthesis request.
8. Missing ffplay is detected before synthesis. Provider errors, corrupt/missing audio, disk-full publication, and interrupted generation preserve recovery.
9. Deletion, rollback, two-process ownership, backup/restore with and without audio, and cleanup preserve privacy and database/file consistency.
10. Mounted controls pass geometry/focus checks. A configured provider and audible native playback establish the end-to-end behavior, including position restoration.

## Size and planning handoff

This is a multi-step project because persistence, interrupted playback, and portable seek support cross storage and runtime boundaries.

After user approval, write the implementation plan, allocate the ADR and atomic Backlog tasks, and select independently testable deliveries. Reuse existing owners and helpers. The builder may settle SQL names/indexes, internal APIs, the smallest shared-WAV helper extraction, and the bounded progress parser. Recheck the source baseline before implementing.
