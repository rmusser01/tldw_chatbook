# Watchlist article listening and saved queues

Date: 2026-11-15
Status: Approved direction; design review incorporated; playback qualification required
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

The global default voice, local-backend scope, cache limits, checkpoint interval, and detailed interruption rules below complete that direction. The proposed player must pass the qualification gate below before feature implementation depends on it.

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

One resumable brief queue exists per local profile. Starting the same saved queue offers Resume. Starting a different brief asks before replacing the saved queue and bookmark, and commits the replacement only after validating its sources. Failed, cancelled, or zero-playable admission preserves the existing session. Reads and conversion prepare a bounded candidate off-loop; the commit rechecks source existence and authority so a concurrent deletion cannot publish a new private snapshot.

Exhausting the queue records Completed and stops. The completed queue remains visible across restarts; Play again explicitly resets its cursor and entry bookmarks, reusing valid retained audio. Resume never silently restarts a completed queue. An all-unavailable queue explains why it cannot start.

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
| Application startup | Restore unfinished sessions paused and retain Completed, with no synthesis or playback |
| Unrelated app audio takes ownership | Checkpoint and pause Watchlists until a manual Resume |

Only successful foreground completion and explicit Skip can automatically return to a previously playing queue. Stop and errors require manual resumption.

Listening from a brief does not mark articles read. Opening an article retains the reader's existing read-state behavior.

## Database and audio identity

Add purpose-specific records to SubscriptionsDB through its migration and transaction conventions.

| Record | Required state |
| --- | --- |
| Listening session | Stable ID, profile and source authority, source brief when present, frozen voice selection, queue cursor, foreground entry, suspended queue's return flag, transport state, revision, timestamps |
| Listening entry | Session/order/foreground role, local item identity, title and source provenance, speech-text snapshot and hash, converter version, voice fingerprint, artifact reference, offset, availability/failure code |
| Article audio | Profile/item/content/voice/converter/dependency identity, generation status and token, relative file key, digest, validated PCM format/frame count, duration, creation and last-use times |

Offsets belong to entries because an interruption and its queue can share audio. Store offsets as PCM frame positions with the artifact's sample rate and digest, clamped to validated duration.

Snapshot speech text at queue admission, including entries awaiting synthesis. Later article edits cannot change a saved entry. A newly queued version receives a different content identity.

Use the app-wide default voice or default profile through the existing TTS request resolver. Persist a canonical allowlisted snapshot of the effective non-secret provider/model/voice/format/speed/options plus its fingerprint. Retain stable profile/reference IDs and revision/digest dependency bindings where applicable. A fingerprint alone cannot reconstruct pending speech after restart or a default change. Re-resolve live credentials and authorized references through existing services before each synthesis; never silently substitute the current default. Pending synthesis pauses if the frozen selection or its dependencies become unavailable or change. An explicit voice update or regeneration resets the affected entry's offset with an explanation.

Credentials, raw provider request objects, and clone-reference paths are excluded from queue records. Existing voice consent and generated-artifact dependency rules still apply. Authorized retained WAV playback requires no provider call.

## Checkpoints and restoration

Pause, Stop, Skip, interruption, replacement, profile switch, and graceful shutdown commit the latest observed media position before releasing playback. Coalesce an off-loop checkpoint at most once per second during playback.

After a hard crash, recovery uses the last durable checkpoint. Up to one checkpoint interval plus player observation latency may repeat. Ordinary interruption resumes the saved position in the same WAV.

Update foreground state, both bookmarks, queue cursor, and automatic-return flag together in one transaction. Commands and completions carry expected session revision and playback generation. An obsolete completion cannot advance a new queue.

Restore an unfinished session paused, including any single article interrupting a queue. Manual Resume continues that article before returning to the suspended queue. Opening the app, Watchlists, or a brief starts neither synthesis nor audio. Completed sessions remain completed.

Capture the profile identity, explicit local backend, and runtime-policy revision for admission and asynchronous work. A runtime source/server-binding change checkpoints and pauses listening, retires that work, and fences late publication even for a local-to-server-to-local transition. Resume requires the matching local authority and fresh policy checks. Missing backend arguments must not silently choose local storage.

A second process must not reconcile away another process's live playback. Reuse existing profile-exclusive ownership where applicable. Otherwise hold one nonblocking profile-private playback lock for the listening service. Only its holder mutates transport or synthesizes for the session. Other instances can display saved state with an ownership explanation. Crash release permits recovery. Revision checks also fence delayed callbacks.

Affected source/session deletion, destructive restore, and cache cleanup must participate in that ownership boundary. The holder quiesces its own playback and publication before mutation; another process must acquire ownership or fail busy with a remedy. A playback-only lock that other destructive writers bypass cannot satisfy stop-before-delete. Recheck source validity and generation in the transaction that publishes a synthesized artifact or attaches its reference.

## Controlled playback proposal

Extend the shared playback owner with a controlled WAV path for Watchlists. The user-selected first delivery is Windows-only, using the qualified installed ffplay build with an owned process handle, bounded shutdown, observed completion, and a start position. Linux/macOS remain unqualified and controlled article listening must refuse before synthesis there; existing ordinary playback retains its platform support. Keep the existing safe argument-list launch and process cleanup patterns.

ffplay documents position seeking through -ss and media-position statistics through -stats, but does not promise an exact seek or a stable machine-readable statistics protocol. Treat its bounded progress parser as a qualified compatibility adapter. Tag observations with the process generation, reject invalid/stale observations, and discard raw output. Checkpoint audio time, never elapsed synthesis or UI time. [ffplay reference](https://ffmpeg.org/ffplay.html)

Pause can checkpoint and terminate the player. Resume starts the retained PCM WAV at that bookmark. Qualify resumed position within 250 ms of the saved position using marked audio on each supported OS. Inability to observe progress stops controlled playback with a remedy and preserves the last trustworthy bookmark. Confirm Paused and release output ownership only after confirmed silence/termination. If bounded termination fails, retain blocked ownership and prevent another output from starting until it is resolved.

Playback qualification is the first implementation gate. Before building persistence and UI around ffplay, establish nonzero seek accuracy, short-clip handling, final audible content and natural completion, invalid/stale telemetry handling, and Stop during process startup on the supported Windows profile. The retained 21-case matrix and Windows listener pass require the additional current-profile startup-Stop control. Linux/macOS need independent qualification before a later platform expansion and do not block this Windows-first delivery. A normal process exit alone is insufficient evidence of full playback. If this contract cannot be established, revisit the backend choice before dependent implementation; do not weaken resume behavior or substitute wall-clock bookmarks.

The initial retained audio shape is PCM16 stereo at 48 kHz with 250 ms of trailing zero PCM. Normalize provider output before publication within existing aggregate memory bounds; frame bookmarks and artifact identity refer to the normalized padded WAV. Exact logical-EOF Resume launches no player. Stop during padding is an interruption, never successful foreground completion or automatic queue return.

Check controlled playback and required conversion/stitching capabilities before spending a synthesis request. Missing ffplay or a required optional dependency leaves the queue readable and offers setup guidance. Default-app handoff lacks the required stop/completion contract for queued listening. Existing ordinary file playback retains its accepted behavior.

Console file and streaming-PCM output, Speech Lab, cast-brief playback, media playback, and Watchlists participate in serialized output admission in both directions. SimpleAudioPlayer currently covers only file playback; the streaming sink and media pipeline have independent outputs. Reuse their output adapters behind the smallest shared ownership boundary. Revoke the previous generation's permission to start or publish output before awaiting retirement, checkpoint Watchlists where applicable, and confirm the previous output is silent before granting the next owner. Late generation/process starts must be fenced as well as callbacks. Stop, navigation cleanup, and unmount may retire only their own owner token; an old surface cannot stop a new owner's audio. Unrelated audio never automatically resumes Watchlists.

A recorded external-app handoff needs special treatment because the app cannot observe its completion or stop it. Explain that external playback must be stopped and require an explicit takeover acknowledgement before controlled listening starts. Releasing an in-app slot cannot be presented as proof that the external application stopped.

## Synthesis and runtime

Compose one app-scoped listening service with SubscriptionsDB and the existing TTS service. It owns transport, player callbacks, and generation. Screens post intents and display snapshots. Listening may continue during in-app navigation, with command-palette Pause/Resume and Stop. Profile switching checkpoints and retires the old service first.

Convert saved text, HTML, or Markdown to speech-friendly text. Read visible link labels, omit URL targets and image descriptions, remove markup and fence markers, discard scripts/styles, and retain readable prose and code text. Store the conversion version with its output.

Reuse exact/native and legacy TTS dispatch, existing chunking, WAV concatenation, and artifact bounds. Long bodies produce complete article audio from multiple requests. An over-limit body fails with an explanation instead of an unannounced partial recording.

Generate the current entry on demand after explicit listening intent. Start with one synthesis in flight and no speculative prefetch. DB/file work, conversion, and synthesis stay off the message loop. Cancellation immediately revokes the operation's publication/playback authority. A thread-backed provider may continue native work; retain its cleanup owner until it settles, block another listening synthesis while it remains in flight, and keep Stop and navigation responsive. Profile changes must not redirect its publication or cleanup into the new profile.

Set explicit per-entry and total queue snapshot limits before conversion/admission, and enforce an aggregate PCM/encoded-byte budget while collecting chunks and stitching. The existing 1,800-character per-request split and backend response bounds do not bound a whole article's accumulated memory or paid requests. Reuse the shared buffered-audio limit where applicable, account for conversion/stitching copies, and stop before retaining an over-limit segment. Reject oversized candidates or entries with an explanation; never silently truncate speech or replace a usable session with a failed candidate.

Partial files cannot become playable. Obsolete results may only be discarded or finalized as unreferenced cache artifacts after dependency checks. They cannot start playback or update the active session.

## Files, deletion, and recovery

Retain profile-private WAVs under an article_audio asset root. Metadata stores validated relative keys. Reuse private-path validation and storage admission so a database value cannot become an arbitrary filesystem or executable target.

Write and validate a private temporary file, atomically finalize it, then complete the artifact row. Recovery handles interrupted rows, missing files, and orphaned finalized files. Reconcile prior interrupted work only after exclusive playback ownership is established.

Referenced audio survives pauses and restarts. Proposed defaults retain unreferenced audio for seven days under a 512 MiB least-recently-used cache ceiling. Active queue references protect files from eviction, including the suspended entry. User-retained active audio can exceed the cache ceiling. Disk-full failure pauses with the entry intact. Clear releases references and makes files immediately eligible for cleanup.

Scope audio reuse to the same profile and source item, including its content, voice, converter, and dependency identity. A queue and foreground reading of that item may share the file. Identical text in unrelated items does not share an artifact; this keeps deletion ownership unambiguous.

Item/source deletion removes its saved text snapshots and invalidates its associated audio under the source's deletion policy. Stop affected active playback first and fence pending synthesis/admission against that deletion. Keep an unavailable placeholder and its reason to preserve order. Deleting a watchlist or removing membership preserves surviving sources and their other listening references, as the existing Watchlists contract requires; release only the deleted container's session references. Deleting the source brief retires its saved queue. An unrelated, still-valid foreground reading becomes standalone with no queued return.

Register article_audio with the existing subscriptions.assets backup owner. Update detached layout, staging, schema census, and recovery policy together. Register listening publication with the existing backup pause/quiescence boundary so capture observes a consistent database and asset inventory. Restored unfinished sessions are paused. A backup without audio or a missing/corrupt file leaves the entry needing audio. Explicit regeneration starts it at zero, rather than applying an old offset to a new synthesis.

SubscriptionsDB is version 3 at this baseline. Allocate the next schema step against the implementation base and add its SQL artifact under DB/migrations. Update table/index allowlists and exact backup/recovery schema contracts. Old profiles begin without a listening session. Failed migration rolls back both schema changes and version.

## UI and privacy

The reader gains Listen and a free single-letter binding, rechecked against current keys. The brief's citation area gains Play cited excerpts and Listen for the selected citation, using existing message/handler patterns and preserving the brief view.

One compact transport shows current article, stored-feed-text status, known elapsed/duration, queue position, and any suspended queue. Expose Pause/Resume, Stop, Skip, and Clear queue. Preparing audio has a distinct state with item/chunk progress where known, and responsive Pause/Stop/Clear; it must not claim playback has started. After restart the primary action for unfinished sessions is Resume saved listening; completed sessions offer Play again. Unknown timing is omitted. Empty bodies, unsupported content, player readiness, storage failures, and TTS failures have distinct explanations and recovery.

Use ADR-150 tokens and ADR-161 component patterns. Verify 80x24 and wider layouts, keyboard reachability, focus during refresh, and truthful footer hints.

Saved excerpts and WAVs are private profile data. Existing TTS consent, readiness, provider disclosure, and credential rules apply. Listening introduces no LLM call. Diagnostics contain bounded states, counts, timings, and failure codes. Article/title text, source URLs, raw player output, text-bearing exception locals, credentials, and voice-reference payloads are excluded.

## ADR check

[ADR-367](../../../backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md) records the durable queue state, audio lifecycle, controlled playback contract and provisional ffplay gate. The [implementation plan](../plans/2026-11-15-watchlist-article-listening.md) maps the work to TASK-41176 through TASK-41182, beginning with native playback qualification.

Existing governing decisions include [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md), [ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md), [ADR-040](../../../backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md), [ADR-023](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md), and [ADR-304](../../../backlog/decisions/304-global-and-studio-tts-settings-ownership.md).

## Acceptance and evidence

Focused regression tests and live playback must establish these outcomes.

1. Listen speaks the complete saved body after conversion, with excerpt identification and useful missing-body/change-item behavior.
2. A brief queues validated cited items once in first-citation order, preserving unavailable placeholders and the brief view. Failed or zero-playable replacement keeps the previous queue; completed queues replay only by explicit Play again.
3. Mid-excerpt interruption plays one article and returns to the original media position. Stop and failure keep both paused.
4. Restart during either queue or foreground playback restores paused. Manual Resume continues the foreground article and its queued return.
5. Graceful resume meets the 250 ms target on the qualified Windows profile. Linux/macOS controlled article listening stays unavailable pending independent qualification. Crash tests use the one-second checkpoint bound.
6. Repeated Listen, replacement foreground, Skip, Clear, profile/runtime-policy changes, unrelated audio, old-screen cleanup, delayed process starts, and obsolete callbacks never overlap audio or mutate a newer queue. Failed stop retains blocked ownership.
7. Admitted long bodies synthesize completely in chunks within aggregate text/audio bounds. Frozen voices survive default changes and restart; changed dependencies pause pending synthesis. Reusing valid retained audio spends no synthesis request.
8. Missing ffplay is detected before synthesis. Provider errors, corrupt/missing audio, disk-full publication, and interrupted generation preserve recovery.
9. Deletion racing admission/synthesis, surviving watchlist membership, rollback, two-process ownership, backup/restore with and without audio, and cleanup preserve privacy and database/file consistency.
10. Mounted controls pass geometry/focus checks. A configured provider and audible native playback establish the end-to-end behavior, including position restoration.

## Size and planning handoff

This is a multi-step project because persistence, interrupted playback, and portable seek support cross storage and runtime boundaries.

Next, write the implementation plan, allocate the ADR and atomic Backlog tasks, and select independently testable deliveries, beginning with playback qualification. Reuse existing owners and helpers. The builder may settle SQL names/indexes, internal APIs, concrete aggregate admission limits, the smallest shared-WAV helper extraction, and the qualified progress parser. Recheck the source baseline before implementing.
