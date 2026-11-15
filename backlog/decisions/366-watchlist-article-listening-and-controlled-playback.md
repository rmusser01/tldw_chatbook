# ADR-366: Watchlists owns saved article listening state

Date: 2026-11-15
Status: Accepted architecture; ffplay backend provisional until qualified
Issue: [#2864](https://github.com/rmusser01/tldw_chatbook/issues/2864)
Design: [article listening](../../Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md)
Plan: [implementation plan](../../Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md)
Related: ADR-029, ADR-033, ADR-040, ADR-051, ADR-150, ADR-161, ADR-304

## Context

Users need to listen to saved Watchlists article text, queue a brief's cited
excerpts, interrupt that queue with one article, and resume at the saved position.
The queue and both playback positions must survive restart without autoplay.

SubscriptionsDB owns local items and briefing provenance. Its schema is version 3
at the design baseline. Existing cast synthesis already chunks long text and
produces WAVs, but cast audio is script-owned and cannot represent article queues.
SimpleAudioPlayer owns legacy file playback; Console streaming PCM and the media
pipeline have separate outputs. Windows external-player handoff has no owned
completion, stop, or seek contract. Re-synthesized speech may have different
timing, so a bookmark cannot safely refer to newly generated audio.

## Decision

1. **Persist domain state in SubscriptionsDB.** Add purpose-specific listening
   session, entry, and article-audio records through the next transactional
   migration. One resumable brief queue and at most one foreground interruption
   exist per active-user profile. The session stores cursor, return intent,
   transport state and revision; each entry owns its offset. A standalone article
   uses the same session model with no queue. Do not introduce a second manifest
   or a root application-state object.
2. **Bind state to saved content and authority.** Admit a bounded snapshot of
   speech-prepared text using explicit local authority and validated saved
   citation provenance. Capture runtime-policy revision and fence away/back
   changes as well as profile changes. The first release does not fetch full
   pages or resolve server IDs against local tables. Listening never marks an
   item read. Change reports remain unavailable with a reason.
3. **Persist reconstructible voice identity.** Store only allowlisted nonsecret
   effective voice settings and stable profile/reference revision and digest
   bindings. Fetch credentials and authorized reference material through existing
   admission services at request time. A changed/unavailable frozen dependency
   pauses pending synthesis; an explicit voice change resets affected bookmarks.
   No raw request objects, reference paths, credentials, or reference payloads are
   copied into listening records. Existing consent/lifetime policy remains in force.
4. **Bookmarks name retained audio.** Store a PCM frame offset with artifact
   digest/sample rate. Cache identity includes profile, source item, text,
   converter, voice and dependencies. Only readings of the same item share an
   artifact. Keep profile-private WAV files under `article_audio`, with validated
   relative keys, temporary publication, validation and atomic finalization.
   Missing/corrupt audio requires explicit regeneration at zero. Completed queues
   stay completed until explicit Play again.
5. **Use controlled output with symmetric ownership.** Add the smallest shared
   output-admission boundary across existing file, streaming-PCM and media
   adapters. Owner tokens fence late starts, callbacks and cleanup. A transfer
   revokes the old owner's start authority, checkpoints where relevant, and
   confirms silence before granting a new owner. Failed termination retains
   blocked ownership; Paused is not evidence that an unresponsive process stopped.
   External-app handoff requires explicit takeover acknowledgement because its
   silence cannot be observed. Unrelated audio leaves Watchlists manually paused.
6. **Qualify ffplay before depending on it.** The candidate controlled WAV path
   uses installed ffplay, safe argument lists, owned process lifetime, `-ss`, and
   bounded transient `-stats` observations. Its output is a compatibility adapter,
   not a promised machine protocol. Native qualification must establish seek
   accuracy within 250 ms, short clips, final content, completion, stale/invalid
   telemetry, and Stop during startup on supported Windows, macOS and Linux
   builds. Simulation, SDL dummy output and process exit alone cannot establish
   audible playback. Separate endpoint-loopback evidence from acoustic/listener
   evidence. If qualification fails, revisit this backend before dependent
   persistence/UI implementation. Keep the promised resume semantics.
7. **Checkpoint atomically and restore manually.** Update both entry bookmarks,
   foreground state, cursor and return intent in one revision-checked transaction.
   Checkpoint at most once per second while playing and on explicit transitions.
   A crash may repeat one checkpoint interval plus observation latency. Restore
   unfinished sessions paused and completed sessions completed, with no synthesis
   on startup. Only successful foreground completion or explicit Skip returns
   automatically to a previously playing queue.
8. **Serialize publication and destructive mutation.** One nonblocking private
   profile lock owns listening transport and synthesis when no stronger existing
   profile ownership applies. Destructive deletion, restore and cleanup join that
   boundary or fail busy. Recheck source existence and generation in the final
   publication transaction. Native inference that outlives cancellation retains
   its cleanup owner but loses publication/playback authority immediately.
9. **Preserve existing data ownership.** Global item/source deletion invalidates
   its text/audio derivatives. Removing membership or deleting a watchlist does
   not delete surviving sources. Deleting a brief retires its queue and leaves a
   still-valid unrelated foreground reading standalone. Keep unavailable entries
   where required to preserve queue order. Use best-effort deletion without a
   forensic-erasure claim.
10. **Integrate backup and bounds with existing owners.** Extend
    `subscriptions.assets` and its core dependency, staging, detached layout,
    schema census and restore/relocation contracts together. Listening writers
    participate in backup quiescence. Protect referenced audio; unreferenced cache
    defaults are seven days and 512 MiB LRU. Explicit text, request/chunk and
    aggregate audio-memory bounds prevent an article from bypassing per-response
    limits. No speculative prefetch or new application runtime dependency is
    required. A qualification-only capture helper may be isolated outside the
    application's environment and must not enter packaged dependencies.

## Alternatives

- A profile-local JSON queue manifest avoids SQL changes but splits atomic state,
  deletion, backup and recovery from the source owner. The user selected the DB.
- Shell/default-player handoff cannot stop, observe completion, or restore an
  exact media position, so it cannot implement the queue contract.
- Reusing the entire video pipeline brings video/demux lifecycle into an audio-only
  feature. Reuse its safe process patterns and existing output adapter instead.
- A new decoder/audio-device runtime or mandatory package remains an alternative
  only if the installed-player qualification fails; it needs a focused ADR update.
- Cross-item content dedup saves some disk space but makes source deletion
  ambiguous. Same-item reuse is sufficient for queue/foreground sharing.

## Consequences and verification

This needs coordinated storage, output and lifecycle changes, but keeps the
feature's state with its existing domain owner. The migration, asset policy and
restore contracts must ship together. Server playback and remote-browser audio
remain separate capability work.

TASK-41176 owns the first qualification gate. Native host evidence must name the
actual build, output route, capture method, numerical errors and limitations;
unsupported/unrun platforms remain unqualified. The implementation plan maps the
remaining deliverables to focused regression tests and mounted/native evidence.
