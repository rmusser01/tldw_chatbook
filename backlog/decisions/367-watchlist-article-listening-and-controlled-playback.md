# ADR-367: Watchlists owns saved article listening state

Date: 2026-11-15
Status: Accepted architecture; recorded Windows-first backend qualified, portable expansion deferred
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
   telemetry, and Stop during startup on the supported Windows build. The first
   delivery is Windows-only; Linux/macOS stay unsupported until independently
   qualified. Simulation, SDL dummy output and process exit alone cannot establish
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

## Windows-first delivery scope, 2026-11-23

The user selected a Windows-first delivery after review of PR #3741. The release
gate now requires the retained Windows 21-case endpoint matrix, its separate
listener confirmation, and a passing startup-Stop control using the current
WASAPI packet capture and fixed-tail profile. Preserve all existing thresholds.
Linux/macOS results remain unqualified and do not block this first delivery;
the portable gate remains open. Do not advertise or start controlled article
listening on those platforms, including before any synthesis request. Ordinary
existing playback keeps its current platform support.

The initial controlled adapter accepts only validated PCM16 stereo 48 kHz WAVs
with the qualified 250 ms trailing-zero policy. Article preparation must normalize
other provider output to that shape before publication, count conversion copies
against the existing memory bound, and bind the bookmark/digest to the resulting
retained artifact. Qualifying additional formats or platforms is later capability
work, not an implicit consequence of this Windows pass.

TASK-41177 owns exact logical-EOF Resume (no player launch), Stop during padding
(never a natural-completion callback), and startup cancellation/owner transfer
regressions. TASK-41180 owns their session/queue transitions. Only after the
Windows gate and review fixes pass may TASK-41177 begin; storage/UI tasks keep
their existing dependency order. This scope change supersedes earlier statements
that Linux/macOS qualification blocks all dependent implementation.

The current-profile three-round startup-Stop supplement passed on source
`1128d94be2`: all packet/prelaunch/post-Stop RMS values were zero, Stop requests
occurred 20.079–22.168 ms after launch, maximum process retirement was 4.507 ms,
and every owned player/capture retired. The initial noisy attempt and subsequent
two-pass/one-unqualified attempt remain preserved; the unmatched nonzero PCM in
that latter row has no attributed source. The diagnostic repeat added numerical
packet RMS observations only, without changing the matcher or grader. The
[Windows release gate](../../Docs/Testing/evidence/watchlist-article-audio/windows-first-release-gate.json)
therefore passes for the recorded build/route/shape. The portable gate remains
unqualified. TASK-41177 may begin under this scope; it still must qualify its
actual adapter and all production ownership transfers before completion.

## Production output contract, 2026-11-23 continuation

The user authorized the controlled adapter, complete production wiring and repair
of the local pytest bootstrap workflow. This continues TASK-41177 beyond its
isolated lease increment. Storage, synthesis orchestration and listening UI keep
their later task boundaries. This record was renumbered from provisional 366 to
367 after dev reserved 366 for CI; the playback evidence itself is unchanged.

- The app owns one admission service. Issue an exact intent ticket at the play
  action, before asynchronous preparation/probe or synthesis intended to play.
  A slow older preparation cannot displace a newer intent. Prepare/validate before
  physical takeover; exact logical EOF needs no player or replacement lease.
- Keep lease authority decisions on the app loop and publish retained work before
  eager callbacks can enter. A one-way thread-safe revocation flag is latched on
  replacement, exact Stop/release and shutdown. A worker checks it at its spawn
  commitment point. Already-committed native work stays owned until it returns
  and its late resource is checked/retired; cancellation never invents absence.
  A controlled start also checks that the exact lease holds that run's bound
  retirement callback. Retain each cleanup task by identity while it settles;
  a retry or confirmation attempt cannot let an earlier waiter grant authority.
- Controlled preparation binds a full physical-file digest, source-frame bookmark,
  content/physical frame counts, PCM16 stereo 48 kHz and versioned 250 ms zero
  padding. Keep a read pin on the exact file and its path ancestry through launch
  and playback, excluding writes, replacement and ancestor redirection. Reuse the
  Windows native path layer with a narrow read-sharing pin; preserve its existing
  default behavior. Recheck private artifact authority on the pinned ancestry.
  Pin and hash the qualified executable too; no unqualified wrapper is executed.
  Pathname consumers must use stable physical volume ancestry, or refuse aliases
  whose DOS-device mapping is outside the retained pins. Check auxiliary handle
  closes and retain ambiguous process births instead of assuming no child.
  For this Windows-only controlled command, use a narrow private CreateProcessW
  launcher with retained PROCESS_INFORMATION and an explicit inherited-handle
  allowlist. Popen hides the birth result behind potentially raising pipe/thread
  cleanup, preventing clean rejection from being distinguished from unknown birth.
  Preserve its argument-list quoting, Unicode environment and checked wait/close
  behavior; ordinary players keep their existing adapters. Native version probes
  reject the extended `\\?\Volume{GUID}` executable spelling with error 87 but
  accept the equivalent `\\.\Volume{GUID}` spelling. Derive that executable
  spelling only from the pinned physical volume identity; retain full ancestry
  pins and qualify the actual WAV consumer path independently.
- Promote only the bounded stdlib progress parser. Keep physical progress distinct
  from the clamped logical bookmark. Natural completion requires corroborating
  advancing, current/fresh source clocks into validated padding, plausible elapsed
  progression and spontaneous checked zero exit, with no Stop/revocation. A last
  stats sample or exit alone is insufficient. Weak evidence is Uncertain; Stop in
  padding is Stopped. The actual adapter's classifier must pass native short,
  near-EOF, resume, EOF and deliberately early-exit controls before activation.
  Validate adjacent as well as cumulative clock progression. Observe the startup
  deadline independently of a pending native spawn, revoke further starts on
  expiry, and keep physical custody until any late birth is checked and retired.
- Reaping ffplay is separate from endpoint silence. Qualify a bounded post-reap
  settling policy with actual old-source end versus successor onset, using the
  production adapter. The candidate 250 ms settling guard was evaluated under
  the unchanged 1.5 s Stop bound; the adapter measurements below qualify it only
  for the recorded Windows profile. Runtime
  retirement combines positive native custody/cleanup and that qualified policy;
  it is not per-run acoustic measurement or a hard real-time/all-device guarantee.
  Default Windows routing policy and the tested executable/PCM shape define the
  first software capability; physical evidence remains scoped to its tested host.
  Explicit alternate/dummy SDL/device routes stay unsupported.
- Successful OS delegation, including unproven wmplayer launcher lifetimes and
  external video opens, records an unresolved in-memory handoff epoch. Before
  any later in-app output, require explicit user acknowledgement that external
  playback has stopped. A newer delegation invalidates older acknowledgements;
  no timer, launcher exit, unmount or artifact deletion clears it. Consecutive
  explicit external opens keep working and accumulate the epoch. Retain/reap only
  owned launcher handles; never kill an unrelated user's media application.
- Lease the entire exact speculative duplex session while its device may render.
  Takeover retires that session through existing close evidence/quarantine rather
  than permanently closing the runtime supervisor. Ordinary PCM and media/file
  adapters must retain uncertain native handles and request checked closes.
  Native workers and process/stream waits run off the UI loop.
- Video-only probes do not displace sound. Audible media joins before every launch
  and seek; checked retirement precedes replacement. The pre-existing Windows
  inherited-fd audio limitation must refuse before takeover until independently
  supported; injected orchestration tests are not native video-audio evidence.
- Approved shutdown closes admission before cancelling producers or tearing down
  views. Exact cleanup cannot target another app/run, including quick quit and
  direct unmount. Do not use process-name scans or killall as retirement proof.

## Windows fixed-tail profile: endpoint qualification passed, portable gate open

The Windows follow-up evaluates **250 ms of local zero PCM appended once to the
retained WAV**, after all original content frames. It does not request more speech
from the provider, regenerate audio on Resume, or relax the 250 ms seek / 20 ms
qualification tail oracles.

- Preserve original content frames unchanged. Record logical content frame count,
  physical WAV frame count, padding frames and padding-policy identity separately.
  The retained artifact digest covers the physical padded file.
- Entry bookmarks and displayed duration use logical content frames. Resume uses
  the saved frame divided by sample rate against the identical retained WAV;
  never add the seek offset twice or offer padding as narrated content.
- Natural completion requires the owned player to finish the validated padded
  artifact. A last statistics sample or crossing logical content duration is not
  sufficient evidence of drain. Stop remains an interruption, not natural EOF.
- This experiment uses PCM16 stereo at 48 kHz on the recorded Windows build and
  default-render route. It does not qualify other WAV shapes, other output
  devices, macOS/Linux, or acoustic/listener output. The TASK-41177 controls below
  separately test exact logical-EOF resume and interruption inside padding.
- At the tested shape, padding adds 12,000 frames / 48,000 bytes per file. Apply
  existing physical-file/cache and aggregate-memory bounds to those bytes too.

The capture setup is a qualification tool concern, not an application dependency.
A focused same-WAV control measured 46-56 ms live matching work and up to 49 ms
receive backlog; the small requested 480-frame capture buffer was actually 1,056
frames (22 ms) and reported a discontinuity. A requested/actual 4,800-frame
capture buffer (100 ms), still emitting 480-frame reads, showed no discontinuities
in that control and passed a same-file Stop/resume. This supports the safer
measurement setup; it is not alone a repeated Windows workaround pass.

The earlier [wrapper-based repeated result](../../Docs/Testing/watchlist-article-audio-qualification.md)
did **not qualify the workaround**. Three rounds produced 21 planned rows and
20 actual launches: 13 natural playbacks and two Stops matched successfully,
including two same-artifact saved-bookmark resumes. Three natural recordings and
one Stop were unmatched, and one natural recording had a noisy baseline/two capture
discontinuities. The unmatched Stop prevented one Resume.
Passing measurements had maximum 2.438 ms seek error, 2.417 ms original-tail error
and 59.5 ms stopped-bookmark error. All started players/capture workers retired.

Those wrapper-based runs lacked reliable timeline evidence for all cases.
The packet-based follow-up below establishes the Windows endpoint pass.
The portable gate and macOS/Linux endpoint/listener qualification remain open.
The TASK-41177 adapter below adopts this profile for Windows controlled output.

## Packet-clock investigation and rerun, 2026-11-23

The gap investigation reproduced two capture mechanisms. SoundCard 0.4.6
inserts elapsed-time-derived zeros after an empty poll. A native 500 ms control
traced 3,871 added frames inside a matched source interval while the source
and device advanced exactly the same number of frames. Both waveform edges
correlated above 0.99999, and the wrapper timeline was 80.646 ms too long.
A second undelayed control lost 10,560 native frames and reported a
discontinuity. Existing legacy receipts are preserved.

Use native WASAPI packet frame positions, first-frame QPC timestamps, and
flags for Windows qualification. The pinned SoundCard setup still opens
the default-render loopback device. The measurement reader bypasses its
estimated-silence path, drains packets independently of analysis, validates
frame continuity and QPC timing, and counts stream-origin and subsequent
discontinuities separately. A native silent packet has measured frames.
Empty polls are monitoring observations. Quiet windows require drain coverage
within actual endpoint buffer capacity and a subsequent drain. Reject
unreliable timestamps and PCM samples.

The 500 ms and playing Stop preflights matched with zero alignment drift,
zero later discontinuities, and zero timestamp-error flags. The same 21-case
matrix then ran with the original 250 ms seek/bookmark, 20 ms tail/drift,
correlation, and 1.5 s Stop thresholds. All 18 launched cases had active
prelaunch audio, with RMS 0.004150–0.018304 against the 0.0002 quiet limit.
The three dependent resumes were skipped. All players and capture workers
retired, and native records reported zero later discontinuities or timestamp
errors. An active Windows render session was identified independently. A
retry after the user's initial pause still rejected all 18 base cases at
RMS 0.001626–0.009073 and skipped three resumes; the session still had a
nonzero peak. Both rejected receipts remain preserved.

After the user muted that session, the identical 21-case matrix passed all
Windows endpoint-loopback checks with every threshold unchanged. All 18
natural playbacks, including three dependent saved resumes, and three playing
Stops passed their applicable checks. Maximum absolute errors were 3.250 ms seek,
2.417 ms natural-content tail, 69.542 ms bookmark alignment and 25.492 ms
telemetry. Maximum process Stop latency was 10.003 ms and last detected
signal after Stop was 32.633 ms. All quiet measurements were zero RMS;
all source spans had zero alignment drift and no later discontinuities or
timestamp errors. All 21 players and capture workers retired.

The 250 ms profile therefore passes for this Windows build, executable,
PCM16 stereo 48 kHz and default-render route. A separate seven-case repeat
passed endpoint checks and the user confirmed audible playback, clean Stop
and finished Resume. This is human listener confirmation, not an instrumented
acoustic recording. It remains unqualified on macOS/Linux and other output
routes. The portable
feature gate remains open. Under the Windows-first amendment, dependent
implementation is unlocked by the passing current-profile startup-Stop supplement.
The [evidence report](../../Docs/Testing/watchlist-article-audio-qualification.md)
links the numerical controls and all three packet matrices.

## Native Linux measurement boundary, 2026-11-23

The supplied host has a real ALSA hardware output, PipeWire 1.4.2's PulseAudio
compatibility server and ffplay 7.1.5. The Windows-style sink-ID/loopback-ID
equality check rejects its valid monitor. Resolve the sink's native
monitor-source relationship and verify the exact capture route; never fall
back to a microphone or fuzzy device match. Keep capture dependencies and
generated files in a private temporary environment, with no system audio
configuration or application dependency change.

Use native Pulse fragments and read indexes for Linux measurement. Validate
negotiated stereo float32/48 kHz, index continuity, exact route, finite PCM,
peek/drop custody and actual buffer geometry. Carry the verified monitor's
native index through attachment and subsequent reads. Verify the owned
player's sink-input route as well as the recording stream's route; a moved
or unavailable player route invalidates its transport grades. Independently
matched monitor PCM may remain as explicitly unattributed diagnostics, with
no passing transport claim or failing tail/bookmark verdict derived from an
uncertain route. This is a
read-only check of the controlled player, with no routing-policy change.
Reject holes, corrupt indexes, route changes and missing capture coverage.
Require an observed trailing quiet boundary before grading a content end;
a capture ending inside active audio is unqualified. Select live matching
windows by a bounded frame span, independent of native fragment splitting.
Drain independently of analysis.
These indexes establish captured stream-frame continuity, not WASAPI device
timestamps. Original waveform seek/tail/drift and bookmark thresholds remain
unchanged. [Pulse fragment contract](https://freedesktop.org/software/pulseaudio/doxygen/stream_8h.html).

The native startup control observed a new owned input with `PA_INVALID_INDEX`
before it was assigned to the expected sink. This is an unassigned creation
state, not an observed move to another sink. Coalesced asynchronous changes
still leave initial routing coverage unqualified. A consumer read index is
not a production boundary for buffered PCM and cannot turn that unknown
history into a qualified route. Retain the detected source positions,
correlations and sample-domain end-boundary evidence separately, without
relaxing any acceptance criterion. A fuller PipeWire graph observer is a
possible future qualification instrument, not an application dependency
introduced by this follow-up.

Retain signed latency and completed timing-update brackets as diagnostics.
Monitor latency can be negative. PipeWire's timing reply does not expose the
age of its last process snapshot, so a short control round trip or configured
graph-buffer limit does not establish a wall-clock uncertainty bound. Keep
absolute telemetry and post-stop timing unqualified without that evidence.
Do not reuse the Windows 5 ms timestamp check or fabricate precise timestamps
from cumulative frame counts. Preserve skipped dependent resumes when their
Stop prerequisites are unqualified. The Linux aggregate cannot pass until
all required fields have qualified evidence. Preserve failure precedence:
a measured applicable failure remains a failure, while missing timing
evidence is unqualified.
[Pulse timing contract](https://freedesktop.org/software/pulseaudio/doxygen/streams.html),
[PipeWire 1.4.2 timing implementation](https://github.com/PipeWire/pipewire/blob/1.4.2/src/modules/module-protocol-pulse/pulse-server.c#L2042-L2081).

The native three-round Linux run produced 21 planned rows, 18 launched
players and three prerequisite-skipped resumes. All 18 valid captures
retained matched, explicitly unattributed source observations with quiet end
boundaries and zero waveform drift. Every player/capture retired. Complete
initial routing coverage and absolute timing remain unqualified, so this
does not qualify the Linux playback profile. Separate controls cover natural,
playing-Stop and 20 ms startup-Stop process behavior. The user has no connected
Linux output and no available macOS host; neither inherits the Windows
listener pass. All receipts were retained before removing the isolated
remote environment. Linux remains unqualified and unsupported for the first
delivery; the Windows-first amendment defines the revised Task 41176 gate.
See the updated [evidence report](../../Docs/Testing/watchlist-article-audio-qualification.md).

## Controlled adapter and shared output evidence, 2026-11-24

TASK-41177 implements the Windows retained-WAV adapter and app-owned admission
boundary. The adapter is separate from ordinary OS file delegation. Its native
controls and transfers are recorded in the
[controlled-output report](../../Docs/Testing/watchlist-controlled-audio-qualification.md).
The canonical 12 controls include seven independently matched Natural runs,
one-frame remaining transport with waveform fidelity explicitly unqualified,
exact logical EOF without launch, and startup/active/padding Stop. Saved Resume
reuses the same file, digest and identity. An owned-process premature zero-exit
fault truncates the observed source at 199.813 ms of 2,500 ms and returns
Uncertain, rather than Natural.

The six transfer conditions ran three times each through the actual controlled
and streaming-PCM adapters. Fifteen independently matched old/new source pairs
have positive endpoint gaps of 78.317–586.411 ms, beyond the unchanged 40 ms
combined edge allowance. Three startup cases independently observed a quiet old
channel. All final matrix rows record positive cleanup and zero channel-isolation
RMS. This qualifies the 250 ms controlled post-reap settling guard for the
recorded Windows default-render profile under the existing 1.5 s Stop bound.
It does not establish a hard real-time bound, per-run acoustic silence, OS-player
control, other devices or portable controlled playback.

Production callers reserve intent before asynchronous preparation and retain
exact native run ownership through cancellation. Failed replacement preparation
preserves healthy output. Realtime keeps its microphone gated until positive
output retirement; video EOF retains the audible process pair until it drains.
An external handoff remains unresolved until the exact acknowledgement. These
paths have guarded targeted tests; synthetic adapter/PCM endpoint transfers and
mounted acknowledgement tests provide different evidence and are not conflated.

The local `recovery_scope_uncertain` pytest failure was an execution-environment
admission failure, not a recovery-policy defect. The normal account with a fresh,
native-verified private home subtree and process-local TEMP/basetemp passes the
unchanged conftest and real-profile guards. `tools/run_guarded_pytest.py` records
that targeted workflow; no existing profile, marker or ACL is repaired or bypassed.
It permits only resolved repository test targets and bounded selectors/reporting
options, pins the repository pytest configuration and controls inherited pytest
option/plugin variables. Conflicting storage or conftest inputs never reach a
child process. Negative or expired controlled retirement publishes blocked
custody immediately; late positive proof can release only that exact lease.
The controlled adapter also checks native private permissions before reading.
New PCM-to-WAV copies set native owner-only permissions on their retained new
descriptor before writing any header or audio bytes.

Article storage, synthesis/queue orchestration and listening controls remain
TASK-41178 onward. This output foundation does not close issue #2864.

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

Initial [Windows qualification evidence](../../Docs/Testing/watchlist-article-audio-qualification.md)
found missing final content despite successful process exits. Subsequent fixed-tail
controls and the wrapper-based Windows matrix remain insufficient for adoption.
The first packet-based attempts rejected the active background mix under the
existing baseline limit. After the user muted that session, all 21 Windows
cases passed with unchanged thresholds. This qualifies the recorded Windows
profile only. The Windows-first amendment's current-profile startup-Stop
supplement now passes and unlocks dependent implementation. Other platforms
await independent qualification. Windows now has explicit human listener
confirmation; no instrumented acoustic measurement was performed.
