# Watchlist Article Listening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speak saved Watchlists text, queue cited excerpts, interrupt/resume accurately, and restore saved listening manually after restart.

**Architecture:** SubscriptionsDB owns one profile-local session, entry bookmarks and source-bound audio metadata. An app-scoped service coordinates existing TTS and output adapters, while screens send intents and render snapshots. Qualify installed ffplay before building dependent storage/UI.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite, existing TTS adapters/WAV helpers, installed ffplay; no new application runtime package.

**Spec:** [approved design](../specs/2026-11-15-watchlist-article-listening-design.md)

ADR required: yes
ADR path: [backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md](../../../backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md)
Reason: durable domain state, audio identity/lifecycle, output ownership and backup contracts cross module boundaries.

## Global Constraints

- Listen speaks the stored excerpt. A feed excerpt is identified as such.
- A brief can queue all cited excerpts in first-citation order. Each article can also play individually.
- Queue order and playback positions survive restarting Chatbook.
- Reopening restores a paused listening session. The user presses Resume to start playback.
- The Watchlists database owns the queue and playback state.
- The first implementation covers local Watchlists articles and locally saved briefings.
- Graceful resume meets the 250 ms target on the qualified Windows profile. Linux/macOS controlled article listening stays unavailable pending independent qualification. Crash tests use the one-second checkpoint bound.
- The initial controlled WAV shape is PCM16 stereo at 48 kHz with 250 ms trailing zero PCM. Normalize provider output before publication within existing aggregate memory bounds; bookmarks and artifact digests name that retained normalized file.
- Start with one synthesis in flight and no speculative prefetch.
- Unreferenced audio: seven days, 512 MiB LRU; referenced audio survives eviction.
- No automatic read-state changes, full-page fetching, server-ID fallback, browser audio streaming or new provider consent policy.
- Use private relative asset paths, parameterized SQL, revision/generation fencing, existing TTS admission and backup quiescence.
- New UI composes ADR-150 tokens and ADR-161 patterns; verify 80x24, focus and truthful hints. Do not edit generated CSS directly.
- Targeted tests only. Native evidence is separate from dummy output, simulation, provider submission and ordinary process exit.
- The 2026-11-23 authorization fixes PR #3741 review findings, changes the first delivery to Windows-only, completes current-profile startup-Stop qualification, and starts **Task 2** only if that Windows gate passes. Tasks 3-7 retain their dependency order. Do not implement around a failed or unqualified supported backend.

## Review Focus

1. A player exits successfully before the final sound reaches output: verify captured tail coverage separately from process exit (Task 1).
2. Stop or old-screen cleanup races a pending launch: a retired owner must neither start output nor stop the next owner (Task 2).
3. Source deletion races snapshot admission or late synthesis: no deleted private text/audio may be republished (Tasks 3-5).
4. The default voice/reference changes between queue creation and restart: pending speech must retain the old valid selection or pause explicitly (Task 4).
5. An unavailable first/last entry, missing restored WAV or completed queue must never create a retry/autoplay loop (Tasks 5-7).

## File and ownership map

| Unit | Files and responsibility |
| --- | --- |
| Qualification | `tools/evidence/watchlist_article_audio/qualification.py`, adjacent `test_qualification.py`, and a dated evidence report; synthetic fixtures, bounded stats parser, endpoint measurement and numerical results only |
| Controlled output | `TTS/audio_player.py`, new `TTS/controlled_wav_playback.py` and `TTS/audio_output_ownership.py`; owned ffplay adapter and serialized owner-scoped admission |
| Existing outputs | `Event_Handlers/TTS_Events/tts_events.py`, `Audio/streaming_sink.py`, `UI/Speech/speech_playback_mixin.py`, `Media_Playback/player_pipeline.py`, `UI/Screens/video_player_screen.py`, Watchlists cast handlers; adapt actual start/stop/cleanup boundaries |
| Durable listening | `DB/Subscriptions_DB.py`, next `DB/migrations/subscriptions_vN_to_vNplus1_article_listening.sql`, new `Subscriptions/article_listening_store.py` and `article_audio_assets.py`; SQL, private profile lock, assets/publication/deletion |
| Preparation | new `Subscriptions/article_speech.py` and `article_listening_audio.py`; bounded text conversion, frozen voices and source-bound synthesis/cache |
| Runtime | new `Subscriptions/article_listening_service.py`, `app_service_wiring.py`, `app_services.py`, `app_shutdown.py`; transport state machine, checkpoints, runtime authority and lifecycle |
| Projection | `UI/Watchlists_Modules/content_pane.py`, `artifacts_pane.py`, new `listening_transport.py`, `UI/Screens/watchlists_collections_screen.py`, `app_command_providers.py`, `css/screens/screen_feature_watchlists.tcss` |
| Recovery | `Subscriptions/recovery.py`, `Backup_Recovery/profile_paths.py`, `detached_layout.py`, `staging.py`, `runtime_maintenance.py`, `storage_admission.py`, `DB/recovery_core.py`, `DB/recovery_operations.py`, `backlog/docs/backup-recovery-owner-inventory.md` |

Paths in this map are relative to `tldw_chatbook/` except qualification, migrations under that package, tests, and documentation. Confirm existing file names against the implementation base before edits; a path correction is not permission to change a subsystem's ownership.

## Delivery order and execution rules

Each numbered task is an atomic reviewable delivery. Its Backlog task is the status/AC owner; later tasks reference only already-created earlier IDs. Before starting a task, set it In Progress via CLI, read it, and add its Implementation Plan including the ADR check. Add Implementation Notes and mark Done only after its AC, relevant tests/static checks and review pass. Keep qualification unfinished while any required platform/evidence is unqualified.

The existing linked worktree on `codex/watchlist-article-audio-design` provides isolation. Plan baseline is `269703654630ddc3740f91ffcc4e152ecb092f2e`; source design baseline is `ecad404bd4567b57139716a6deaa6b37fe833749`. Recheck current dev before production implementation, including schema version and independently moving audio/recovery code. Do not merge/rebase the worktree while qualification owns a live child/capture.

Use the existing Python environment explicitly on Windows:

```powershell
& 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe' <arguments>
```

Commands below use `python` as shorthand for that verified interpreter. Check subprocess output as UTF-8 explicitly. Generated WAVs, raw telemetry, isolated test dependencies and numerical run receipts belong under this plan's git-ignored `.superpowers/` workspace. Commit only synthetic fixture recipes, tool source, tests and the sanitized summary report.

---

### Task 1: Qualify installed ffplay before feature work (TASK-41176)

**Files:** Create `tools/evidence/watchlist_article_audio/qualification.py`, adjacent `test_qualification.py`, `Docs/Testing/watchlist-article-audio-qualification.md`; update ADR/design evidence links.

**Interfaces:**
- Produces `ProgressObservation(position_s: float, observed_at_s: float, generation: int)` and `parse_stats_position(record: bytes) -> float | None`.
- Produces a bounded `StatsReader.feed(data: bytes, *, generation: int, observed_at_s: float) -> list[ProgressObservation]`, plus explicit freshness/generation selection.
- Produces CLI `qualification.py --output-dir PATH --capture-loopback [--ffplay PATH]`. Without capture, run only parser/fixture checks and report native audio unqualified. JSON receipts contain build/hash/platform, numerical offsets/tail coverage/stop latency and independent pass/fail/unqualified fields; never raw captured audio or endpoint names.
- Consumes installed ffplay and synthetic PCM16 WAVs only. Optional SoundCard 0.4.6 is an isolated qualification helper; existing NumPy/SciPy/CFFI support measurement. No app/provider imports or paid synthesis.

- [x] **Step 1: Write failing parser/measurement tests.** Use captured-format fixtures, split CR/LF records, NaN/infinity/negative positions, arbitrary log text, overlong records, retired generations and stale observations. Assert no accepted observation for invalid/stale data and bounded retained bytes after oversized input. A seeded synthetic delayed/resampled signal must recover its known source offset within 10 ms; unrelated/silent data must not claim a match. Start with minimal empty helper stubs so failures assert missing behavior rather than a broken environment.
- [x] **Step 2: Observe RED.** Run `python -m unittest discover -s tools/evidence/watchlist_article_audio -p test_qualification.py -v`. Expect assertion failures for valid stats/known-offset recovery, not unrelated dependency errors.
- [x] **Step 3: Implement only the qualification tooling.** Use safe fixed argv, `-nodisp -autoexit -stats -sync audio -ss <offset>`, private synthetic WAVs, incremental bounded stderr reads and owned process cleanup. Resolve the real executable beyond a package-manager shim and record hash/version. Do not patch production playback yet.
- [x] **Step 4: Observe GREEN and static checks.** Repeat the exact unit command; run `python -m ruff check tools/evidence/watchlist_article_audio` and `python -m ruff format --check tools/evidence/watchlist_article_audio`. Expected: all pass. Do not broaden to the full test suite.
- [x] **Step 5: Run native Windows qualification.** Capture only the default render endpoint's stereo loopback, never a microphone. Use bounded in-memory synthetic marker capture, reject a noisy/unmatched baseline and discard captured samples. Exercise zero and nonzero starts (including repeated resume), 120 ms and 500 ms clips, near-EOF seek, last-marker delivery, telemetry versus matched media position, Stop during startup and during playback, and retired-generation observations. Observe process return and post-stop silence separately. Target seek error <=250 ms; use 1.5 s stop/kill bounds, report measured latency rather than claiming a hard real-time guarantee.
- [x] **Step 6: Record honest gate results.** Separate endpoint-loopback measurements from acoustic/listener confirmation. Record actual executable build/OS and every run count; don't average away a failing seek/tail case. The Windows-first gate requires the retained 21-case matrix, separate Windows listener confirmation and current-profile startup-Stop control. Windows digital results cannot qualify macOS/Linux; keep them unqualified and unsupported for this delivery. Portable expansion needs their native matrix/listener evidence later. A dummy driver can test process plumbing only.
- [x] **Step 7: Review and commit qualification.** Independent review checks measurement alignment, false-positive controls, child cleanup and evidence claims. Update notes/lessons for a generalizable trap supported by the run. Commit the source/tests/report after focused checks; keep Task 1 open until the full gate passes.

#### Authorized Windows trailing-silence follow-up

- [x] Extend the qualification tool with fixed 250 ms zero-PCM padding and one-to-three bounded rounds. Preserve original content bytes and frame count separately from the padded file; leave the unpadded control available.
- [x] Red-green test exact padding, short near-EOF measurement controls, fresh frame bookmarks and same-artifact Stop/resume orchestration. Existing 250 ms seek and 20 ms final-content oracles remain unchanged.
- [x] Run three rounds of the minimal padded matrix: 120/500 ms clips, nonzero seek, 140/60 ms remaining near EOF, and playing Stop followed by the captured-bookmark resume of the identical WAV. The target is 21 launches and 18 natural completions; the final attempt launched 20 players and correctly skipped one dependent resume. Keep failures and unmatched cases; do not retry to manufacture passes.
- [x] Independently review code and numerical evidence, update ADR367 with the exact Windows scope and logical-content/padded-file contract, and record the result in the qualification report. The wrapper-based result was unqualified; the corrected packet-based 21-case rerun passes for the recorded Windows profile. Task1 and the cross-platform feature gate remain open until their remaining evidence exists.

#### Authorized listener and native Linux follow-up, 2026-11-23

- [x] Repeat the seven Windows cases for a listener. Record the user's audible-playback, clean-Stop and finished-Resume confirmation separately from numerical loopback measurements; keep the missed first attempt and its one unqualified baseline.
- [x] Inspect the supplied native Linux host's actual ffplay build and default hardware sink/monitor relationship. Capture no microphone and change no system audio configuration.
- [x] Isolate qualification-only dependencies and files in a private temporary directory on that host. Diagnose and test Linux route/timing assumptions before treating its capture as qualified. Amend ADR367 for any changed measurement contract and retain the existing numeric thresholds.
- [x] Run the same 21 planned cases and the separate startup-Stop control, preserving unknown native measurement fields. The matrix has 18 launched players and three prerequisite-skipped resumes; all 18 matrix captures retained matched but unattributed source evidence. Record exact scope, preserve initial unqualified controls and clean up owned remote resources after collecting receipts. Native Linux and macOS cannot inherit Windows or listener passes.

#### Authorized Windows-first review follow-up, 2026-11-23

- [x] Add RED/GREEN regressions and fix aggregate grading so every measured telemetry failure remains a failure, including short clips and playing Stop; optional missing observations keep their prior eligibility rules. Remove duplicate evidence keys without changing values and correct stale listener wording.
- [x] Run a separate three-round launch-relative 20 ms startup-Stop control against the same retained padded WAV, direct ffplay executable and WASAPI packet path. Keep the existing 21-case matrix unchanged. Require quiet prelaunch and post-Stop coverage, process return and observed last-signal delay within 1.5 s, confirmed player/capture retirement, and no applicable measured failure. Record every rejected/unqualified attempt; do not cancel sends or playback with an outer test deadline.
- [x] Run targeted tests/static checks and independent review; preserve sanitized numerical evidence and update the Windows release gate separately from the still-open portable gate. Mark Task 1 Done only if its revised Windows-first AC all pass, then put Task 2 In Progress through the CLI and add its implementation plan.

### Task 2: Integrate controlled WAV output with shared ownership (TASK-41177)

**Prerequisite:** TASK-41176 passed for the Windows-first profile; amend ADR first if the backend changes. Linux/macOS remain unavailable for controlled article listening.

**Initial execution increment:** After the gate passes, implement and review only the shared exact-owner lease coordinator and its targeted tests. Keep it disconnected from production outputs until the controlled adapter and every actual producer can participate. Task 2 remains In Progress; an isolated lease test does not satisfy app-wide transfer or native playback AC. Preflight identified additional owners: provider realtime PCM, speculative duplex sessions, first-run samples, artifact/Library replay, media seek and quick exit. Their real retirement boundaries, including failed-close custody and external-handoff acknowledgement, remain required in the production cutover. Recheck current dev before that cutover; record the base of this isolated increment.

**Files:** New `TTS/controlled_wav_playback.py`, `TTS/audio_output_ownership.py`; modify the existing output files in the map and `app.py`. Tests: `Tests/TTS/test_controlled_wav_playback.py`, `test_audio_output_ownership.py`, affected Console/Speech/media playback tests.

**Interfaces:**
- `AudioOutputOwnership.acquire(owner: str, retire: Callable[[], Awaitable[bool]]) -> AudioOutputLease`; a lease has `generation: int`, `is_current() -> bool`, `release() -> None`. A failed retirement leaves admission blocked. Serialize intent through the owning app, not a new player singleton.
- `ControlledWavPlayback.start(path: Path, *, start_frame: int, sample_rate: int, generation: int) -> None`, `position() -> ProgressObservation | None`, `stop() -> bool`; publish typed completion distinguishing natural EOF, stopped, failed and uncertain. Calls involving process waits execute off-loop.

- [ ] **Step 1: Write failing tests.** Assert all file/stream/media start orders retire the previous output before admitting the next; a startup-stop race never launches late; old unmount/cleanup cannot stop a new lease; failed kill retains ownership; missing ffplay or unsupported OS refuses before synthesis. Validate PCM16 stereo 48 kHz WAV duration/digest/padding before launch and reject stale telemetry/early successful exit as completion evidence. Exact logical-EOF Resume launches no player; Stop during padding cannot publish natural completion or release a newer owner.
- [ ] **Step 2: Observe RED.** Run `python -m pytest Tests/TTS/test_controlled_wav_playback.py Tests/TTS/test_audio_output_ownership.py -q`; verify failures name missing contract behavior.
- [ ] **Step 3: Implement the qualified adapter and owner.** Promote only needed qualified parser logic, retain legacy ordinary playback behavior, wrap actual streaming/file/media acquisition and owner-scoped cleanup, and preserve external-handoff takeover acknowledgement.
- [ ] **Step 4: Observe GREEN.** Run those tests plus existing `Tests/TTS/test_audio_player_formats.py` and mapped affected surface tests; verify native cross-surface takeover with synthetic audio. Run Ruff on changed Python and the repository's targeted architecture checks for message-loop waits.
- [ ] **Step 5: Review and commit.** Record caller coverage and native evidence in task notes; commit `feat: coordinate controlled audio playback ownership`.

### Task 3: Persist listening state and source-owned assets safely (TASK-41178)

**Prerequisite:** TASK-41177. **Files:** Subscriptions DB/migration, `article_listening_store.py`, `article_audio_assets.py`, deletion services and all recovery files in the map. Tests: new `Tests/Subscriptions/test_article_listening_store.py`, `test_article_audio_assets.py`, applicable `Tests/DB/` migration and `Tests/Backup_Recovery/` contracts.

**Interfaces:**
- `ArticleListeningStore.load_session() -> ListeningSession | None`; immutable records include session revision, entry frame offsets, frozen voice and source authority.
- `replace_session(candidate: ListeningCandidate, *, expected_revision: int | None) -> ListeningSession`, `checkpoint(update: ListeningCheckpoint, *, expected_revision: int) -> ListeningSession`; stale revisions fail without mutation. State/types live with this store, not in `app.py`.
- `ArticleAudioAssets.publish(candidate: GeneratedArticleAudio, *, source_item_id: int, generation: int) -> ArticleAudio`; validate private relative identity, WAV shape, source validity, digest and ownership before publication/reference attachment.
- Provide a profile-local nonblocking listening mutation lease using existing private-lock primitives; service, deletion, cleanup and restore share it. Register the writer with existing recovery quiescence.

- [ ] **Step 1: Write failing real-SQLite/file tests.** Exercise old-profile upgrade and migration rollback, two independently opened DBs, revision collisions, atomic cursor/two-bookmark updates, source deletion during candidate admission and artifact publication, source-bound sharing, all-unavailable replacement, missing/corrupt WAV, publication failure/orphans, lock crash release, and fail-busy destructive mutation. Deleting membership preserves sources; deleting a brief preserves an unrelated valid foreground reading.
- [ ] **Step 2: Observe RED.** Run only the two new Subscriptions test files and selected migration nodes. Expect missing tables/contracts, with existing base migration cases still green.
- [ ] **Step 3: Implement store, migration and asset lifecycle.** Allocate next schema version at the implementation base (v3 here), add its SQL artifact and indexes/allowlists. Enforce source existence/generation transactionally, use existing private atomic writers/path validation, and reconcile only after ownership. Enforce seven-day/512 MiB unreferenced LRU while protecting references.
- [ ] **Step 4: Integrate backup atomically.** Extend `Subscriptions/recovery.py::_BriefingAudio`'s `subscriptions.assets` declaration, `DB/recovery_operations.py::_SubscriptionsAdapter`'s exact schema catalogs/versions, migration policy, discover/relocate/dependency validation and SQLite authorizer, core dependency discovery, inventory/census, detached layout and staging. Register publication in `RuntimeMaintenance.settle_producers/resume` and `acquire_storage`. Test owners present AND absent, DB-only backup, assets backup, rollback and relocation; restoration never starts audio. Missing files retain regenerable entries with offsets reset only by explicit regeneration. Existing focused suites include `test_briefing_audio_dependencies.py`, `test_briefing_audio_restore.py`, `test_detached_layout.py`, `test_runtime_maintenance.py` under `Tests/Backup_Recovery/` and `Tests/Architecture/test_backup_owner_inventory.py`.
- [ ] **Step 5: Observe GREEN and review.** Run targeted Subscriptions/migration tests and all changed Backup_Recovery owner suites, including optional-owner absence cases. Run relevant derived-artifact guards and Ruff. Record any baseline failures separately with a base comparison before calling them pre-existing. Review and commit `feat: persist watchlist listening and owned audio assets`.

### Task 4: Prepare bounded article speech and frozen-voice audio (TASK-41179)

**Prerequisite:** TASK-41178. **Files:** New `Subscriptions/article_speech.py`, `article_listening_audio.py`; small extraction from `briefing_audio.py` only where reuse requires it; existing HTML and TTS request/admission helpers. Tests: new `Tests/Subscriptions/test_article_speech.py`, `test_article_listening_audio.py`; existing briefing synthesis regressions.

**Interfaces:**
- `prepare_article_speech(body: str, *, content_format: str) -> PreparedArticleSpeech` carries normalized text, UTF-8 hash and converter version. Admission limits: 256 KiB UTF-8 input per entry, 2 MiB total snapshot text per queue, 256 entries, 128 TTS chunks per article. Reject whole over-limit candidates; keep unavailable placeholders for ordinary missing bodies.
- `freeze_listening_voice(resolution: CharacterTTSRequestResolution | VoiceSelection) -> FrozenListeningVoice` retains allowlisted nonsecret settings and dependency identity; `resolve_listening_voice(snapshot: FrozenListeningVoice) -> VoiceSelection` revalidates via existing admission. Never reconstruct authority from a saved reference pathname.
- `synthesize_article(entry: ListeningEntry, *, generation: int) -> GeneratedArticleAudio` uses the store/assets contract. At most one listening request in flight; enforce existing 64 MiB `MAX_BUFFERED_AUDIO_BYTES` across encoded/decoded/stitching buffers, counting retained copies. No partial artifact is playable.

- [ ] **Step 1: Write failing tests.** Cover text/HTML/Markdown including scripts/styles, link labels versus targets, images, malformed markup/control codes, code prose, empty conversion, exact boundary and over-limit input/queue/chunk counts. Assert full long-body coverage without 800-character truncation. Change defaults/profile/reference after enqueue and restart; verify frozen valid voice or explicit pause, no silent substitution. Retained valid audio spends zero provider calls; cancellation/deletion cannot republish or autoplay; aggregate limit stops before retaining the next chunk.
- [ ] **Step 2: Observe RED.** Run the two new test files with real WAV fixtures and fake provider transport only; expect missing preparation/synthesis behavior.
- [ ] **Step 3: Implement minimal preparation and orchestration.** Reuse `Subscriptions/html_text.py`, existing Markdown parser, exact/native and legacy TTS admission/chunking/WAV concatenation. Normalize provider audio to PCM16 stereo 48 kHz and append the qualified tail once before publication; bound retained conversion/stitching copies and test mono/other-rate input. Check required optional dependencies and Windows controlled-playback capability before the first request. Record converter version and original item authority; do work off-loop. No speculative prefetch.
- [ ] **Step 4: Observe GREEN.** Run new tests and `Tests/Subscriptions/test_briefing_audio_synthesis.py`, `Tests/TTS/test_audio_limits.py` plus affected resolver/admission nodes. Verify one configured provider manually with nonprivate synthetic prose; no speech text, reference paths or raw provider errors enter diagnostics. Run Ruff.
- [ ] **Step 5: Review and commit** `feat: synthesize bounded saved article speech` with request-count and limit evidence.

### Task 5: Own the resumable session and interruption state machine (TASK-41180)

**Prerequisite:** TASK-41179. **Files:** `Subscriptions/article_listening_service.py`, `app.py`, runtime-policy/profile hooks and source/session deletion coordination. Tests: new `Tests/Subscriptions/test_article_listening_service.py` plus focused app lifecycle tests.

**Interfaces:**
- `ArticleListeningService.play_brief(briefing_id: int, *, replace: bool = False)`, `listen(item_id: str, *, runtime_backend: str)`, `pause()`, `resume()`, `stop()`, `skip()`, `clear()`, `play_again()`, `close()` are async intents; `snapshot() -> ListeningSnapshot` is an immutable UI projection.
- Compose store/assets, one synthesis owner and the existing output owner. Consume all pages of saved `briefing_items` citation order through validated local authority; `get_briefing_provenance_for_agent` defaults to 50 entries and its `cited_has_more` must not truncate the queue. Use existing legacy citation extraction plus membership validation only when provenance is absent.
- `ListeningSnapshot` includes transport phase, current/suspended title, excerpt flag, queue position/count, known timing, preparation progress, unavailable/error code, revision and action availability. It contains no provider secrets/reference payloads and grants no mutation authority.

- [ ] **Step 1: Write failing state-machine tests.** Queue order/dedup/unavailable skipping, including more than 50 cited items; a playing versus paused queue interrupted twice; independent offsets when same item shares WAV; repeated Listen; foreground completion/Skip versus Stop/failure; exact-EOF Resume and Stop inside padding with no accidental automatic queue return; startup during each role; completed/all-unavailable behavior; failed replacement; one-second coalescing and atomic terminal checkpoint; profile/runtime away/back races, lock holder versus second instance, cancelled native inference finishing late, deletion and stale callbacks.
- [ ] **Step 2: Observe RED.** Run the new service file with real SQLite/files and a deterministic controlled output seam; failures must identify state transitions, not depend on wall-clock sleep.
- [ ] **Step 3: Implement service/lifecycle.** Wire one app-scoped service through `ServiceWiringMixin._wire_watchlists_and_notifications_services` and `app_services.py`; checkpoint/drain before TTS/DB teardown in `app_shutdown.py::_drain_app_unmount_services`. Use private profile ownership, revision/generation-checked commands, off-loop checkpoint writes and actual player observations. Keep the old cleanup owner until native work returns; block another listening synthesis in that interval. On startup publish paused state without synthesis; on runtime change retire permission to publish/start.
- [ ] **Step 4: Observe GREEN.** Run service and affected lifecycle tests; kill a private native probe between durable checkpoints and confirm manual restoration from the same retained WAV. Verify no automatic read mark or server/local ID crossover. Run Ruff and changed architecture checks.
- [ ] **Step 5: Review and commit** `feat: coordinate resumable article listening sessions`.

### Task 6: Expose reader, brief and global transport controls (TASK-41181)

**Prerequisite:** TASK-41180. **Files:** reader/artifacts panes, new transport widget, Watchlists screen, palette provider/app action registration, source CSS modules and generated CSS via `build_css.py`; user guide/architecture. Tests: new `Tests/UI/test_watchlist_article_listening.py` and affected Watchlists/palette tests.

**Interfaces:**
- Reader posts `ArticleListenRequested(item_id: str, runtime_backend: str)`; citation selection posts its separate Listen intent, never `CitationActivated` (the existing Open path changes view/read state).
- Transport consumes `ListeningSnapshot` and posts typed intents only. Reader `l` is the candidate binding after checking descendants/global keys. Palette exposes live Pause/Resume/Stop actions through the same service.

- [ ] **Step 1: Write failing mounted tests.** Listen on reader and selected citation; Play cited excerpts count/unavailable preview; same queue Resume/different queue replacement confirmation; source body excerpt/output-host notice; preparation state; suspended queue labels; restored Resume versus completed Play again; actionable failures and Clear. Assert brief stays visible, read state unchanged, footer hints truthful, selection/focus retained and old-screen unmount cannot stop new output.
- [ ] **Step 2: Observe RED.** Run only new mounted nodes and relevant existing Watchlists citation/palette nodes; verify missing controls/actions cause failures.
- [ ] **Step 3: Implement projection.** Read `backlog/docs/design-language.md`, component patterns and keybinding ADR before UI edits. Compose existing token-backed compact controls with no ad-hoc sizes/colors. Keep handlers thin and provider work off-loop. Disable unsupported server/change items with a reason; expose host-local audio location in serve mode before play.
- [ ] **Step 4: Observe GREEN.** Run the mounted tests at 80x24 and wide sizes, palette/keyboard tests, `Tests/UI/test_design_token_governance.py` and CSS bundle sync guard after rebuilding. Native terminal/serve checks establish focus, progress and host-output notice; no remote-browser streaming claim. Run Ruff on changed Python.
- [ ] **Step 5: Review and commit** `feat: add watchlist article listening controls` with mounted geometry/focus and native evidence.

### Task 7: Qualify end-to-end recovery and document release limits (TASK-41182)

**Prerequisite:** TASK-41181. **Files:** focused end-to-end/recovery tests, evidence report and `Docs/User_Guide/watchlists.md`, `Docs/Architecture/subscriptions-watchlists.md`; targeted fixes only if a reproduced regression requires them.

**Interfaces:** Consumes the service intents/snapshot, backup restore API and qualified controlled output from prior deliveries. Produces the final issue acceptance evidence and supported-build matrix.

- [ ] **Step 1: Add any missing failing integration cases.** Complete cross-boundary deletion/publication, startup/foreground resume, short final queue item, missing restored WAV, backup owner absent/present, two-process destructive mutation and live cross-surface preemption cases without duplicating already-proven pure tests.
- [ ] **Step 2: Run the targeted regression set.** Execute new article-listening files and affected existing TTS/Subscriptions/UI/Backup_Recovery nodes; run relevant schema/owner/diagnostic/CSS guards. A full test sweep needs explicit user opt-in.
- [ ] **Step 3: Run the native workflow.** With private synthetic saved articles and a configured provider: queue in citation order, interrupt mid-sentence, resume the same retained position, Stop/Resume/Skip/Clear, restart during both roles, kill/recover, switch runtime/profile, delete/restore, missing/corrupt audio and unrelated output takeover. Repeat native playback cases on the qualified Windows profile; platform expansion needs independent native evidence. Record measured errors, exact scope and failures.
- [ ] **Step 4: Review and close out.** Fresh independent review checks spec coverage, source/privacy boundaries and truthful evidence. Resolve important findings, rerun affected checks, update task notes/ADR/docs and commit. Only claim #2864 resolved when all agreed behavior and native qualification pass; issue closure/PR integration follows the user's workstream authorization and current repository merge rules.

## Self-review and gate status

The seven deliveries cover all ten design acceptance outcomes. The five Review Focus cases each have an owning test step above. Store, preparation, output and service signatures match their consumers. Tasks are ordered so a storage/voice/UI implementation cannot bypass qualification. SQL names and schema number are intentionally selected against the implementation base, while their ownership and rollback contracts are fixed here.

Gate status: [native evidence](../../Testing/watchlist-article-audio-qualification.md) records the passing 21-case Windows matrix, explicit Windows listener confirmation and passing current-profile three-round startup-Stop supplement. The Windows-first release gate passes; the two earlier supplemental attempts remain preserved and all nine players/captures retired. Linux route/timing/listener evidence and native macOS evidence remain unqualified; they are unsupported for this first delivery and the portable gate remains open. Task 1 is Done under its revised Windows-first AC. Task 2 is In Progress for the isolated ownership increment; Tasks 3-7 have not started.
