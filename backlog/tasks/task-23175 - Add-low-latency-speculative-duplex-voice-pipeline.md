---
id: TASK-23175
title: Add low-latency speculative duplex voice pipeline
status: In Progress
assignee: []
created_date: '2026-08-28 23:49'
updated_date: '2026-09-08 06:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce perceived voice-response latency by starting a cancellable speculative reply after a short configurable silence while preserving natural turn extension, reliable echo cancellation, and safe fallback behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bounded configurable silence threshold defaults to 700 ms and starts speculative generation from rolling transcription.
- [ ] #2 Speech detected while a speculative reply is generating or playing cancels stale work and extends the same user turn with the latest transcript.
- [ ] #3 Speech after assistant playback completes starts a new user turn.
- [ ] #4 Native streaming STT is preferred and rolling-window incremental transcription is supported as fallback.
- [ ] #5 TTS output remains sequential and cancellable across providers.
- [ ] #6 Reliable echo cancellation is enabled by default and unsafe full-duplex capture fails closed to half duplex.
- [ ] #7 Targeted automated tests and latency/cancellation telemetry verify the end-to-end behavior.
- [ ] #8 Physical qualification derives safety, latency, route-switch, and soak metrics from the live production audio path rather than operator-entered assertions, while retaining no captured audio or transcript content.
- [ ] #9 A route with operational AEC but no observable echo path may use full duplex only after the shared runtime proves sustained acoustic isolation; ambiguous, correlated, discontinuous, or overflowing capture remains fail-closed.
- [ ] #10 During assistant playback, VAD-positive audio on either full-duplex path is held for at most 50 ms while residual echo is classified; admitted cleaned frames reach rolling STT in original order with the first phoneme preserved, while coherent render leakage is discarded and fences the route.
- [ ] #11 Qualification source identity covers the isolation runtime, live runner, reference asset and manifest, tests, and approved design/plan; changing any listed source invalidates prior evidence, and rollout remains disabled until the complete platform/device matrix is regenerated from one committed digest.
- [ ] #12 A source-checkout-only development launcher can exercise the speculative pipeline through the visible Hands-free control without changing packaged qualification authority, and the launcher is absent from release artifacts.
- [ ] #13 A blocked UI event loop does not stall live capture processing or interruption fencing/cancellation; bounded owner-loop bridges preserve TTS cleanup, provider authority, promotion ordering, and app-wide quarantine.
- [ ] #14 The native device callback makes progress without Python/GIL entry, retains bounded capture and actual-render timing through a Python pause, fences stale output and teardown safely, and cannot reinterpret native device/data-loss faults as recoverable clock-only failures.
- [x] #15 Speculative Hands-free refuses unavailable provider entry before STT or audio starts, fences stale readiness results, and preserves drafts while reporting preparation failures with safe actionable copy and content-free categories.
- [x] #16 Audio-critical processing and local interruption fencing continue in a separate process during a bounded app-interpreter GIL hold; bounded IPC preserves original app-side provider/promotion authority, checked cleanup and process-tree quarantine, and an irrecoverably broken transport cannot remain enabled and generating.
- [x] #17 Retired speculative requests that were never issued reject provider terminal traffic while admitted requests preserve bounded valid late-record handling.
- [x] #18 Queued startup heartbeat or close traffic cannot displace the already-read bootstrap; startup control sequence and credit accounting remain valid and early close still prevents native start.
- [x] #19 The combined audio integration branch is based on current dev, includes the existing Console functionality required by the complete audio feature, preserves newer dev behavior and schema history, and passes targeted port/integration checks before a PR targets dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow `Docs/superpowers/plans/2026-08-28-low-latency-speculative-duplex-voice-pipeline.md` task by task with test-driven implementation and independent spec/quality review checkpoints.
2. Land configuration and dependency-free audio contracts before native or device imports.
3. Vendor and package the pinned WebRTC AEC3 closure, then implement the app-owned duplex transport and fail-closed preprocessor.
4. Add incremental STT, revision-aware speculative turn orchestration, sequential cancellable TTS, and canonical message persistence.
5. Wire settings and lifecycle ownership into the Console, then add observability, recovery, stress, and performance verification.
6. Follow `Docs/superpowers/plans/2026-09-02-live-physical-voice-qualification-and-isolated-headset.md` to replace asserted physical observations with live measurements and add the isolated-headset safety path at the shared preprocessor gate.
7. Regenerate all source-bound automated, soak, and physical evidence after the amended runtime and qualification source is committed.
8. Keep this task In Progress after the macOS USB run; mark it Done only after strict validation of built-in, USB, and Bluetooth evidence for `macos-arm64`, `macos-x86_64`, `windows-x86_64`, `linux-x86_64`, and `linux-aarch64` from the frozen source digest.
9. Add a Git-worktree- and exact-HEAD-bound development launcher outside the application package, prove the visible Hands-free control selects `ConsoleSpeculativeHandsFreeSession`, and keep the launcher out of wheel and sdist artifacts before one ordinary USB-headset conversation smoke test.
10. Diagnose the failed ordinary conversation from existing content-free events and device-free regressions before any further operator input. Reproduce rolling PCM timestamp quantization, preserve actual gap/overlap rejection, and expose categorical transcript/synthesis failures. Investigate scheduler interference separately; a native throughput check alone is not proof of callback continuity. This is routine repair under ADR-098 and does not change its acoustic admission policy.
11. Apply the ADR-098 local Parakeet execution-isolation amendment: one spawned, bounded session worker owns model loading, native contexts, and in-memory rolling inference. Verify process ownership, timeout/cancellation cleanup, stream identity, and the production composition with targeted tests. Compare generated-speech scheduling and sequential TTS without opening hardware before preparing any further ordinary conversation.
12. Diagnose the idle readiness failure without restarting hardware: reproduce dropped overflow evidence with continuous fake device timestamps, preserve the latched overflow cause so it cannot enter clock-only startup recovery, and profile the mounted Console using isolated test data and a simulated backend. This is a routine safety/diagnostic correction under ADR-098; do not relax timing, queue, or AEC limits. Any further runtime-boundary change requires an explicit ADR check before implementation.
13. Following user approval of the isolation direction, review `Docs/superpowers/specs/2026-09-04-speculative-voice-ui-isolation-design.md`. After written boundary approval, amend ADR-098 and write a task-by-task implementation plan before changing runtime ownership. Keep the hardware-free UI-block/cancellation gate ahead of any further ordinary conversation.
14. Execute the user-approved `Docs/superpowers/plans/2026-09-04-speculative-voice-ui-isolation.md`: one app-owned voice loop, bounded owner-loop UI/TTS handoffs, true cleanup receipts, and blocked-UI software acceptance before the ordinary conversation. Do not run qualification, route switching, repetitions, soaks, or a full suite.
15. The user approved the native-callback direction after the hardware-free GIL diagnosis. Review `Docs/superpowers/specs/2026-09-04-speculative-voice-native-callback-design.md` before implementation planning. Amend ADR-098 after written approval; preserve native record ownership, callback-clock evidence, actual render completion, cancellation and teardown fences. No hardware run is part of implementation.
16. The user approved the revised written boundary after all five review corrections. Execute `Docs/superpowers/plans/2026-09-04-speculative-voice-native-callback.md` under the accepted ADR-098 amendment, with test-first native/transport/terminal integration and task-scoped reviews before the final production-composition gate. Keep physical hardware closed during implementation.
17. Repair the four final-review findings at `4c11ef5901` under the same ADR: native output fencing before provider/TTS cleanup, capture coverage for half-duplex terminal seals, permanent native deactivation and independent checked-close observation before unrelated cleanup, and public version tuple consistency. Reproduce each before changing code; run only affected software tests and one scoped re-review before final runtime preflight. No new ADR is required.

18. Execute the written, user-approved post-smoke repair design through `Docs/superpowers/plans/2026-09-04-speculative-voice-post-smoke-repairs.md`: complete paired DSP output evidence without changing submission receipts, bounded whole-context Parakeet ownership and exact-chunk quiet closure, then content-free fault/stage diagnostics. Follow the ADR-098 clarification and targeted software RED/GREEN cases. The ordinary conversation has been consumed; do not repeat hardware, qualification, soaks or a full suite. Host-overflow causality remains unresolved and this task remains In Progress.

19. Execute the user-approved provider-readiness correction through `Docs/superpowers/plans/2026-09-05-speculative-voice-provider-readiness.md`: bounded selected-provider validation before STT/audio, stale-entry ownership fences, and safe visible preparation errors with preserved drafts. Verify only with software tests. Existing ADR-098 applies; no new architecture decision, acoustic policy change, release qualification or hardware rerun is authorized.

20. The user approved the process-isolation direction after the follow-up-turn overflow reproduction. Review `Docs/superpowers/specs/2026-09-05-speculative-voice-process-isolation-design.md`, obtain written approval, then amend ADR-098 and write an implementation plan before moving runtime ownership. Preserve app-owned original provider/promotion objects, child-local capture/AEC/VAD/transcript/turn/output processing, independent STT execution, bounded IPC and checked teardown. Verify only with software tests; no further ordinary conversation, hardware, qualification, route switching, repetitions, soaks or full suite. This supersedes the historic plan's requests to regenerate physical evidence for the current work.

21. The user approved the independently reviewed written process-isolation spec on 2026-09-05. Follow `Docs/superpowers/plans/2026-09-05-speculative-voice-process-isolation.md` under the accepted ADR-098 amendment after its plan review: extract dependency-light policy/core, add bounded private IPC and contained process lifetime, retain local STT semantics and app-owned normalized TTS/provider/promotion authority, switch the visible production control, then prove 720 ms/3 s app-GIL isolation and fatal-session shutdown with targeted software tests. No hardware, model inference, provider request, full suite or release qualification.

22. Approved bounded N1 follow-up: reproduce rejection of all three provider terminal opcodes after cancellation and retirement of a never-issued local proposal. Omit late-record bookkeeping for unadmitted requests at the existing shared retirement point; preserve admitted late records, duplicate/conflicting rejection and existing bounds. Run only affected software tests and a narrow review; no hardware, native holds, qualification, soaks or full suite.

23. Approved bounded startup-race follow-up: add a deterministic real-pipe regression that queues a heartbeat or close before the child registers its already-read bootstrap. Register bootstrap with the existing mailbox/receiver before starting pipe threads; preserve sequence, credit, priority, validation and close behavior. Verify only affected software startup/lifetime/IO tests and request a narrow review. Keep the original I1 incident unclassified because its stage/exit evidence is missing; do not rerun native groups, live audio, qualification, soaks or the full suite.

24. Approved dev integration (2026-09-06): follow `Docs/superpowers/plans/2026-09-06-speculative-voice-dev-integration.md` in the separate `codex/speculative-duplex-voice-dev` worktree. Import only committed voice-owned source from `258beb6120a5e8c84d4f83b12a39427733301e57`, retain current dev's non-voice source and schema history, and resolve shared integration contracts without reintroducing unrelated branch ancestry. Use targeted software verification only; no live conversation, hardware, qualification, route switching, repetitions, soaks, model/provider requests or full suite. Keep original feature worktree unchanged and preserve release hard-off authority.

Dev-integration ADR check:
ADR required: no new ADR for a behavior-preserving port
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Reuse the approved voice contracts and existing ADR-094 accepted-turn prerequisite. The user clarified that full audio integration includes its required existing Console functionality; preserve current dev privacy/trace/Canvas behavior instead of adding a substitute acceptance subsystem.

Startup-race ADR check:
ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Routine correction of initialization ordering within the existing private transport; no protocol, custody, timeout or runtime-boundary change.

N1 ADR check:
ADR required: no new ADR
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: Routine enforcement of the existing issued-identity/custody boundary; no protocol or authority change.

Feature ADR check:
ADR required: yes
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
Reason: The feature changes long-lived runtime boundaries, cancellation semantics, audio safety policy, and cross-module service contracts.
25. Task 3 app-side dev integration starts at `2244fc47a5e5cffb61353f3d2124a36a19ca8617` under `Docs/superpowers/plans/2026-09-06-speculative-voice-dev-integration.md`. Port only pinned source `258beb6120a5e8c84d4f83b12a39427733301e57` voice modules and narrow shared-file hunks, preserving accepted-turn ownership and current dev diagnostics/trace/privacy/Canvas behavior. Amend ADR-098 first. Confirm schema 68 and free slot 69; add the immediate-predecessor migration and connection-local insertion-only authority without replacing prior guards. Reuse actual one-pass scoped/custom PII projection before sealing artifacts, bounded immutable span metadata, and atomic canonical revision masks/omissions at winning import. Wire existing production registry/context/import callbacks; consume next-send privacy only after the exact successful winning claim, including Capture OFF, without forcing privacy into the text-only effectful fallback. Add focused RED/GREEN source/adaptation and real-factory fake-inference composition checks, retain unrelated dev behavior, then require specification and independent quality review before Task 4. Preserve all historical qualification/resource caveats; no live app/audio/provider/model/native-hold/hardware/soak/full-suite/install/config work.

ADR required: no new ADR; amend existing ADR-098.
ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`.
Reason: preserve the approved runtime/promotion boundary while integrating current dev scoped/custom trace privacy and migration slot 69; no new owner, service or table.
26. Follow `Docs/superpowers/plans/2026-09-08-voice-pr-2504-followup.md` for the authorized local PR #2504 rebase and scoped fixes. Back up the published head, consolidate its exact tree, rebase onto pinned fetched dev, and compose voice schema 70 after unchanged dev schema 69. Verify targeted SQLite/fake-software cases and derived inventory only; do not push or merge.

ADR required: no new ADR.
ADR path: backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md
Reason: behavior-preserving integration under the existing voice/trace boundary; record the migration renumbering in ADR-098.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Provider-readiness repair (2026-09-05): implemented the approved spec/plan under existing ADR-098 in c7f7e632d115 and bf1e39d73315. Visible speculative Hands-free checks the owning selected provider before worker/STT/audio; session, activation, settings and config revisions fence stale/ABA startup. Later preparation failures preserve drafts once and provide fixed safe copy and content-free categories; cancelled/stale delivery is suppressed. Independent spec review found and corrected the initial-snapshot publication race; spec re-review and code-quality review passed. Final joined targeted gate: 203 passed, zero skips, 217.72 seconds, exit 0 on bf1e39d733 with owned integration edits. All 11 changed Python files passed Ruff check/format; diff checks passed. Three known dependency warnings only. Source/lint inventories, development guide and incident lesson updated. No hardware, provider requests, app relaunch, qualifications, route changes, repetitions, soaks or full suite. Only AC15 completed; keep overall task In Progress and preserve branch/worktree. Existing app remains older code; USB playback overflow and live completion remain unresolved.

Process-isolation Task 5: retained native and rolling local STT semantics in the existing model-process service, with bounded typed results/options and actual cleanup disposition. ONNX uses an explicitly injected current-process buffer owner that reuses source/provenance/root-and-VAD lease validation; ordinary shared execution is unchanged. Failed native cleanup retains custody through exception-proof nonzero model retirement and actual parent reap. ADR-098 records the voice-only refinement, preserving ADR-025; process spec/plan and the concrete testing lesson are updated. Independent spec and quality reviews approved. Final joined targeted gate: 359 passed, zero skips, 50.91 seconds; one existing requests warning. Eight-file Ruff lint and diff checks passed; seven files are formatter-clean, with only pre-existing untouched executor formatting retained. No hardware, live model/provider execution, app relaunch, qualification, soak or full suite. Production composition and GIL acceptance remain pending; AC16 stays unchecked and task remains In Progress.

Process-isolation Task 6: original TTS synthesis, normalized PCM decoding and first actual response cleanup remain on the parent owner loop. One global PCM credit window and one shared session-wide credited cleanup lane bound transport custody; three accepted owners stay retained until receipt consumption or transport retirement. Actual cleanup is independent of acknowledgement, and failed or unexpectedly cancelled production blocks replacement. Independent spec and quality reviews resolved stale cancelled/naturally exhausted owner fencing and unsolicited cancellation gaps with RED-first regressions; both reviews approved. Final joined targeted gate: 254 passed, zero skips, 8.18 seconds; one existing requests warning. Seven-file Ruff lint/format and scoped diff checks passed. Existing ADR-098/spec document the minimal credited-receipt refinement. No hardware, app, model/provider request, qualification, repetitions, soaks or full suite. Production composition and GIL acceptance remain pending; AC16 stays unchecked and the task remains In Progress.

Process-isolation Task 7: added bounded parent-owned provider IPC, exact claim observation and retained context/draft/terminal custody. Accepted handoff preserves original context authority while using the latest transcript/revision. Two review fixes close cross-view admission during pending cleanup and require every present terminal/cleanup receipt before retirement; both independent review gates approve. Final root joined gate: 252 passed, zero skips, 40.27s; twelve owned Python files lint/format clean. Existing requests dependency warning only. Plan and testing-evidence lesson updated. Existing process-isolation ADR/spec apply; no new ADR required. Production composition and finite software-only acceptance remain Tasks 8–10. No live app/audio, qualification, soak, provider/model call or full suite. AC16 remains unchecked; task remains In Progress.

Task 8 implementation refinement: existing shared VoiceDispatchSupervisor gains observation-only wait_for_cleanup() and targeted cancelled/cross-loop/pending-to-survivor tests. Task 7 effects may legitimately finish DETACHED disposition before actual survivor exit; app quit must retain that original custody alongside process/effects/TTS/claimed work. No new cleanup owner or admission authority. Task 8 plan scope updated before implementation; existing ADR-098 applies, no new ADR needed. Mounted process composition remains in progress; no live or broad verification.

Task 8 composition exposed cancelled-provider retirement without a final data boundary: priority cleanup can overtake old deltas. The existing cleanup receipt now requires the original final published last_sequence (zero for no data); retain discard-only child custody through that boundary and actual cleanup, with consistent prior terminal end. Plan/spec/ADR-098 refined before implementation; narrow ProcessVoiceEffects + its test added to Task 8 scope. New admission must stop without stranding published writer custody. No new opcode/authority or native change. Focused RED/GREEN and full Task 8 review remain pending; no live verification.

Task 8 direct cancellation regressions: three expected failures expose closed admission blocking already-published provider writes, missing provider final sequence, and cancelled PCM receiving actual cleanup without ever reaching a final data boundary. Narrow TTS bridge/process-TTS test scope added for truthful existing pcm_end on cancellation/never-started zero. Keep actual resource close independent and the existing single end slot bounded for stalled/queued cancellation; no capacity enlargement or fabricated consumption. Plan/spec/ADR-098 updated before implementation. Corrections and independent review pending; no live or broad run.

Task8 cancellation refinement (software only): required tts_closed.last_sequence supersedes the initial cancellation pcm_end proposal. Priority cleanup can release retirement before the one uncredited end slot is consumed, so extra cancellation ends are unsafe. Record the original final published PCM sequence (zero if none); only fenced/cancelled streams use it for discard retirement, while normal playback requires actual pcm_end. Matching late cancelled ends are redundant; conflicts fail. Keep exact late-record handling bounded and actual resource close independent. Scope adds Audio/voice_phrase_sequencer.py and Tests/Audio/test_voice_process_io.py; existing Chat phrase tests are compatibility-only. Plan/spec/ADR098 updated before implementation. Task8 remains in progress; no live or full-suite testing.

Task8 joined verification exposed obsolete mutable-session preflight fixtures and a suspected real preparation-failure regression: raw parent preparation exceptions currently flow to generic transport failure, losing the original safe no-reply explanation/recovery path. Scope now includes Tests/Chat/test_console_voice_preflight.py, with actual parent/child failure proof required before direct-core fixture migration; copying removed production callbacks into tests is not sufficient. Existing category/currentness/draft/no-auto-retry behavior must remain. Self-review also found failed-terminal recovery-slot retirement before a hanging observer completed; repair is in progress. Task8 is not frozen or reviewed.

Task8 production recovery refinement: preparation failure before a handle uses existing provider_failure with final sequence zero; trusted category stays parent-local. One closed child-to-parent draft_recovery control (exact turn/revision/epoch, request/revoke, positive recovery ID, no payload) references the existing consumed draft, with existing control/two-turn bounds and observed-revocation/UI-currentness checks. This preserves original safe notices/recovery without mutable-core test copies or diagnostic misuse. Also preserve cleanup/terminal receipts after authority fencing; delivery failure and actual cleanup/settlement remain independent. Both gaps have meaningful REDs; targeted production proof is required before freeze. Plan/spec/ADR098 updated. No live testing or full suite.

Process-isolation Task 8 completed production composition beneath the visible Hands-free wrapper, with distinct guarded child PID/source proof, typed startup, legacy-dictation refusal, actual data/cleanup/claim custody, safe preparation-failure recovery and independent native/runtime shutdown. Existing ADR-098/spec record bounded final-sequence, recovery-control and cleanup-observer refinements. Independent spec review passed; quality review found and fixed accepted handoff preceding delayed cancelled provider data with paired-pipe RED/GREEN, and scoped re-review approved. Final root gate: 515 passed, zero skips, 144.06s; 26 owned Python paths lint/format and diff checks clean. Two existing requests/pydub warnings only. One Minor unwritten native-test event assertion remains explicitly deferred to final review. No live app/audio, hardware, provider/model, qualification, repetition, soak or full suite. Tasks 9-10 and final review remain; AC16 is unchecked and TASK-23175 stays In Progress.

Task9 acceptance clarification before implementation of its terminal-order cases: both terminal-event orders means post-boundary speech admission before versus after AttemptPlaybackTerminal delivery, using actual native boundary/default pre-roll and held original parent promotion. Existing native terminal_first and known-boundary reducer tests establish this meaning; existing final-gate reducer coverage retains generation/playback ordering. No new native-hold matrix, production boundary or ADR change. Task9 remains in progress and its first measurements are preliminary, not final evidence.

Task9 confirmed a new process-path follow-up bug, separate from the historical USB overflow: independently retired draft slots gave the same new turn different credit lanes on parent and child. Real two-pipe RED fails both asymmetric-retirement orders. Approved narrow repair before runtime edits: required sender draft_slot integer0|1, exact receiver/reservation/credit identity, bounded two-turn slot bindings, slot+turn replay state and full-stream credit coalescing. Preserve actual custody, strict validation and independent cleanup; no new lane, capacity, owner or release override. Task9 scope adds Audio protocol/lifecycle and protocol/IO/effects tests; entry retirement unchanged. Existing ADR098/spec/plan amended, no new ADR. Native acceptance remains incomplete until the continuation and all targeted checks pass.

Task9 finite software checkpoint: independent native producer/audio child continued through720/3000/720ms parent-GIL holds; actual same-turn interruptions, revised promotion, distinct follow-up, both terminal orders and off/draft fault controls verified. Paired pipes exposed and repaired explicit sender draft-slot identity across independent retirement; review-reproduced mailbox writer race fixed under the existing condition with deterministic RED/GREEN. Independent spec matched; Q1 scoped re-review approved. Final root11-target gate357 passed,zero skips,2 existing warnings,84.48s; nine Python lint/format checks clean. Existing ADR098 applies; no ABI/lane/capacity change. I1 remains unclassified: one earlier startup-only transport failure lacked original stage/exit evidence; later success does not establish cause or repair. Accept only the finite software slice, not startup reliability/live USB/release qualification; final whole-change review sees I1. Task10 source/artifact closure and AC16 remain pending; overall task stays In Progress. No live/hardware/provider/model/soak/full-suite work.

Task10 joined v1 software gate: 1414 passed, 3 obsolete production-factory integration fixtures failed before behavior (252.90s, zero skips). Source review identified three missing directly composed voice runtime fingerprint files. Task10 scoped fix1 now includes fingerprint correction and migration of those three fixtures to actual child/model/parent promotion owners, retaining their behavior checks; no compatibility shim, live tests or authority changes. ADR required: no new ADR; existing ADR098 governs process ownership.

Task10 implementation checkpoint: fixed53-source runtime handshake,576 source/203 Python inventory guards, corrected direct provider/voice-trace coverage and three production-fixture migrations. Spec fix1 and fresh quality approved; final root46-target gate1420 passed,zero skips,3 existing warnings,244.22s. Corrected wheel/sdist include all53 byte-identical runtime files and exclude source-only launcher/test child.65 Python lint/64 format pass; untouched executor format exception remains. Exact evidence/limits in process plan; all-unqualified authority/native/unrelated19 diffs unchanged. I1 startup failure remains unclassified, not repaired by later success. Whole-change review remains before AC16 completion; task stays In Progress. No live/hardware/provider/model/soak/full-suite work.

Final whole-change review covered all74 paths and found Important R1 delayed cancelled-provider data invalidated by replacement admission, and R2 recoverable child speech failure wrongly routed to global fatal.1420-pass checkpoint is retained but not completion. One scoped repair wave planned with actual pipe/coordinator and child-policy RED/GREEN, stale/newer-speech and fatal cleanup negatives, both test Minors, then scoped re-review/final joined verification. Existing ADR098 applies; no new authority/lane/capacity or live/full-suite work. I1 remains separately unclassified; AC16 unchecked.

Final repair scoped review addressed R1/R2/M1/M2 with Minor N1 explicitly parked (retired unissued zero-data terminal bookkeeping; no provider/promotion authority). Root exact46 gate then failed3 follow-up cases with1429 passes,zero skips,3 warnings,270.41s. Diagnosis proved their shared async test wrapper violates synchronous VoiceWinningPromotion claim timing and omits terminal_claim; normal pending cancellation cleanup is not itself the cause. User persistence permits an explicit bounded Task10 test-only fixture correction: preserve real sync claim, gate actual owner publication, assert child claim/result/retirement, then narrow review and one fresh existing joined gate. Five frozen repair paths stay unchanged; all53 runtime bytes match wheel/sdist. No completion/AC16 claim, no live/hardware/full-suite work; overall task remains In Progress.

Approved process-isolation software slice complete at c36f89e24a1aa73cc302f30f962a6dea34e1244d: final whole-change R1/R2/M1/M2 repairs and separately reviewed synchronous-claim fixture correction. Fresh exact46 targeted gate1432 passed,zero skips,3 existing warnings,241.18s; all8 source/test hashes matched freeze. All53 runtime sources byte-identical in both built wheel/sdist; source-only launcher/test child excluded.65 Python full-file lint/64 format pass, retaining untouched executor formatting and existing Requests/audioop/pkg_resources warnings. Native synthetic holds preserve progress/fencing and revised+distinct completed follow-up with actual claim/result receipts. Failed prior1429/3 fixture gate retained, not reclassified. Only AC16 newly checked; TASK-23175 remains In Progress. I1 startup cause unrepaired/unclassified; Minor N1 impossible retired zero-data terminal bookkeeping acceptance parked with no authority impact. No live/acoustic/non-host/release qualification claim. Historical authority/native/protected19 unchanged; no app/audio/provider/model/physical/route/soak/full suite. Exact commands, source,20 rulings and recoverable evidence archive are in the process plan/developer guide.

Approved N1 follow-up: retired never-issued local proposals no longer receive late-terminal bookkeeping. Three real-Mailbox rejection regressions failed before the three-line shared-retirement fix; six positive cases preserve admitted late records for all three provider terminal opcodes with zero/one-data boundaries, duplicate rejection and conflicting end boundaries. Final four-file targeted gate: 244 passed, zero skips, one existing Requests dependency warning, 6.55s; both changed Python files pass full-file Ruff lint/format, and git diff --check passes. Independent read-only N1 review accepts with no findings. Existing ADR098 applies (no new ADR, protocol, authority or bounds changes); developer guide and process spec updated while historical deferred evidence remains intact. Only AC17 newly checked; task remains In Progress, I1 unresolved. No app/audio, provider/model, native hold, qualification, hardware, soak or full-suite run. Protected19 and qualification/build identities unchanged. Exact commands, RED/GREEN output, hashes and review are retained at /private/tmp/tldw-voice-n1.qYPBK7/evidence.md.

Approved startup-race repair: register the already-read bootstrap with the existing mailbox/receiver before starting pipe threads. Deterministic real-subprocess lease/early-close cases failed with child EOF before the reorder and pass after it; they preserve control sequence 2, startup credit suppression, early-close prevention of native start and both clean closure receipts. Independent review accepted production ordering and found one P3 test-helper exit-code gap; queued_startup-only propagation plus a wrong-root bootstrap refusal regression closed it (RED exit0 vs2, then GREEN). An initial malformed-transport exit oracle correctly triggered POSIX self-containment -9; its expected2 was a test assumption, not a runtime defect, and the discarded variant/evidence remain recorded. Final six-file targeted gate: 302 passed, zero skips, one existing Requests dependency warning,19.80s. Three changed Python files pass Ruff lint/format; diff checks clean; narrow re-review accepts with no remaining findings. Existing ADR098 applies; no new protocol, custody, timeout, priority or runtime boundary. Developer guide, process spec and incident-backed lesson updated. AC18 checked; overall task remains In Progress. This fixes a reproduced startup race present at the historical I1 boundary but cannot conclusively attribute the old incident without its missing stage/exit evidence. No app/audio/native hold, hardware/qualification, provider/model, soak or full-suite run. Protected19 and qualification/build identities unchanged. Exact commands, RED/GREEN/raw outputs, hashes and review retained at /private/tmp/tldw-voice-startup-race.R4Q8QY/evidence.md.
<!-- SECTION:NOTES:END -->

## Implementation Notes — final native integration corrections, 2026-09-04

- Production attempt/tool/sequencer cancellation now fences native PCM and old
  producer admission synchronously; async TTS resource cleanup stays bounded and
  supervised. Late old-attempt cleanup cannot invalidate replacement audio.
- Session terminal classification requires native capture/DSP coverage even in
  half duplex, before pending admission and the transcript seal. Existing acoustic
  gates still reject playback-period speech.
- Permanent session shutdown starts native deactivation and the two-second close
  deadline at the synchronous facade/core fence. Checked-close observation is
  independent of stalled/failed transcription cleanup, and caller cancellation
  cannot discard its categorical result. Delayed startup cannot reopen capture.
- Public VERSION_TUPLE now matches 0.1.9.0; the existing static version checker
  verifies both public representations. Existing source/lint inventories already
  cover every changed file. No native source/artifact or qualification change.
- Changes cover transport, session effects/cleanup, sequencer, version metadata,
  and focused tests, with small standalone test-fixture migrations for explicit
  terminal receipts and the earlier device-close ordering. Exact RED/GREEN and
  limitations are in the ignored final-fix report. Governing ADR: existing
  `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`; no new
  decision. Scoped re-review and final runtime/live preflight remain pending.
- Final targeted verification: 479 passed, no skips, three existing dependency/
  deprecation warnings. Ruff lint/format, scoped diff and public/native version
  lock checks passed. No hardware, provider call, qualification or full suite ran.

## Implementation Notes — ordinary conversation diagnosis, 2026-09-04

- The ordinary USB conversation has **not passed**. The captured events show a
  rolling session with speech admission but no STT window, followed by a new
  native session with recognition, input overflow, generated replies, and a
  generic TTS failure. The failed app was stopped; no further speech was requested.
- Fixed nearest-sample timestamp rounding in `Audio/rolling_transcript.py`:
  a negative gap smaller than half a 48 kHz sample now matches positive-gap
  rounding without dropping PCM. Larger overlaps still suspend transcription.
  Device-free regressions reproduce the previous one-nanosecond failure.
- Added categorical transcript failure events in the session and exception-class
  diagnostics in the phrase sequencer; tests assert that spoken text and exception
  messages do not enter these events.
- Verification: 136 targeted transcript, phrase sequencer, and pipeline tests
  passed; Ruff lint, formatting, and diff checks passed. No full suite, physical
  qualification, route matrix, or soak was run.
- ADR-098 remains governing. No acoustic safety threshold or admission decision
  changed. Runtime callback stalls/overflow and the underlying TTS exception
  remain unresolved; idle silence is insufficient to prove acoustic isolation.

## Implementation Notes — Parakeet execution isolation, 2026-09-04

- Applied ADR-098's local execution-boundary amendment with one spawned process
  per voice session. Native contexts and in-memory rolling recognition share one
  model. Requests and results are bounded; stream identities are fenced; only
  categorical errors return. Other STT adapters and AEC policy are unchanged.
- Teardown stops the decoder before awaiting transcript cleanup, including a
  hung inference call. The source identity and lint inventories include the new
  runtime and regression tests. Unrelated trace-ledger/CSS changes are untouched.
- Read-only review identified descriptor-guard startup contention and a dead
  prewarm worker incorrectly advertising rolling fallback. Regressions reproduce
  both: the shared guard now supports an optional bounded wait (unchanged default
  for existing callers), and terminal worker failure keeps capture closed.
- Verification: 183 targeted tests passed, including process ownership, native
  and rolling adapters, stale handles, timeout/hung cleanup, production factory
  selection, visible Hands-free activation, and source/launcher guard tests.
  Ruff lint, formatting, and diff checks passed; the limited re-review found both
  reported issues resolved. Existing dependency/deprecation warnings remain.
- Generated speech, with no audio device opened: native inference recognized the
  full 3.55 s input in 3.083 s; the independent 10 ms thread's maximum gap fell
  from 81 to 24 ms. Rolling inference took 697 ms, with a 21 ms maximum gap versus
  64 ms previously. These observations are not a hardware qualification.
- Generated macOS TTS with production playback transport and simulated callbacks
  cancelled in 1 ms, rejected stale output, and completed a sequential revised
  reply during isolated STT. Capture/reference overflows were zero. The prior
  live generic synthesis failure did not reproduce; its exact cause remains
  unproven. The ordinary USB conversation remains outstanding. No full suite,
  physical qualification, hardware matrix, route switch, or soak was run.

## Implementation Notes — idle overflow evidence, 2026-09-04

- The latest app run selected the speculative session but failed while idle,
  before speech admission. Existing events show 686 ms and 3010 ms UI stalls
  around transport faults; both exceed the 640 ms capture queue. The logs lack
  the original overflow evidence, so the cause of those live stalls is still
  unproven. The app was stopped; no further hardware test was requested.
- Continuous fake device timestamps reproduced dropped capture/reference
  overflow being mislabeled as a clock-only discontinuity. Overflow evidence
  now remains latched for the clock generation, independently of later format
  or processing errors. Only a generation reset clears it. Session tests prove
  overflow cannot enter the one-shot idle clock-gap recovery path.
- Review caught error-string overwrites in the first fix. Four added cases
  failed before the independent latch and passed afterward; both ring types,
  subsequent acknowledgements/format errors, and clean restart are covered.
  Narrow re-review found no remaining actionable issues.
- A full isolated Textual app test clicks the visible Hands-free switch, proves
  `ConsoleSpeculativeHandsFreeSession` owns the production transport, and feeds
  continuous simulated callbacks without capture overflow. Native AEC, STT,
  and audio hardware are replaced; this is UI/scheduling coverage, not proof
  that the real conversation or interruption works. Source/lint inventories
  include the test.
- Verification: 132 targeted transport, session, mounted UI, and source/lint
  inventory tests passed. Ruff lint/format and diff checks passed; existing
  dependency/deprecation warnings remain. Changes are uncommitted on
  `cf3cd80b38cfed021aa477337b775b128a76d6f6` pending continued diagnosis.
- ADR-098 remains governing; no queue, timing, or acoustic admission thresholds
  changed. The real scheduling-stall cause and ordinary conversation remain
  unresolved. No qualification harness, route change, hardware repetition,
  soak, or full test suite was run. Unrelated user changes remain untouched.

## Diagnostic follow-up — shared UI/audio scheduling, 2026-09-04

- Traced `ConsoleSpeculativeHandsFreeSession.enter()` and `_pump_audio()`:
  capture consumption/AEC/VAD is an asyncio task on the UI's event loop, not
  independently scheduled from Textual. The Parakeet subprocess does not change
  this ownership. The active DB-size timer already offloads file inspection;
  its presence in a stall event is not evidence that it caused the stall.
- Injecting one 800 ms blocking pause into the isolated mounted app reproduced
  five capture drops while fake callbacks continued (18 ms maximum observed
  gap). This proves UI-loop starvation can cause audio loss independently of
  device-clock faults, not which operation caused the original live stalls.
- Extended the mounted regression with an event-bounded pause lasting 80 fake
  callbacks: responsive UI has no transport fault; blocked UI reports exactly
  `buffer-overflow`, never clock-only recovery. Both cases passed. Ruff lint,
  formatting, and diff checks passed. No audio device or external provider was
  opened. Existing warnings remain.
- Proposed next direction, **not implemented or approved**: isolate the live
  audio owner from UI scheduling, retaining bounded transport, acoustic admission,
  epoch fencing, and fail-closed teardown. ADR-098 explicitly deferred a sidecar
  until in-process boundaries proved insufficient; this finding justifies
  revisiting that choice, not silently changing it. Discuss the runtime boundary
  with the user before another fix attempt. The ordinary conversation remains
  unpassed and no further operator speech has been requested.

## Implementation Notes — UI scheduling isolation, 2026-09-04

- Implemented the approved ADR-098 scheduling amendment: the production visible
  Hands-free control creates a view-owned core on one lazy app-owned voice loop.
  Capture/AEC/VAD, transcript coordination, provider cancellation and sequential
  playback no longer depend on Textual's event loop. Parakeet remains isolated
  in its existing process; this is not protection from arbitrary GIL stalls.
- UI preparation/promotion retains UI authority. Bridges fence closed sessions,
  discard late preparation capabilities, coalesce previews, and bound diagnostics
  off the pump. Shared TTS synthesis/iteration/cleanup stays on its owner loop,
  using eight 64 KiB blocks and independent completion receipts.
- Review regressions fixed cancelled-phrase serialization, cancellation during
  response cleanup, stranded metadata waiters, and fail-open quarantine after
  owner-loop closure. Startup exit and orphan-before-gateway shutdown are covered.
  The final compatibility regression preserves Python 3.11's executor signature;
  the driver join remains bounded to two seconds after cleanup.
- Verification: 147 targeted transport/core/sequencer/mounted-UI tests passed;
  102 worker, runtime shutdown, source/launcher guard tests passed; the final
  26 worker/TTS/supervisor/interruption tests passed after review fixes, with
  asyncio debug enabled. Existing dependency/deprecation warnings remain.
  The mounted app loses zero capture frames through 80 and 300 blocked-UI fake
  callbacks. A real coordinator/effects test cancels a fake provider and queued
  audio within 150 ms before UI resumes, preserving same-turn revision and
  post-playback new-turn behavior. These are software gates, not live evidence.
- Read-only readiness: Logi USB Headset input/output at 48 kHz; Python 3.12.11
  arm64; sounddevice 0.5.5, webrtcvad 2.0.10, tldw-voice-aec 0.1.8.0,
  parakeet-mlx 0.5.2. Resolved silence is 700 ms and AEC is enabled. The sandbox
  exposes no default audio device; unsandboxed enumeration verified the route
  without opening a stream. No credentials were printed.
- The ordinary USB conversation remains outstanding and the task stays In
  Progress. No release qualification or safety threshold changed. No physical
  harness, route switch, repeat, matrix, soak, or full suite was run. Unrelated
  trace-ledger/CSS work remains untouched.

## Live readiness failure — 5c22aef5b943, 2026-09-04

- The user enabled the visible control once. The exact-commit development app
  reached native Parakeet STT and speculative `session_ready` at 17:09:38, then
  reported `status-input_overflow` at 17:09:40 and degraded to half duplex.
  No operator speech or conversation test was requested after the failure.
- Existing macOS CoreAudio messages at 17:09:40.151 and 17:09:40.225 report
  `ClientHALIODurationExceededBudget` for the Logi USB output/input; another
  output overload appears at 17:09:41.518. These identify native client I/O
  deadline overruns, not their internal Python/native call or lock cause.
- The first native status is not latched like application ring overflow:
  subsequent frames propagate `prior_failure` as generic discontinuity, so
  the session requests clock-only recovery at 17:09:40. Recovery also fails.
  Code inspection establishes this classification gap; no fix was attempted.
- One two-second read-only stack sample of the already failing process is at
  `/private/tmp/tldw-voice-5c22aef5b9-current.sample.txt`. It was taken after the
  fault and does not establish which callback operation exceeded its deadline.
  The app log is under `speculative_voice_dev_5c22aef5b943/tldw_cli_app.log` in
  the normal user data directory. Do not equate prior software tests with this
  native callback boundary being healthy.
- Gracefully stopped the exact app PID 13872; `app_stopping` was logged at
  17:11:38 and the development process is gone. No restart, route switch,
  physical harness, repetition, or soak was performed. Investigate native
  callback scheduling/ownership before proposing another architectural fix or
  requesting another conversation. The task remains In Progress.

## Hardware-free native callback diagnosis — 2026-09-04

- A native C driver exercised the installed CFFI callback boundary with 250
  synthetic silent frames, real transport and native AEC, without audio hardware.
  The standalone worker's maximum callback was 268 us; the mounted Console's
  baseline was 2248 us, with zero deadline misses and zero transport faults.
- One controlled full GC during mounted capture delayed Python callback entry
  by 183640 us, while the callback body stayed below 594 us. Clock drift and
  capture timing failure followed, without capture ring overflow. Recovery was
  disabled in this diagnostic core only to keep one native stream/timing array.
  This proves same-process GC can defeat thread isolation, not that GC caused
  the earlier real USB fault. That run has no fault-time entry/GC trace.
- Native AEC already releases the GIL around processing. Before another runtime
  change, review a GIL-independent, bounded native device callback boundary;
  another Python thread cannot remove this demonstrated dependency. No runtime
  fix or threshold relaxation was made. The status/recovery classification gap
  also remains open. The ordinary conversation is still outstanding.
- Detailed evidence and throwaway C/Python probes are retained under
  `/private/tmp/tldw-voice-callback-boundary-findings.md`. The temporary mounted
  test was moved out of the checkout; unrelated changes were preserved. No
  microphone, physical harness, route switch, soak, model, or provider was used.

## Native callback design review corrections — 2026-09-04

- Incorporated the user's requested five review corrections into
  `Docs/superpowers/specs/2026-09-04-speculative-voice-native-callback-design.md`:
  distinct submission IDs/AEC ordinals, capture-time playback context, ordered
  boundary discovery before post-playback speech admission, checked/bounded
  teardown with retained native quarantine, and separate harmless priming versus
  fatal status latches. Added explicit adversarial test requirements for each
  contract; no test implementation was added in this documentation revision.
- Boundary discovery no longer implies playback completion. Delivery waiting and
  teardown observation have explicit deadlines; cancellation cannot free native
  owners or turn buffered playback audio into unprotected idle speech.
- This is a documentation revision, not implementation or passing test evidence.
  The corrected written boundary remains subject to final review before planning.
  ADR-098 remains governing; its amendment follows final boundary approval. No
  production code, release qualification, hardware or unrelated files changed.

## Implementation Notes — native callback software integration, 2026-09-04

- Implemented the approved ADR-098 native callback amendment through native
  ownership, transport receipts, causal playback/admission and production UI
  integration. App/companion metadata is locked at 0.1.9.0; historical evidence
  and packaged unqualified rollout are unchanged.
- The mounted visible switch selects the production speculative session and
  passes the actual native callback/userdata to sounddevice. One controlled
  80 ms GIL hold at full-GC entry preserves eight native capture/reference pairs
  through real AEC; fencing before drain preserves historical playback context.
  Separate mounted synthetic speech/provider events prove same-turn interruption
  and post-boundary next-turn routing with default preroll in both terminal
  orders. This is software evidence, not acoustic/provider/live qualification.
- Categorical native-unavailable and shutdown-unconfirmed notifications expose
  actionable content-free failures on the UI loop, with late results fenced by
  generation. Actual speech onset routes turns while retained context remains
  available to STT; admission stays within the existing classification barrier.
- Relevant sources, native ABI tests, mounted tests and approved spec/plan are
  included in source/lint inventories. Task-scoped and final controller reviews,
  final-HEAD preflight and the ordinary USB conversation remain outstanding.
  TASK-23175 remains In Progress. No hardware, provider, route switch, physical
  harness, repetition, matrix, soak or full suite was run for this integration.

## Implementation Notes — post-smoke software repairs, 2026-09-04

- Implemented the approved post-smoke repair spec and plan under ADR-098.
  Complete paired DSP references now include actual callback-written silence
  without manufacturing submission receipts or admitting raw playback speech.
  Parakeet native, prewarm and rolling work share context-lifetime ownership;
  exact-full-chunk quiet expiry preserves cached transcript coverage, and failed
  cleanup retains the service fence until confirmed process death.
- Added bounded, content-free native fault snapshots and trusted STT categories,
  including dropped-record faults without fabricated timing. Provider exception
  messages cannot spoof app-owned reasons. First-stage telemetry separates provider
  text, eligible phrase, successful synthesis-stream cleanup and the exact first
  native output receipt; output abort revokes stale receipt timing authority.
- Implementation and review corrections: DSP `ac6e9089e7`; STT ownership
  `30f77d0b81` and `5dcdc29214`; diagnostics `faf0c59f9d`, `adcfc4df1e` and
  `cc4c468c50`; timings `7e7a62a1bb` and `9e3aa155ac`. Each repair passed
  independent specification and code-quality review; final whole-repair review
  found no actionable issues. Review caught and corrected trusted-category loss,
  late receipt logging after output abort, and insufficient multi-submission coverage.
- The first joined gate exposed a stale positive qualification test fixture:
  696 passed, one failed because synthetic matching identities still used
  `0.1.8.0` after the earlier app version bump. Test-only correction `0bc6868f49`
  uses the current version and explicitly retains old-version rejection; it passed
  separate specification and quality review. Production authority checks and the
  packaged unqualified manifest were not changed. Recorded the incident in
  `backlog/docs/lessons-testing-evidence.md`.
- Final software verification at committed code/test HEAD
  `0bc6868f49f7a6fe30e04d96b342d76b25c5cfcb`: the exact 22-file joined gate in
  the approved plan passed **701 tests, zero skips, in 108.03 seconds**. It ran
  on the supplied feature worktree with owned documentation/inventory updates
  and preserved unrelated trace-ledger/CSS edits, not a clean release-evidence
  checkout. Ruff and formatting passed for all 19 changed Python files; version
  lock and diff checks passed. Existing warnings: requests dependency versions,
  pydub `audioop` deprecation, and webrtcvad `pkg_resources` deprecation.
- The joined gate executed the mounted visible Hands-free selection, real
  native/AEC with fake PortAudio, blocked-UI continuity, same-turn revision,
  post-completion next-turn, default-preroll and checked-shutdown regressions.
  App/companion remain `0.1.9.0` / native ABI 1; no native rebuild, install,
  dependency change, AEC threshold change or silence-default change was made.
  Updated source identity inventory includes the approved repair spec and plan;
  historical evidence hashes and packaged rollout authority remain unchanged.
- Live evidence remains limited to the earlier ordinary USB conversation at
  `7c4cc693f36d`: the user perceived a relevant reply and interruption/revision,
  but revised playback stopped on host input overflow. Complete playback and a
  subsequent new turn were not demonstrated live; desktop-switch causality and
  the host overflow cause remain unknown. No further audio, app launch, provider,
  model load/download, qualification harness, route change, repetition, matrix,
  soak or full suite was run. The old exact-HEAD launcher was not rebound/opened.
  Keep TASK-23175 **In Progress** and the branch/worktree intact; these results
  complete the software repair slice, not live or release qualification.

## Process-isolation design checkpoint — 2026-09-05

The ordinary conversation at `fd89a7b0c341` admitted the follow-up after playback
completed, then suffered native ring loss during a roughly 674 ms consumer gap;
the later attempt generated against permanently rejected render admission. A
software-only mounted positive control completed two distinct turns and 1,000
callbacks without loss. A deliberate 720 ms app-GIL hold reproduced the overflow
and enabled-but-unusable state. This establishes a dependency mechanism, not the
historical source of the live scheduling pause.

The user approved process isolation. The written design is
`Docs/superpowers/specs/2026-09-05-speculative-voice-process-isolation-design.md`.
The first review identified a whole-WAV/IPC-credit deadlock; the revised proposal
retains decoding in the parent and sends bounded normalized PCM. Independent
round-2 review approved it with no remaining serious planning gaps. Written user
approval still precedes ADR-098 amendment and implementation planning. Document
links, placeholder checks and the scoped diff check passed; no runtime tests ran. No
runtime code, audio route, provider selection, safety threshold or qualification
authority changed at this checkpoint. AC16 is uncompleted and the task remains
In Progress; further verification stays software-only.

## Process-isolation implementation plan — 2026-09-05

The user approved the reviewed written spec. ADR-098 now records the accepted
audio-process/authority/cleanup amendment. The implementation plan is
`Docs/superpowers/plans/2026-09-05-speculative-voice-process-isolation.md` and both
documents are in the source identity inventory. Independent whole-plan review
found that the actual provider gateway could bypass downstream backpressure;
Task 7 now bounds voice-specific synchronous producer admission before scheduling
delivery, with a real-gateway regression. Round 2 approved the corrected plan.

Source-inventory and lint-scope guards passed **21 tests** with one existing
requests dependency warning. Document links, planned file targets and the owned
diff check passed. No runtime code changed or runtime tests ran at this planning
checkpoint. Implementation continues task-by-task in the supplied worktree with
subagent specification/quality reviews and software-only targeted verification.
AC16 remains unchecked and the overall task remains In Progress. No new live
conversation, app launch, provider request, model inference or qualification is
authorized by this checkpoint.

## Process-isolation Task 1 — lightweight policy prerequisite

Moved the existing reducer to the Audio package with validated decision/request/
context handles. The parent compatibility adapter retains original authority
objects, projects tool/promotion results and keeps one bounded current mapping.
Shared enums/constants retain public aliases; UI and Widgets now install the
existing Textual shim at their own package boundaries. Unknown cleanup receipts
fail closed. Corrected one baseline-reproduced property fake that omitted the
required classification drain without weakening its winning-snapshot assertion.

Independent specification and quality reviews approved the 13-file Python change.
Fresh root verification passed **278 targeted tests, zero skips, in 17.57 seconds**;
Ruff and formatting passed. The two warnings were existing requests dependency
compatibility and webrtcvad/pkg_resources deprecation. The process/core/IPC work
is still pending; this is not evidence of GIL isolation or live voice success.
The approved ADR-098 process-isolation plan continues; AC16 and overall task status
remain incomplete/In Progress. No hardware, app launch, provider/model execution,
qualification, installation or full suite was performed.

## Process-isolation Task 2 — extracted audio core

Moved existing capture/AEC/VAD/seal, transcription and phrase algorithms into
dependency-light Audio modules. Parent compatibility facades retain original
provider authority and TTS decoding. Preparation and startup have separate,
retained owners; concurrent callers cannot duplicate capture or pumps, and close
fences prevent late startup/readiness. Native-STT pending PCM has a typed terminal
capacity bound. Permanent transport faults fence work immediately, retain the
latest current/follow-up draft once and report failure independently of cleanup.

Review reproduced and corrected production STT cleanup/diagnostic wiring,
cancellation during iterator and natural response closure, overlapping startup,
and an unnecessary unbounded sequence index. Actual cleanup receipts now remain
owned until resource closure; the original bounded transcript lookup is retained.
The factory/fake fidelity incident is recorded in `lessons-testing-evidence.md`.

Independent specification and code-quality reviews approved the corrected scope.
Fresh root verification: **473 targeted tests, zero skips, 27.25 seconds**; Ruff
check/format passed all 12 changed Python files, with only the two existing
requests/webrtcvad dependency warnings. Governing ADR remains ADR-098. This is
the reusable-core prerequisite, not proof of subprocess isolation or live voice
completion. Tasks 3–10 continue under the approved plan; AC16 stays unchecked and
the overall task stays In Progress. No hardware, live app, provider/model execution,
qualification, route changes, soaks or full test suite was used.

## Process-isolation Task 3 — bounded private messaging

Added the closed, strictly validated voice wire schema and bounded pipe transport.
Credits cover sender/pipe/receiver custody, release only after actual consumption
or matching fenced discard, and cannot acknowledge unwritten framing. Separate
bounded priority lanes preserve cancellation, closure and cleanup receipts under
full data pressure. Reader endpoint ownership lasts until its thread exits.

Independent reviews reproduced and corrected draft/replacement identity gaps,
premature credit receipts, queued-data cancellation and descriptor-reuse races.
Both final specification and quality reviews approved the corrected scope with
134 passing protocol/pipe tests each. Root verification passed **166 targeted
tests, zero skips, in 3.66 seconds**; Ruff check/format passed all four new Python
files. The sole warning is existing requests dependency compatibility.

For subsequent integration, ordinary cancellation cancels the sole producer and
drains known fenced data through matching discard/credit; `CreditWindow.close()`
is terminal admission closure, not the normal barge-in path. ADR-098 remains the
governing decision. Tasks 4–10 continue; AC16 remains unchecked and the overall
task remains In Progress. These are software messaging results, not subprocess/
GIL or live qualification evidence. No hardware, app launch, provider/model
execution, installation, route change, soak or full suite was performed.

## Process-isolation Task 4 — supervised process ownership

Added private child bootstrap, source/ABI checks, separate readiness/start permits,
bounded leases and retained shutdown ownership. POSIX process-group containment
and the existing Windows Job API keep descendant absence distinct from audio PID
exit. The app-wide device lease requires explicit native and typed resource
cleanup receipts, drained pipes, confirmed tree exit and no forced or uncertain
closure; otherwise it stays quarantined.

Review-driven real-pipe regressions corrected unread final receipts, hidden
child-forced model termination and parser failures lost in either direction.
Native/resource facts remain separate from sticky protocol uncertainty. Empty
frame-boundary EOF is normal; incomplete framing is a fixed transport failure.
Child fault reporting is bounded to one attempt, with abnormal exit evidence if
the writer cannot deliver it. The incident is recorded in testing-evidence lessons.

Final specification and quality reviews approved the ten-file Python scope. Root
verification passed **247 targeted tests, zero skips, in 26.88 seconds**; Ruff
check/format and scoped diff checks passed. One existing requests warning remains.
Actual POSIX fake-child cleanup was exercised; real Windows is unverified. The
release entry remains intentionally fail-closed until Task 8 composes the real
pipeline. ADR-098 governs this prerequisite, not new qualification authority.
Tasks 5–10 continue; AC16 remains unchecked and the task stays In Progress. No
hardware, app, real provider/model, installation, native production rebuild, soak
or full suite was used.

## Current-dev integration Task 3 — app authority checkpoint

Integrated the committed speculative voice app authority onto the reviewed
accepted-turn runtime and current dev trace implementation (ADR-094/098). Winning
claim/pair persistence remains synchronous and pair-first; provisional trace
capture now uses the frozen scoped/custom privacy policy, bounded pre-seal masks,
atomic canonical masks/omissions, and exact retry reconciliation. Schema 68→69
adds only exceptional voice provenance and managed exact-call import authority,
preserving current Canvas, semantic mutation, GC and privacy boundaries.

Adapted current semantic memory/durability reconstruction and original provider
work custody; RAW Capture On/Off stays speculative with PII enabled, while
non-RAW memory keeps existing accepted preparation. The only additional config
prerequisite is the source lock-free generation baseline accessor, retaining the
existing final publication fence. Voice-only persistence tests retain exact
rollback/retry/census assertions, including current-dev branch-memory locators.

Targeted candidate evidence includes 161 authority cases and a final overlapping
38-case integration gate, each passing with two existing dependency warnings.
Full command/inventory evidence is in the existing integration plan's Task3
checkpoint report; these are software checks, not release qualification. The
ordinary custom-provider shape expectation fails identically on clean BASE2244
and remains unchanged. Task4 still owns real preview/settings/UI/entry composition
and its explicitly deferred fake-IPC/ephemerality catch-up tests; no native or
live qualification was repeated. Task3 specification and quality review and the
remaining integration tasks must finish before overall completion; task status
and unchecked release criteria are unchanged.

## Current-dev integration Task 5 — targeted software checkpoint

Normally merged exact dev `3cccd9326c556a245fc87ab5013c192242e389cf` in
`75b72a8a17`, preserving latest appearance/cache/batch APIs, diagnostics,
validation cleanup, private-root locking (ADR-127) and caret/repaint behavior.
The existing ADR-094/098 architecture and schema69 extension are unchanged.
Narrow test adaptations await actual accepted-runtime completion and exercise
explicit exact-draft/session recovery through the production UI action. Bounded
same-file teardown closes observed fixture-only SQLite connections; no production
DB lifecycle, limits or privacy policy was changed.

Joined software gate: **326 passed, three existing dependency warnings**, 166.38s.
Includes the 28 previously deferred parameters, real producer→sealed gateway→
SQLite privacy/reconstruction checks, visible source launcher/actual factory with
fake child, and latest-dev preservation tests. Six-check preflight passes after
affected diagnostic/index inventory updates; boot CSS remains803996/804000 bytes.
Exact commands, selectors, intermediate failures, narrowly diagnosed upstream
fixture retention, existing lint/format debt and final application wheel/sdist
hashes are in the integration plan's Task5 evidence report. Counts from overlapping
historical cohorts are not summed or relabeled as current passes.

All packaged platforms remain unqualified; original authority JSON bytes and
version0.2.0/exact speech pin remain intact. No full-suite, live/native/physical/
soak or installation run occurred. Independent final reviews and dev-targeted PR
publication remain pending with the root task; AC19 publication is not complete.
Task status and unchecked release/live criteria remain unchanged.

Publication refresh then normally merged the Library-only dev update
`37bf45fb62` in `2e5da4ebd2`. Four synthetic danger-style/spacing paint cases
plus the unchanged boot CSS budget passed (5 cases, four warnings); all six
preflight checks pass. No voice/provider/schema/config boundary changed and
the earlier 326-case gate was not repeated or relabeled. Final refreshed archive
paths/hashes and exact commands are retained in the Task5 report; publication
and broader qualification remain pending.

## Current-dev integration publication

Opened [PR #2504](https://github.com/rmusser01/tldw_chatbook/pull/2504) from
`codex/speculative-duplex-voice-dev` to current dev `37bf45fb62`. All five
specification/quality review pairs and the final independent whole-branch review
approved implementation `f6ed08120e` without actionable findings. Root's final
committed boundary gate passed 18 cases with three existing dependency warnings;
six derived-artifact checks and independent final wheel/sdist inspection passed.
The earlier joined 326-case result retains its original checkpoint attribution.

AC19 is complete; this task remains In Progress and broader release/live criteria
remain unchecked. Original source HEAD, its 19 unrelated dirty paths, and both
packaged authority resources are unchanged. Normal automatic PR CI has started,
not yet asserted passing. No PR merge, release publication, manual workflow,
additional live audio, physical qualification, local native execution or soak
was performed. The integration plan records the exact review/evidence boundaries.

## PR #2504 follow-up — schema composition, 2026-09-08

Backed up published `47de10b893` and consolidated its exact tree before a real
rebase onto fetched dev `7e81ed55db`. Kept dev's schema-69 source-pin SQL and
migration method byte-identical; voice provenance now follows from 69 to 70.
Canvas assertions retain dev's current-version behavior. The voice migration
test and index census follow the new slot; the source-pin fixture seeds its
historical call shape directly and compares schema-69 parity at version 69.
ADR-098 records the composition; no new architecture decision was required.

A genuine schema-69 predecessor regression first failed at 69 versus 70. After
composition and historical-fixture repair, the focused migration/Canvas/voice
trace repository gate passed 59 cases in 24.70s with two existing dependency
warnings. Index-plan census passed; the official diagnostic generator changed
only combined aggregate counts. Scoped lint passes; inherited whole-DB formatter
debt is retained, with the new migration method checked separately. Exact
commands, tree/ref/authority evidence and limitations are in the ignored Task 1
report under `.superpowers/sdd/2026-09-08-voice-pr-2504-followup/`.

AC19 remains checked and the task remains In Progress. This local checkpoint
does not qualify audio or claim full-suite/PR CI success. No push, merge, native
build/load, app/audio, provider/model, hardware, soak, install or config change
was performed; original dirty worktrees and packaged hard-off authority remain
unchanged.

## Renumbering provenance

This task was initially created as `TASK-23113`. It was renumbered before its first
commit because that ID was already owned by the older Wizard-suite load-flake task on
multiple remote refs. `TASK-23175` was clear across all remote refs and local worktrees
at filing time.
