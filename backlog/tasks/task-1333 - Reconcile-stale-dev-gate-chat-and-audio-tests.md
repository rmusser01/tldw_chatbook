---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-29 08:11'
updated_date: '2026-07-29 15:04'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-29-dev-gate-test-contract-repair-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the mandatory dev test gate by aligning stale or nondeterministic tests with the current retired-Chat and audio-recording contracts, preserving the current dev shell-test repair, and safely refreshing the reviewed diagnostic inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Worker-event regressions retain the live non-streaming delegation and failure coverage without importing retired message classes, recreating worker-owned citation/streaming behavior, or duplicating existing native Console and streaming-rejection contracts.
- [ ] #2 The current dev chat-shell regression retains live session/persona label coverage without importing or replacing the retired `TabState` model.
- [ ] #3 The audio stream-error regression invokes one synchronous recording loop without VAD or thread races and proves the exact pre-error callback sequence, stream closure, and stopped state.
- [ ] #4 The PyAudio flow regression invokes one synchronous recording loop without VAD or thread races and proves exactly three callbacks, stream closure, and stopped state.
- [ ] #5 The SoundDevice flow regression disables VAD for its synthetic callback, waits boundedly for the mocked stream callback, and cleans up its recording thread even on failure before proving audio was queued.
- [ ] #6 The Llama.cpp, DeepSeek, and local-LLM request tests patch the live runtime-config snapshot seam without restoring or emulating deleted mutable module-level settings.
- [ ] #7 Every real-seam Notes fixture creates its temporary trusted base directory as owner-only before constructing `NotesInteropService` and closes per-user Notes DB connections during teardown, without weakening production path verification.
- [ ] #8 Every changed diagnostic owner is reviewed against ADR-029, no unsafe payload logging is admitted, persistent sink topology remains unchanged, and the checked inventory matches production.
- [ ] #9 The large-batch RAG indexing regression retains its 1,000-item persistence and retrieval coverage without host-dependent wall-clock assertions.
- [ ] #10 The retargeted Evals profile regression creates its temporary trusted profile directory as owner-only before selecting the config path and closes its test-owned database connection, without weakening production path verification.
- [ ] #11 Local skill names cannot shadow the registered `search_run_log`, `run_log_stats`, or `run_log_slice` runtime tools, and the existing shadow-set drift guard passes.
- [ ] #12 The `quick_ingest()` fallback-path regression expects the canonical profile-aware `tldw_chatbook_media_v2.db` filename while retaining configured-path and traversal-rejection coverage.
- [ ] #13 The reusable RAG citation benchmark host context creates an owner-only isolated config profile before selecting `TLDW_CONFIG_PATH`, runs without reading or mutating host config/data/secrets, and retains its existing output-privacy assertions.
- [ ] #14 The production Console Stop regression drives a rendered visible button before clicking it, and proves that the user action cancels the provider stream and preserves the stopped partial response without relying on app teardown.
- [ ] #15 The production Media lifecycle regressions wait boundedly for replaced windows to become closed and detached before exercising stale-owner and cross-window write ordering, without weakening the fresh-screen or durable last-edit-wins contracts.
- [ ] #16 The production provider-selection ownership regression waits for the recomposed Settings controls, deterministically stages and saves the selected provider/model through their live handlers, and retains per-session preservation plus subsequent handoff coverage.
- [ ] #17 RAG UI integration coverage no longer expects a recognized canonical media candidate to enter prompt context when current authority cannot be established; the dedicated citation-capture suite retains both fail-closed authority coverage and the unsupported-source legacy fallback contract, and the public capture docstring describes that boundary accurately.
- [ ] #18 The lazy RAG-admin app fixture publishes its fake local runtime state through the live runtime-policy projection owner instead of assigning deleted writable compatibility fields, while retaining lazy-construction and service-wiring coverage.
- [ ] #19 The affected modules and repository-wide suite collect and run without these baseline failures.
- [ ] #20 Legacy bulk-backup cancellation and worker-failure regressions inspect the live profile-aware backup root and accept either an absent root or an existing empty root while continuing to prove that no database, temporary, manifest, success-notification, or in-progress artifact survives.
- [ ] #21 Backup orchestration regressions derive their expected storage from the live profile-aware user-data owner and retain distinct-directory, manifest-content, partial-failure, publication-order, and no-artifact coverage without hard-coding the retired unscoped backup path.
- [ ] #22 Manifest staging regressions observe the secure exclusive-create helper in the worker thread, exercise cleanup failures only after a stage exists, and assert injected private values remain absent from public diagnostics without rejecting unrelated isolated config-path logs.
- [ ] #23 Profile-backup integration fixtures inject temporary ChaChaNotes and Media paths through the live canonical resolver map while retaining the direct Prompts resolver patch, so success and partial-failure manifests prove the intended legacy and optional TTS Profile entry sets without relying on ignored `config_data` fields.
- [ ] #24 The TTS preference read-purity regression guards the live public config mutation helpers without monkeypatching the deleted `atomic_write_text` implementation symbol, while retaining input immutability and zero-persistence-call coverage.
- [ ] #25 Parakeet MLX returns no synthetic segment when the model emits empty text and no sentences, for both zero-duration and very-short audio, while retaining sentence timestamps and the single untimed fallback for non-empty text without sentence metadata.
- [ ] #26 A valid zero-frame audio file returns the standard empty Parakeet MLX result before model loading or inference, avoiding MLX tensor-length underflow while preserving invalid-file errors and normal non-empty decoding.
- [ ] #27 The Parakeet MLX no-SoundFile regression patches both the availability flag and runtime module seam, fails locally for its nonexistent fixture path, and never attempts a real model download.
- [ ] #28 Command-palette provider regressions read the current provider from the mounted Console session and stage provider switches through the pending-handoff owner, without assigning or asserting the retired app-root provider reactive.
- [ ] #29 The Library ingest option-persistence integration regression observes the live batched config writer and proves the submitted PDF and generic option groups are persisted together, without patching the retired per-key save path.
- [ ] #30 Structured config-mutation regressions observe the live private atomic writer for one-replacement, pre-replacement failure, overlap rejection, shared-lock, batch-wrapper, and delete-wrapper coverage; the packaging source-seam regression requires that same profile-aware private writer and application-owned directory posture, without restoring the deleted generic writer symbol or hard-coded default path.
- [ ] #31 Console unknown-command composer regressions expect the complete current registered-command hint, including image generation and rewind, while retaining first-Enter interception, second-Enter literal send, edit-disarm, and message-count coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-29-dev-gate-test-contract-repair.md

ADR required: no
ADR path: backlog/decisions/029-local-private-data-boundary.md
Reason: Reconciles tests with accepted production contracts and applies ADR-029's existing metadata-only inventory review requirement without making a new architectural decision.

1. Remove the retired StreamDone import, duplicate streaming assertion, and fully obsolete worker-local citation capture file while preserving unique retained-adapter and non-streaming failure coverage.
2. Preserve the current dev chat-shell repair rather than carrying a superseded branch edit.
3. Make both PyAudio loop tests synchronous, VAD-independent, and exact; keep the SoundDevice fixture VAD-independent with explicit cleanup.
4. Patch all stale provider request tests through the live runtime-config snapshot seam instead of deleted module globals.
5. Create temporary trusted Notes roots in each stale real-seam fixture.
6. Remove host-dependent timing assertions from the large-batch indexing test while retaining its functional coverage.
7. Create the retargeted Evals profile fixture directory before selecting its config path.
8. Synchronize the fixed Library skill collision set with registered run-log runtime names.
9. Align the Local Ingestion fallback assertion with the canonical media database filename.
10. Create the isolated RAG benchmark's trusted config profile directory before application imports.
11. Let the Textual pilot render the visible Console Stop state before issuing the pointer click, retaining the real user-action cancellation assertion.
12. Wait for outgoing Media windows to finish Textual's asynchronous close/detach lifecycle before asserting replacement behavior.
13. Await the recomposed provider controls and boundedly observe staging and persistence around their live Settings handlers.
14. Remove the obsolete DB-free RAG UI fixture and assertion that bypass current prompt authority, retain the dedicated live citation-capture contracts, and align the public capture docstring with that accepted boundary.
15. Publish the lazy RAG-admin fixture's fake runtime state through the current runtime-policy owner.
16. Align all backup path assertions with the live profile-aware user-data owner, including the legacy no-artifact cases, without requiring an empty parent directory to survive cleanup.
17. Retarget manifest staging and cleanup regressions to the current secure create seam and post-stage failure boundary.
18. Point the profile-backup fixtures' temporary legacy database paths through the current canonical resolver owner.
19. Remove the deleted config write implementation from the TTS preference reader's live persistence guard.
20. Align Parakeet MLX empty-result segment normalization with the other transcription providers.
21. Short-circuit valid zero-frame Parakeet MLX input before model loading and inference.
22. Isolate the Parakeet MLX no-SoundFile regression from installed optional dependencies and network downloads.
23. Retarget command-palette provider regressions to the mounted Console session and pending-handoff ownership seams.
24. Retarget the Library ingest option-persistence regression to the current batched config writer.
25. Retarget structured config-mutation regressions to the private atomic writer that now owns config replacement.
26. Align the Console unknown-command expected hint with the complete live command registry.
27. Review all changed production diagnostics and sink topology against ADR-029 before regenerating the checked inventory.
28. Run affected, static, inventory, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->
