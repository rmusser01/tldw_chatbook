---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: Done
assignee:
  - '@codex'
created_date: '2026-07-29 08:11'
updated_date: '2026-07-31 18:38'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/033-application-session-state-ownership.md
  - backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md
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
- [x] #1 Worker-event regressions retain the live non-streaming delegation and failure coverage without importing retired message classes, recreating worker-owned citation/streaming behavior, or duplicating existing native Console and streaming-rejection contracts.
- [x] #2 The current dev chat-shell regression retains live session/persona label coverage without importing or replacing the retired `TabState` model.
- [x] #3 The audio stream-error regression invokes one synchronous recording loop without VAD or thread races and proves the exact pre-error callback sequence, stream closure, and stopped state.
- [x] #4 The PyAudio flow regression invokes one synchronous recording loop without VAD or thread races and proves exactly three callbacks, stream closure, and stopped state.
- [x] #5 The SoundDevice flow regression disables VAD for its synthetic callback, waits boundedly for the mocked stream callback, and cleans up its recording thread even on failure before proving audio was queued.
- [x] #6 The Llama.cpp, DeepSeek, and local-LLM request tests patch the live runtime-config snapshot seam without restoring or emulating deleted mutable module-level settings.
- [x] #7 Every real-seam Notes fixture creates its temporary trusted base directory as owner-only before constructing `NotesInteropService` and closes per-user Notes DB connections during teardown, without weakening production path verification.
- [x] #8 Every changed diagnostic owner is reviewed against ADR-029, no unsafe payload logging is admitted, persistent sink topology remains unchanged, and the checked inventory matches production.
- [x] #9 The large-batch RAG indexing regression retains its 1,000-item persistence and retrieval coverage without host-dependent wall-clock assertions.
- [x] #10 The retargeted Evals profile regression creates its temporary trusted profile directory as owner-only before selecting the config path and closes its test-owned database connection, without weakening production path verification.
- [x] #11 Local skill names cannot shadow the registered `search_run_log`, `run_log_stats`, or `run_log_slice` runtime tools, and the existing shadow-set drift guard passes.
- [x] #12 The `quick_ingest()` fallback-path regression expects the canonical profile-aware `tldw_chatbook_media_v2.db` filename while retaining configured-path and traversal-rejection coverage.
- [x] #13 The reusable RAG citation benchmark host context creates an owner-only isolated config profile before selecting `TLDW_CONFIG_PATH`, runs without reading or mutating host config/data/secrets, and retains its existing output-privacy assertions.
- [x] #14 The production Console Stop regression drives a rendered visible button before clicking it, and proves that the user action cancels the provider stream and preserves the stopped partial response without relying on app teardown.
- [x] #15 The production Media lifecycle regressions wait boundedly for replaced windows to become closed and detached before exercising stale-owner and cross-window write ordering, without weakening the fresh-screen or durable last-edit-wins contracts.
- [x] #16 The production provider-selection ownership regression waits for the recomposed Settings controls, deterministically stages and saves the selected provider/model through their live handlers, and retains per-session preservation plus subsequent handoff coverage.
- [x] #17 RAG UI integration coverage no longer expects a recognized canonical media candidate to enter prompt context when current authority cannot be established; the dedicated citation-capture suite retains both fail-closed authority coverage and the unsupported-source legacy fallback contract, and the public capture docstring describes that boundary accurately.
- [x] #18 The lazy RAG-admin app fixture publishes its fake local runtime state through the live runtime-policy projection owner instead of assigning deleted writable compatibility fields, while retaining lazy-construction and service-wiring coverage.
- [x] #19 The affected modules and repository-wide suite collect and run without these baseline failures.
- [x] #20 Legacy bulk-backup cancellation and worker-failure regressions inspect the live profile-aware backup root and accept either an absent root or an existing empty root while continuing to prove that no database, temporary, manifest, success-notification, or in-progress artifact survives.
- [x] #21 Backup orchestration regressions derive their expected storage from the live profile-aware user-data owner and retain distinct-directory, manifest-content, partial-failure, publication-order, and no-artifact coverage without hard-coding the retired unscoped backup path.
- [x] #22 Manifest staging regressions observe the secure exclusive-create helper in the worker thread, exercise cleanup failures only after a stage exists, and assert injected private values remain absent from public diagnostics without rejecting unrelated isolated config-path logs.
- [x] #23 Profile-backup integration fixtures inject temporary ChaChaNotes and Media paths through the live canonical resolver map while retaining the direct Prompts resolver patch, so success and partial-failure manifests prove the intended legacy and optional TTS Profile entry sets without relying on ignored `config_data` fields.
- [x] #24 The TTS preference read-purity regression guards the live public config mutation helpers without monkeypatching the deleted `atomic_write_text` implementation symbol, while retaining input immutability and zero-persistence-call coverage.
- [x] #25 Parakeet MLX returns no synthetic segment when the model emits empty text and no sentences, for both zero-duration and very-short audio, while retaining sentence timestamps and the single untimed fallback for non-empty text without sentence metadata.
- [x] #26 A valid zero-frame audio file returns the standard empty Parakeet MLX result before model loading or inference, avoiding MLX tensor-length underflow while preserving invalid-file errors and normal non-empty decoding.
- [x] #27 The Parakeet MLX no-SoundFile regression patches both the availability flag and runtime module seam, fails locally for its nonexistent fixture path, and never attempts a real model download.
- [x] #28 Command-palette provider regressions read the current provider from the mounted Console session and stage provider switches through the pending-handoff owner, without assigning or asserting the retired app-root provider reactive.
- [x] #29 The Library ingest option-persistence integration regression observes the live batched config writer and proves the submitted PDF and generic option groups are persisted together, without patching the retired per-key save path.
- [x] #30 Structured config-mutation regressions observe the live private atomic writer for one-replacement, pre-replacement failure, overlap rejection, shared-lock, batch-wrapper, and delete-wrapper coverage; the packaging source-seam regression requires that same profile-aware private writer and application-owned directory posture, without restoring the deleted generic writer symbol or hard-coded default path.
- [x] #31 Console unknown-command composer regressions expect the complete current registered-command hint, including image generation and rewind, while retaining first-Enter interception, second-Enter literal send, edit-disarm, and message-count coverage.
- [x] #32 Console composer send regressions assert the dispatch-time session id, and Console/Library prompt-insert regressions use the live typed pending-handoff channel instead of the retired app-root prompt field, while retaining lifecycle, retry, blocked, append, collapse, navigation, dirty-editor, and empty-prompt coverage.
- [x] #33 Console live-work helper, rendering, action, inspector, staged-context, and Home-isolation regressions stage and inspect the live typed pending-handoff channel instead of assigning or asserting the retired app-root launch field, while retaining normalized payload, consumption, navigation, action routing, and destination-isolation coverage.
- [x] #34 Library prompt editor initialization and post-save recomposition ignore mount-time field-change echoes when the rendered fields still match canonical editor state, while genuine edits remain dirty; successful save/conflict recovery clears dirty state and the clean, empty, and dirty Console-insert paths retain their exact behavior.
- [x] #35 The nested Library UI harness executes app-owned prompt-import workers through the active harness worker manager, retaining all exact import status and database assertions without changing production worker ownership; the real-app durable-owner and survive-unmount coverage remains green.
- [x] #36 The bundled-CSS multi-row Console approval geometry regression lets the card's deferred mount-time hide settle before installing a batch, then retains all nonzero-size, fixed-width, non-overlap, compact-height, and action-position assertions.
- [x] #37 Schedules/Workflows recent-work, Artifacts target, and Home-to-Study regressions seed and settle the current screen-state and typed handoff owners instead of retired app-root fields, while retaining positive recent-work, requested-before-latest, exact payload, consumption, launch, and one-hop routing coverage.
- [x] #38 The MCP approval-cancellation audit regression expects the current metadata-only `approval_cancelled` category rather than retired persisted error text, while retaining the denied decision, blocked outcome, identity, and durable-record coverage required by ADR-029.
- [x] #39 The curated Console remote-default regression expects Anthropic's current configured and cataloged `claude-sonnet-5` default, while retaining exact curated defaults and catalog-membership coverage for every provider in the representative set.
- [x] #40 Console and selector merge-cap regressions pass saved provider models and the catalog scope as the current explicit resolver inputs, while retaining runtime-discovery labels, warning, cap-boundary, uncapped-picker, transient-current-model, and scope-call coverage.
- [x] #41 Console missing-key recovery copy uses the shared human provider display name even when the active session stores a canonical lowercase provider key, while retaining the exact setup target, field, and send-blocking behavior.
- [x] #42 The Console “Choose model” action-routing regression explicitly configures a valid provider with no selected model before opening settings, retaining the live button-click and destination assertion without relying on an internally contradictory blank-provider fixture.
- [x] #43 The live-config Console journey publishes its fake local runtime state through the current runtime-policy projection owner instead of assigning deleted writable compatibility fields, while retaining real boot, Settings persistence, navigation, and no-restart unblocking coverage.
- [x] #44 The live-config Console journey gives its scheduler a private file-backed Subscriptions database so schema initialized during app construction remains visible to the scheduler thread, while retaining the real cross-thread scheduler and full no-restart journey.
- [x] #45 The Console resolution-view regression stages its user provider override through the current Console-owned control state instead of retired app-root provider/model reactives, while retaining fresh persisted-default fallback and Console-over-default precedence coverage.
- [x] #46 Normal-send and picker-driven skill regressions assert the active dispatch-time Console session id alongside the exact raw `$name [args]` draft, while retaining controller-side skill execution and transcript behavior.
- [x] #47 The retired bare-slash skill regression expects the complete current unknown-command hint, including image generation and rewind, while retaining non-execution, draft preservation, and second-Enter arming coverage.
- [x] #48 The Console Workbench parity matrix references current native Console image-attachment coverage instead of a deleted legacy Chat UI test, and every matrix file/test reference resolves.
- [x] #49 The active-conversation workspace marker regression invokes the current argument-free Console workspace sync after restoring the native persisted session, without constructing or importing retired `ChatSessionData`, while retaining the single selected-row assertion.
- [x] #50 Expanded Watchlists source rows paint strictly to the right of their parent watchlist names at both visual-parity viewport sizes after tree labels are left-aligned; the existing relative-column regression remains intact, and source plus generated CSS documentation matches the four-space textual indent.
- [x] #51 The generic disabled-action recovery suite mounts current `STTSScreen` ownership under missing local speech dependencies, verifies the exact phase-five recovery taxonomy in the inspector, and verifies install guidance on the rail summary tooltip.
- [x] #52 A never-run bench renders `Not yet checked` for its target and no target-readiness recovery callout inside `#evals-inspector-bench`, while unrelated screen-level recovery callouts remain permitted.
- [x] #53 A legacy Evals bench with a duplicate target id composes exactly two target rows in the editor and exactly two readiness rows in `#evals-inspector-bench`; all four rows render with nonzero regions and retain distinct index-derived ids despite the shared underlying target id and sibling primary-action status.
- [x] #54 The real Lab route mounts Models without a lifecycle `NoMatches`; local-model delete confirmation is hidden on first paint and its mounted reactive state can show and hide it without mount-order errors; Lab strip navigation still completes.
- [x] #55 Selected and empty Library Collections retain all current copy, selection, and geometry coverage while asserting the established pressable-but-blocked Console handoff state rather than a disabled button.
- [x] #56 When an ingest completes while Notes is selected, the transiently recomposed Library rail eventually remounts its Media row with count 1, Notes remains selected, and ingest-path plus ingest-job widgets remain absent.
- [x] #57 MCP import-file regressions override only the workbench's import-containment-root seam, so temporary picked files remain valid without replacing process-wide home expansion or redirecting the isolated application config path to a directory; picker loading, unreadable-file, outside-home rejection, and size-cap coverage remain intact.
- [x] #58 MCP audit-detail fixtures use the current metadata-only execution-record schema; the rendered inspector retains identity, decision, duration, argument-name/count, result-type/size, and drill-through control coverage while proving raw argument values, result excerpts, and exception text are absent.
- [x] #59 Current Media browsing-shell regressions identify the isolated mounted widget as the active screen-owned Media destination and await the Textual worker manager after background search and item-detail dispatch before inspecting results, resetting the search mock, selecting or reading a result, or asserting query/pagination calls; list population, detail loading, filter propagation, and pagination coverage remain intact without sleeps or production changes.
- [x] #60 The non-obscuring focus contract follows Textual's live `CollapsibleTitle` DOM, pins both global and ID-scoped Library/RAG focus treatment plus the QA-scoped hover owner in source and bundle, and no longer reads or asserts CSS retired with legacy chat tabs, sidebar presets, resize controls, or an unowned `Collapsible.-active` state; all remaining static focus contracts pass without activating dead app-wide selectors.
- [x] #61 Personas character-generation wiring tests dispatch mounted editor controls through Textual's direct `Button.press()` event seam rather than coordinate hit testing, preserving live field/context/preview/failure/regeneration and whole-character behavior while remaining deterministic when the later Library/RAG settings module is collected in the same pytest process.
- [x] #62 The Personas character-import failure regression expects the current fixed recovery message, proves the raw importer exception text is not surfaced, declares its unused dictionary service unavailable during selection setup, and retains the selected-character state after failure.
- [x] #63 The product-maturity Search/RAG-to-Console core-loop regression no longer reads the retired app-root `pending_chat_handoff` field and still proves the real route, staged-source count, live-work title, evidence readiness, RAG state, and suggested composer draft through current visible Console behavior.
- [x] #64 The product-maturity service-unavailable matrix expects Library's established pressable-but-blocked Console handoff while continuing to require disabled handoffs for Watchlists and Skills, with every destination's recovery copy and tooltip retained.
- [x] #65 The completed first-run character-chat UAT task record has valid unique `TASK-672` YAML frontmatter and a matching task heading, while its acceptance criteria, plan, implementation notes, and history remain unchanged.
- [x] #66 Focused Study screen, dashboard, quizzes, flashcards, and product-maturity harnesses provide the current typed pending-handoff store; tests that stage Study scope or initial sections use the corresponding typed channels, screen-level runtime changes recompute Study scope without claiming app-root mutation, and existing application-order, restored-state precedence, workspace/global scope, dashboard, quiz, flashcard, and source-generation behavior remains covered without production compatibility state.
- [x] #67 The app-level Study runtime callback regression commits the requested source through a real `RuntimePolicyContext`, invalidates the server-context cache, and forwards the committed source to the active screen without constructing or asserting retired writable app-root backend fields.
- [x] #68 The first-time character-chat UAT observes the typed Chat payload at the real handoff store's staging boundary, forwards it unchanged for live Console consumption, and proves settlement by the absence of pending/in-flight work plus a character-bound Console session, without polling the retired app-root handoff field or weakening the import, recovery, send, reply, and persistence journey.
- [x] #69 The first-time character-chat UAT waits for the Personas destination to be both active and mounted before invoking its import continuation, so production's stale-owner guard cannot discard the selection presentation while the imported database row, full handoff journey, and exact selected-character assertions remain covered.
- [x] #70 The app-free Console responsiveness regression stubs the current native-sync collaborators, proves the instrumented core sync executes while one worker is active, and proves the worker count returns to zero afterward without constructing a full application or bypassing the production lifecycle instrumentation.
- [x] #71 The service-backed destination worker-policy sentinel uses syntax-aware decorator inventory to account for Library's six reviewed blocking-thread workers—including verified Parakeet installation and source-ingest preflight—while retaining exact zero-thread-worker enforcement for Personas and Skills and recognizing the required worker-loop annotation anywhere in each complete `asyncio.run` call span.
- [x] #72 The File Notes Git bulk-unstage summary regression observes its retained action and postflight refresh with a bounded event-loop wait instead of requiring global Textual screen idleness, while retaining exact unstage ids, refresh count, complete displayed-snapshot summary, and the paired bulk-stage coverage.
- [x] #73 The File Notes Git bulk-stage summary regression observes its retained action and postflight refresh with the same bounded event-loop boundary, while retaining exact stage ids, refresh count, and complete displayed-snapshot summary.
- [x] #74 The Library source-action style contract extracts the exact base selector instead of the newer blocked-state selector that shares its prefix, while retaining source and bundled CSS coverage for transparent, borderless, left-aligned controls.
- [x] #75 The Library footer-ownership regression expects the `u` handoff hint only for the Search/RAG row where that action is live, while retaining the screen-owned registration and untouched app-footer contracts.
- [x] #76 The Console pending-skill-script preservation regression seeds and reads the current screen-owned task-resume state instead of the retired `chat_state` wrapper, while retaining exact preservation and clear behavior.
- [x] #77 The Parakeet MLX file-loader construction regression supplies non-empty audio metadata so it reaches the loader seam instead of the zero-frame fast path, while retaining exact loader-error chaining and stale-debug coverage.
- [x] #78 Transcription rejects a nonexistent local audio path before format conversion, provider setup, or model loading, returning `TranscriptionError` without attempting a Parakeet MLX download while preserving existing-file behavior for every provider.
- [x] #79 Model-artifact regressions that observe `os.scandir` scope their process-wide monkeypatches to the service call, preserving the traversal and directory-identity assertions without intercepting pytest's integer-file-descriptor cleanup.
- [x] #80 Full-app runtime-policy regressions suppress the unrelated startup model-catalog refresh through their existing startup helper, retaining exact action-owned notification assertions without filtering informational notifications or changing production behavior.
- [x] #81 Provider Settings regressions wait for the current Textual `Select` to mount its public `OptionList` descendant before assigning a value, instead of treating the earlier private `#label` child as readiness, while retaining exact save, placeholder, session-preservation, and handoff coverage.
- [x] #82 The real STT compatibility-facade regression supplies an existing temporary audio path before entering the mocked recognizer, preserving configured provider/language forwarding coverage while retaining the production missing-file guard.
- [x] #83 The historical v17-to-current ChaChaNotes migration regression removes the post-v17 conversation authority column before rolling back its schema version, preserving full migration replay and system-prompt trigger coverage without weakening the dedicated v27-to-v28 authority migration suite.
- [x] #84 Console schema ownership coverage distinguishes durable assistant identity from persona presentation: identity remains absent from session settings and required on native sessions, while user/persona labels and `assistant_name` remain absent from both, consistent with ADR-037.
- [x] #85 Every remaining ChaChaNotes regression that synthesizes a pre-v28 database from the current schema removes the v28 conversation authority column before replay, preserving the v16 local-marks and v20/v21 world-book migration assertions without weakening production migration validation.
- [x] #86 Incremental Chatbook import performance compares robust early and late medians so sustained slowdown still fails while a single millisecond-scale host scheduling outlier cannot fail an otherwise successful import sequence.
- [x] #87 The ChaChaNotes thread-local connection regression retains each returned connection object through its identity assertion, so short-lived thread and object-id reuse cannot collapse five distinct connections into four.
- [x] #88 TTS profile cleanup regressions scope process-wide `tempfile.mkstemp` and `os.unlink` replacements to the candidate-validation call, preserving cleanup signal/error precedence without intercepting pytest temporary-directory removal.
- [x] #89 Real Parakeet MLX integration tests run only when the installed cached module exposes the callable `from_pretrained` API required by production, while all mocked unit coverage continues to run on macOS when that runtime API is absent.
- [x] #90 Faster-whisper tests that instantiate a real model are consistently classified as slow, so the mandatory offline gate does not download model artifacts while explicit `--run-slow` runs retain the real integration coverage.
- [x] #91 Shared-RAG concurrency regressions use a fresh construction lock for their controlled race, so an unrelated in-flight application model build cannot prevent the test threads from reaching their controlled synchronization points.
- [x] #92 Evals results-grid tests wait for the selected run group's grid to mount instead of assuming one event-loop pause completes the scheduled screen recompose.
- [x] #93 ProductionApp file-notes owner lifecycle synchronization uses a bounded timeout that tolerates concurrent repository test load without turning the guards into performance assertions.
- [x] #94 Skeletal destination Console-action coverage waits for the final recovery copy instead of assuming a background Workflows load and recompose finish within a fixed sleep.
- [x] #95 File Notes Git hook-cleanup coverage gives the released commit cycle a bounded, contention-tolerant settlement guard instead of treating one second as a performance requirement.
- [x] #96 File Notes Git commit-integration synchronization uses one shared, bounded timeout for controlled signals and cycle settlement, so repository load does not turn one- or two-second literals into accidental performance gates.
- [x] #97 Hidden-action File Notes Git reopen coverage accepts focus on either visible Git navigation owner—the refreshed rows or the intentional Back-control fallback—while still rejecting focus left on the hidden entry.
- [x] #98 Native Console chat-flow synchronization uses one shared, bounded timeout for controlled fake-gateway and handoff signals, so repository load does not turn one- or two-second literals into accidental performance gates.
- [x] #99 The MCP lifecycle-cancellation regression waits for the workbench's initial workers to settle before invoking its private lifecycle seam, while preserving the deliberately blocked operation, cancel request, and in-flight cleanup assertion.
- [x] #100 Library missing-note conflict resolution waits for the old Reload or Overwrite control to leave the recomposed DOM after the editor has reset to list state, while preserving the selected-note, detail, and autosave reset assertions.
- [x] #101 The MCP lifecycle-cancellation regression lets Textual finish the post-worker mount cycle before invoking its private lifecycle seam, while preserving the deliberately blocked operation, cancel request, and in-flight cleanup assertion.
- [x] #102 The Library Database-to-Files remount journey waits for the retained File Notes editor subtree, not only its workspace root, before asserting editor identity and hidden-file refresh behavior.
- [x] #103 Library export starts its worker after the running-state recompose has refreshed, so an immediate completion updates the current canvas and leaves Export enabled for retry while preserving typed-field identity on completion.
- [x] #104 Focused Study scope-load regressions await the current deferred initial-load seam instead of the now-synchronous `on_mount`, while preserving scope precedence, validation, controller reset, initialization ordering, and runtime-backend recomputation coverage.
- [x] #105 File Notes repository-trust retries wait for each exact Cancel or Confirm control to mount before pressing it, while preserving decline/retry, identity-revalidation, fresh-status, and disabled-mutation coverage.
- [x] #106 The MCP profile-form cancellation regression waits boundedly for the newly recomposed Cancel control to render before clicking it, while preserving form dismissal, overview restoration, and zero-save coverage.
- [x] #107 A File Notes poll that completes while the workspace subtree is being detached exits without raising `NoMatches`, while normal polling still reconciles entries and retains the editor.
- [x] #108 Settings provider-navigation preselection coverage waits boundedly for the recomposed Select and Input to expose the routed provider/model values, while preserving category selection, recovery copy, and zero-draft assertions.
- [x] #109 Skill editor scroll coverage waits boundedly for focus-driven scrolling to place the Trust review control inside the canvas viewport, while preserving structural, keyboard-scroll, and positive-scroll assertions.
- [x] #110 Responsiveness artifact writer coverage supplies an explicit allowed temporary root, so a pytest `--basetemp` override outside the OS temp root does not invalidate the fixture while traversal rejection remains covered.
- [x] #111 Watchlists create-source form coverage waits boundedly for controller submission and form closure instead of relying on a fixed pause, while preserving the stronger upstream form/focus/Select readiness checks, real typing, tab order, both viewports, geometry assertions, and repeatable cross-app mounting.
- [x] #112 Real Parakeet MLX integration suites are classified as slow and selected from platform plus installed-package availability without consulting the lazy-import cache, so explicit slow runs reach production's first-use loader validation while mocked coverage remains in the mandatory gate.
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
27. Retarget Console send and Library-to-Console prompt-insert regressions to dispatch-time session and typed pending-handoff ownership.
28. Retarget Console live-work and Home-isolation regressions to the typed pending-handoff launch channel.
29. Ignore prompt-editor mount echoes that match canonical rendered state, and bridge the nested Library test harness to the active worker manager without changing production import ownership.
30. Settle the Console approval card's deferred mount callback before the multi-row geometry fixture installs its batch.
31. Seed Schedules/Workflows recent-work and Artifacts/Study destination tests through the current screen-state and typed handoff stores.
32. Align the MCP approval-cancellation regression with the metadata-only audit record.
33. Align the curated Anthropic Console default assertion with the current configured and cataloged model.
34. Retarget provider-model option regressions to the resolver's explicit saved-model and catalog-scope inputs.
35. Render the shared provider display name in missing-key recovery copy without changing canonical provider storage or routing.
36. Give the Console choose-model routing regression an explicit valid provider and missing model.
37. Publish the live-config Console fixture's fake runtime through the current projection owner.
38. Use a private file-backed Subscriptions path in that live-config journey's cross-thread scheduler fixture.
39. Retarget the resolution-view user override to the Console-owned control state.
40. Align all skill-send spy assertions with the dispatch-time session id.
41. Align the skill module's curated unknown-command hint with the complete registry.
42. Retarget the parity matrix's deleted image-attachment reference to current native Console coverage.
43. Remove the retired workspace-sync argument and its dead `ChatSessionData` import.
44. Restore visible Watchlists source nesting at the existing textual-indentation seam, update only the explanatory comments, and regenerate the bundled stylesheet.
45. Retarget the generic Speech recovery regression to the current screen-owned inspector detail and rail summary tooltip.
46. Scope the never-run Evals recovery absence assertion to the target-readiness inspector owner.
47. Scope duplicate-target Evals readiness-row counting to the target-readiness inspector owner.
48. Make local-model delete-confirm visibility independent of parent/child mount ordering and retain a real-shell show/hide regression.
49. Align both Library Collections handoff assertions with the established pressable-but-blocked recovery contract.
50. Make the different-canvas ingest isolation regression tolerate the current Library rail remount while still requiring the updated Media count.
51. Retarget MCP import-file path fixtures from process-wide `os.path.expanduser` replacement to the workbench's existing import-root seam.
52. Align MCP audit-detail fakes and assertions with the current metadata-only execution-log schema.
53. Give current Media browsing-shell fixtures a live screen owner and settle their search workers before dependent actions and assertions.
54. Retarget live Collapsible focus/hover contracts and remove CSS assertions for retired or unowned selectors.
55. Drive mounted Personas generation buttons through Textual's direct press seam so wiring coverage does not race coordinate hit testing against debounced editor validation.
56. Align the Personas import-failure assertion with the current fixed recovery copy and isolate its unused dictionary-service fixture.
57. Remove the product-maturity core-loop test's obsolete app-root handoff-consumption wait while retaining its complete visible Console outcome proof.
58. Align only the Library row in the product-maturity service-unavailable matrix with its pressable-but-blocked recovery contract while retaining disabled Watchlists and Skills rows.
59. Repair the renumbered first-run character-chat UAT task's missing YAML identity and stale heading without rewriting its completed record.
60. Migrate the focused Study test harnesses and their staged scope/section inputs to the current typed pending-handoff store without adding production fallbacks.
61. Retarget the remaining app-level Study runtime callback fixture to the current runtime-policy owner and server-context invalidation seam.
62. Observe the first-time character-chat UAT payload and settlement through the typed Chat handoff owner.
63. Wait for mounted Personas ownership before the UAT invokes its import continuation.
64. Retarget the app-free Console responsiveness fixture to the current native-sync collaborator set.
65. Synchronize the Library thread-worker policy sentinel with its two reviewed blocking operations.
66. Make the File Notes Git bulk-unstage summary wait on its retained work rather than global screen idleness.
67. Apply the same retained-work boundary to the paired bulk-stage summary after its required verification reproduces the identical Pilot timeout.
68. Make the Library source-action CSS check select its exact base rule instead of a prefix-sharing blocked modifier.
69. Align the Library footer-ownership regression with the current Search/RAG-only shortcut registration.
70. Retarget the pending-skill-script preservation test to the screen-owned task-resume state.
71. Give the Parakeet MLX loader-construction fixture non-empty audio metadata.
72. Validate the shared local-audio input boundary before conversion or provider dispatch so a missing WAV cannot trigger Parakeet model loading.
73. Scope both Model Artifacts `os.scandir` monkeypatches to their service calls so pytest cleanup sees the original standard-library function.
74. Disable unrelated startup model-catalog refresh work in the full-app runtime-policy startup helper while preserving exact notification coverage.
75. Replace the two stale provider-Select `#label` readiness waits with the mounted public `OptionList` boundary and re-query the live Select after recomposition.
76. Give the real STT compatibility-facade regression an existing temporary audio file before it calls the mocked recognizer.
77. Remove the post-v17 conversation authority column from the v17 migration fixture before replaying migrations to the current schema.
78. Align the Console schema ownership regression with ADR-037's durable assistant identity versus persona presentation boundary.
79. Remove the v28 conversation authority column from the remaining v16, v20, and v21 synthetic rollback fixtures before replaying migrations to the current schema.
80. Replace the incremental Chatbook import test's single-sample maximum-deviation assertion with an early-versus-late median degradation check.
81. Retain live connection objects in the thread-local ChaChaNotes identity regression until all worker results have been asserted.
82. Scope the two TTS profile unlink-cleanup tests' process-global standard-library patches to their candidate-validation calls.
83. Classify both real Parakeet MLX integration entry points as slow and avoid consulting the intentionally empty lazy-import cache during collection.
84. Mark the remaining faster-whisper tests that load the real tiny model as slow, matching the rest of that real-model integration class.
85. Isolate shared-RAG construction-race regressions from unrelated process-wide background builds with a class-local autouse fixture that replaces their build lock.
86. Replace the Evals results-grid test helper's single-pause mount assumption with a bounded selector wait.
87. Replace one-second file-notes lifecycle deadlock guards with a shared contention-tolerant settlement timeout.
88. Replace the skeletal Console-action test's fixed precondition sleep with a bounded wait for its exact recovery copy.
89. Give the released File Notes Git hook-cleanup commit cycle a bounded, contention-tolerant settlement timeout.
90. Apply one shared contention-tolerant timeout to the File Notes Git commit-integration module's controlled asynchronous guards.
91. Align hidden-action File Notes Git reopen focus coverage with the visible Back-or-rows fallback contract.
92. Apply one shared contention-tolerant timeout to the native Console chat-flow module's controlled fake-gateway and handoff signals.
93. Let the MCP workbench's initial workers settle before directly starting the lifecycle operation exercised by the cancellation regression.
94. Wait for the old Library conflict action to leave the recomposed DOM after missing-note resolution has reset the editor to list state.
95. Let Textual finish the MCP workbench's post-worker mount cycle before starting the cancellation regression's private lifecycle operation.
96. Require the retained File Notes editor subtree to remount before the Database-to-Files journey asserts its identity and refresh behavior.
97. Defer Library export worker dispatch until the running-state recompose has refreshed so fast completion targets the current canvas.
98. Retarget direct Study scope-load tests from the intentionally synchronous mount hook to its deferred initial-load seam.
99. Require the exact Session Git trust-dialog action control before each direct test press.
100. Wait for the MCP profile-form Cancel control to render before its coordinate click.
101. Stop a completed File Notes poll before it projects results into a partially detached workspace subtree.
102. Wait for routed Settings provider/model values instead of assuming one post-recompose pause is sufficient.
103. Wait for the Skill editor's focus-driven scroll to satisfy its exact viewport geometry.
104. Give the responsiveness artifact writer test an explicit allowed temporary root instead of coupling it to pytest's configurable base directory.
105. Wait for Watchlists controller submission and form closure instead of relying on a fixed pause, and keep the global test CSS cache from sharing its mutable rule-list container across app instances.
106. Review all changed production diagnostics and sink topology against ADR-029 before regenerating the checked inventory.
107. Run affected, static, inventory, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the dev gate by removing retired test contracts, retargeting fixtures
to current state/config/runtime owners, and replacing host-sensitive sleeps and
timing checks with exact bounded outcomes. Production changes remain narrow:
prompt dirty-state handling, Watchlists hierarchy styling, File Notes
lifecycle/export guards, missing-key provider copy, local-audio validation, and
Parakeet empty/zero-frame handling. Real Parakeet integration remains lazy and
is selected behind `--run-slow` without consulting the empty module cache. The
test CSS parse cache returns a fresh rule-list container to each app so repeated
full-shell mounts cannot corrupt the next viewport case.

PR review hardened AC #78 by routing the shared local-audio entry point through
`validate_path_simple(..., require_exists=True)` before conversion or provider
dispatch; unsafe and missing paths retain the public `TranscriptionError`
contract and cannot reach model loading.

Final review also kept the fast missing-file regression in the mandatory unit
suite, restored the CSS helper's exact base-selector input, and consolidated
the duplicated Settings value waiter without changing its bounded behavior.

ADR required: no new ADR. The work applies ADR-029, ADR-033, and ADR-037 and
does not introduce a new storage, security, runtime, or ownership boundary.

Verification resumed from the prior failure instead of replaying the cleared
suite prefix. Every corrected focused node/module passed, the final ordered
suffix passed 1,994 tests with 6 skips, and the rebased Parakeet modules pass
28 tests with 13 intentional slow skips. The final reviewed inventory contains
430 owners, 1,067 TASK-492 calls, 6,631 TASK-494 calls, and four unchanged sink
files; its checker and all three architecture guards pass. Final diff checks
are recorded in the implementation plan.
<!-- SECTION:NOTES:END -->
