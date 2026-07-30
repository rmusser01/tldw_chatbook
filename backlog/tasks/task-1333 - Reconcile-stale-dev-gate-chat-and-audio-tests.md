---
id: TASK-1333
title: Reconcile stale dev-gate chat and audio tests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-29 08:11'
updated_date: '2026-07-29 18:00'
labels:
  - testing
  - baseline
  - cleanup
dependencies: []
references:
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/033-application-session-state-ownership.md
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
- [ ] #32 Console composer send regressions assert the dispatch-time session id, and Console/Library prompt-insert regressions use the live typed pending-handoff channel instead of the retired app-root prompt field, while retaining lifecycle, retry, blocked, append, collapse, navigation, dirty-editor, and empty-prompt coverage.
- [ ] #33 Console live-work helper, rendering, action, inspector, staged-context, and Home-isolation regressions stage and inspect the live typed pending-handoff channel instead of assigning or asserting the retired app-root launch field, while retaining normalized payload, consumption, navigation, action routing, and destination-isolation coverage.
- [ ] #34 Library prompt editor initialization and post-save recomposition ignore mount-time field-change echoes when the rendered fields still match canonical editor state, while genuine edits remain dirty; successful save/conflict recovery clears dirty state and the clean, empty, and dirty Console-insert paths retain their exact behavior.
- [ ] #35 The nested Library UI harness executes app-owned prompt-import workers through the active harness worker manager, retaining all exact import status and database assertions without changing production worker ownership; the real-app durable-owner and survive-unmount coverage remains green.
- [ ] #36 The bundled-CSS multi-row Console approval geometry regression lets the card's deferred mount-time hide settle before installing a batch, then retains all nonzero-size, fixed-width, non-overlap, compact-height, and action-position assertions.
- [ ] #37 Schedules/Workflows recent-work, Artifacts target, and Home-to-Study regressions seed and settle the current screen-state and typed handoff owners instead of retired app-root fields, while retaining positive recent-work, requested-before-latest, exact payload, consumption, launch, and one-hop routing coverage.
- [ ] #38 The MCP approval-cancellation audit regression expects the current metadata-only `approval_cancelled` category rather than retired persisted error text, while retaining the denied decision, blocked outcome, identity, and durable-record coverage required by ADR-029.
- [ ] #39 The curated Console remote-default regression expects Anthropic's current configured and cataloged `claude-sonnet-5` default, while retaining exact curated defaults and catalog-membership coverage for every provider in the representative set.
- [ ] #40 Console and selector merge-cap regressions pass saved provider models and the catalog scope as the current explicit resolver inputs, while retaining runtime-discovery labels, warning, cap-boundary, uncapped-picker, transient-current-model, and scope-call coverage.
- [ ] #41 Console missing-key recovery copy uses the shared human provider display name even when the active session stores a canonical lowercase provider key, while retaining the exact setup target, field, and send-blocking behavior.
- [ ] #42 The Console “Choose model” action-routing regression explicitly configures a valid provider with no selected model before opening settings, retaining the live button-click and destination assertion without relying on an internally contradictory blank-provider fixture.
- [ ] #43 The live-config Console journey publishes its fake local runtime state through the current runtime-policy projection owner instead of assigning deleted writable compatibility fields, while retaining real boot, Settings persistence, navigation, and no-restart unblocking coverage.
- [ ] #44 The live-config Console journey gives its scheduler a private file-backed Subscriptions database so schema initialized during app construction remains visible to the scheduler thread, while retaining the real cross-thread scheduler and full no-restart journey.
- [ ] #45 The Console resolution-view regression stages its user provider override through the current Console-owned control state instead of retired app-root provider/model reactives, while retaining fresh persisted-default fallback and Console-over-default precedence coverage.
- [ ] #46 Normal-send and picker-driven skill regressions assert the active dispatch-time Console session id alongside the exact raw `$name [args]` draft, while retaining controller-side skill execution and transcript behavior.
- [ ] #47 The retired bare-slash skill regression expects the complete current unknown-command hint, including image generation and rewind, while retaining non-execution, draft preservation, and second-Enter arming coverage.
- [ ] #48 The Console Workbench parity matrix references current native Console image-attachment coverage instead of a deleted legacy Chat UI test, and every matrix file/test reference resolves.
- [ ] #49 The active-conversation workspace marker regression invokes the current argument-free Console workspace sync after restoring the native persisted session, without constructing or importing retired `ChatSessionData`, while retaining the single selected-row assertion.
- [ ] #50 Expanded Watchlists source rows paint strictly to the right of their parent watchlist names at both visual-parity viewport sizes after tree labels are left-aligned; the existing relative-column regression remains intact, and source plus generated CSS documentation matches the four-space textual indent.
- [ ] #51 The generic disabled-action recovery suite mounts current `STTSScreen` ownership under missing local speech dependencies, verifies the exact phase-five recovery taxonomy in the inspector, and verifies install guidance on the rail summary tooltip.
- [ ] #52 A never-run bench renders `Not yet checked` for its target and no target-readiness recovery callout inside `#evals-inspector-bench`, while unrelated screen-level recovery callouts remain permitted.
- [ ] #53 A legacy Evals bench with a duplicate target id composes exactly two target rows in the editor and exactly two readiness rows in `#evals-inspector-bench`; all four rows render with nonzero regions and retain distinct index-derived ids despite the shared underlying target id and sibling primary-action status.
- [ ] #54 The real Lab route mounts Models without a lifecycle `NoMatches`; local-model delete confirmation is hidden on first paint and its mounted reactive state can show and hide it without mount-order errors; Lab strip navigation still completes.
- [ ] #55 Selected and empty Library Collections retain all current copy, selection, and geometry coverage while asserting the established pressable-but-blocked Console handoff state rather than a disabled button.
- [ ] #56 When an ingest completes while Notes is selected, the transiently recomposed Library rail eventually remounts its Media row with count 1, Notes remains selected, and ingest-path plus ingest-job widgets remain absent.
- [ ] #57 MCP import-file regressions override only the workbench's import-containment-root seam, so temporary picked files remain valid without replacing process-wide home expansion or redirecting the isolated application config path to a directory; picker loading, unreadable-file, outside-home rejection, and size-cap coverage remain intact.
- [ ] #58 MCP audit-detail fixtures use the current metadata-only execution-record schema; the rendered inspector retains identity, decision, duration, argument-name/count, result-type/size, and drill-through control coverage while proving raw argument values, result excerpts, and exception text are absent.
- [ ] #59 Current Media browsing-shell regressions identify the isolated mounted widget as the active screen-owned Media destination and await the Textual worker manager after background search and item-detail dispatch before inspecting results, resetting the search mock, selecting or reading a result, or asserting query/pagination calls; list population, detail loading, filter propagation, and pagination coverage remain intact without sleeps or production changes.
- [ ] #60 The non-obscuring focus contract follows Textual's live `CollapsibleTitle` DOM, pins both global and ID-scoped Library/RAG focus treatment plus the QA-scoped hover owner in source and bundle, and no longer reads or asserts CSS retired with legacy chat tabs, sidebar presets, resize controls, or an unowned `Collapsible.-active` state; all remaining static focus contracts pass without activating dead app-wide selectors.
- [ ] #61 Personas character-generation wiring tests dispatch mounted editor controls through Textual's direct `Button.press()` event seam rather than coordinate hit testing, preserving live field/context/preview/failure/regeneration and whole-character behavior while remaining deterministic when the later Library/RAG settings module is collected in the same pytest process.
- [ ] #62 The Personas character-import failure regression expects the current fixed recovery message, proves the raw importer exception text is not surfaced, declares its unused dictionary service unavailable during selection setup, and retains the selected-character state after failure.
- [ ] #63 The product-maturity Search/RAG-to-Console core-loop regression no longer reads the retired app-root `pending_chat_handoff` field and still proves the real route, staged-source count, live-work title, evidence readiness, RAG state, and suggested composer draft through current visible Console behavior.
- [ ] #64 The product-maturity service-unavailable matrix expects Library's established pressable-but-blocked Console handoff while continuing to require disabled handoffs for Watchlists and Skills, with every destination's recovery copy and tooltip retained.
- [ ] #65 The completed first-run character-chat UAT task record has valid unique `TASK-672` YAML frontmatter and a matching task heading, while its acceptance criteria, plan, implementation notes, and history remain unchanged.
- [ ] #66 Focused Study screen, dashboard, quizzes, flashcards, and product-maturity harnesses provide the current typed pending-handoff store; tests that stage Study scope or initial sections use the corresponding typed channels, screen-level runtime changes recompute Study scope without claiming app-root mutation, and existing application-order, restored-state precedence, workspace/global scope, dashboard, quiz, flashcard, and source-generation behavior remains covered without production compatibility state.
- [ ] #67 The app-level Study runtime callback regression commits the requested source through a real `RuntimePolicyContext`, invalidates the server-context cache, and forwards the committed source to the active screen without constructing or asserting retired writable app-root backend fields.
- [ ] #68 The first-time character-chat UAT observes the typed Chat payload at the real handoff store's staging boundary, forwards it unchanged for live Console consumption, and proves settlement by the absence of pending/in-flight work plus a character-bound Console session, without polling the retired app-root handoff field or weakening the import, recovery, send, reply, and persistence journey.
- [ ] #69 The first-time character-chat UAT waits for the Personas destination to be both active and mounted before invoking its import continuation, so production's stale-owner guard cannot discard the selection presentation while the imported database row, full handoff journey, and exact selected-character assertions remain covered.
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
64. Review all changed production diagnostics and sink topology against ADR-029 before regenerating the checked inventory.
65. Run affected, static, inventory, and repository-wide gates; review and close only if the full Definition of Done is satisfied.
<!-- SECTION:PLAN:END -->
