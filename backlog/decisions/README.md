# Architecture Decision Records

Canonical ADRs for `tldw_chatbook` live in this directory.

ADRs explain why significant architectural decisions were made. Backlog tasks, Superpowers specs, and implementation plans explain what work is being done and how it is being executed.

## Index

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR-001](001-adopt-backlog-decisions-as-canonical-adrs.md) | Accepted | Use `backlog/decisions/` as the canonical ADR location and require ADR checks for significant architectural choices. |
| [ADR-002](002-openai-compatible-model-discovery.md) | Accepted | Keep OpenAI-compatible model discovery local, manual, and scoped to configured providers with explicit user persistence. |
| [ADR-003](003-settings-library-rag-defaults.md) | Accepted | Keep Library/RAG Settings scoped to persisted global defaults while Library owns active search and Console owns staged context. |
| [ADR-004](004-settings-storage-defaults-restart-boundary.md) | Accepted | Keep Settings storage defaults persisted under `database` config while active storage handles remain restart-boundary owned. |
| [ADR-005](005-console-workspace-server-readiness.md) | Accepted | Keep Console workspace switching local-first while exposing honest server-readiness, handoff, runtime, and ACP task/run states behind adapter boundaries. |
| [ADR-006](006-provider-aware-generation-settings.md) | Accepted | Keep Settings as the owner of persisted generation defaults while Console resolves effective session settings and provider adapters translate sampler/thinking controls into provider-specific request payloads. |
| [ADR-009](009-local-skill-trust-boundary.md) | Accepted | Use a passphrase-rooted authenticated trust boundary with logical quarantine for Chatbook-managed local skills. |
| [ADR-010](010-console-conversation-local-marks.md) | Accepted | Keep Console conversation stars as durable local-only marks outside conversation metadata, sync, server payloads, and chat metadata mirror reports. |
| [ADR-011](011-chatbook-workbench-ui-system.md) | Accepted | Adopt a shared Textual-native Workbench UI System with stable composition, explicit state, visible workflow controls, responsiveness gates, and route-owner migration policy. |
| [ADR-013](013-media-search-plain-text-fts-boundary.md) | Accepted | Keep raw media search text separate from optional preformatted FTS MATCH expressions across the local media-reading boundary. |
| [ADR-016](016-palette-liveness-and-hotkey-layer.md) | Accepted | Palette commands must be live (notify-only commands deleted or wired); Ctrl+N destination hotkey layer zipped from `SHELL_DESTINATION_ORDER`; generic BINDINGS-driven F1 help. |
| [ADR-018](018-watchlists-tui-screen.md) | Proposed | Replace the placeholder Watchlists destination shell with a full three-pane TUI screen reusing the local/server scope service and mirroring `tldw_server` Watchlists IA. |
| [ADR-022](022-textual-8-runtime-floor.md) | Accepted | Support Textual 8.x, test MCP against exactly 8.0.0, and fail closed on unreviewed future major versions. |
| [ADR-025](025-shared-stt-artifacts-and-runtime-routing.md) | Accepted | Use Parakeet ONNX for explicit supported languages, faster-whisper for automatic/broad routing, transcribe.cpp for curated optional breadth, and a shared verified model-artifact core. |
| [ADR-021](021-file-backed-notes-disk-authority-and-recovery.md) | Proposed | Keep linked note files disk-authoritative, project them locally, and store journaled safety plus opt-in recovery replicas in an independent recovery database. |
| [ADR-026](026-brand-asset-source-and-export-boundary.md) | Accepted | Keep validated Chatbook brand masters authoritative, commit reviewed exports, isolate the optional export toolchain from runtime/default CI, and keep asset-bearing work off the public remote until release clearance. |
| [ADR-027](027-portable-database-note-session-coordinator.md) | Accepted | Keep the active Database Note draft, revisioned save queue, editor-conflict gating, and flush outcomes in a portable coordinator outside Textual presentation and separate from File Notes authority. |
| [ADR-028](028-structured-prompts-and-auxiliary-improvement-calls.md) | Superseded by ADR-029 | Original structured Prompt and auxiliary improvement decision; replaced after discovering the existing server schema-v1 collision and composer segment boundary. |
| [ADR-029](029-local-private-data-boundary.md) | Accepted | Enforce owner-only POSIX storage for local private artifacts, make config persistence single-owner, keep persistent logs payload-free, and contain legacy Notes sync paths. |
| [ADR-029](029-versioned-prompt-artifacts-and-safe-improvement-transactions.md) | Accepted | Use schema v2 plus a first-class Prompt/Recipe discriminator, segment-safe composer transactions, recipe-fill responses, and sensitive auxiliary provider calls. |
| [ADR-030](030-derived-index-lifecycle-and-atomic-media-migrations.md) | Accepted | Keep media SQLite authoritative, reconcile derived semantic-index deletion asynchronously, and make versioned media migrations genuinely atomic. |
| [ADR-030](030-local-library-agent-tool-boundary.md) | Accepted | Share one byte-bounded lexical-only local Library read contract across Console and MCP while keeping Console mode selection and MCP policy boundaries distinct. |
| [ADR-031](031-bounded-evaluation-and-tool-worker-execution.md) | Accepted | Adapt synchronous eval providers off-loop, make eval failure and cancellation terminal states truthful, and record the retired ToolExecutor contract as superseded. |
| [ADR-032](032-immutable-installed-distribution-assets.md) | Accepted | Treat installed distributions as immutable, package explicit runtime assets and vendored notices, and verify built artifacts outside the source checkout. |
| [ADR-032](032-local-agent-tool-permission-boundary.md) | Accepted | Register workspace-local agent tools as a first-class provider reusing the MCP permission store under a synthetic server key, with fail-closed approval and workspace-root path confinement. |
| [ADR-033](033-local-agent-process-execution-boundary.md) | Accepted | Bound model-invocable process execution to fixed-argv read-only git tools without the `process` risk tag (with a binding tripwire), and reject a raw shell tool in favor of tldw_server's governed virtual-CLI design. |
| [ADR-033](033-application-session-state-ownership.md) | Accepted | Keep runtime authority, screen snapshots, and revisioned single-slot handoffs behind explicit application-scoped owners. |
| [ADR-034](034-shared-rail-disclosure-glyphs.md) | Accepted | Give the rail disclosure glyphs a single owner in `Widgets/destination_rail.py`, re-exported by `Chat/console_glyphs.py`, replacing a two-file duplication held together by a test in a third. |
| [ADR-035](035-file-notes-session-git-index-controls.md) | Accepted | Let File Notes report and reversibly stage current-session paths without taking ownership of pre-existing Git index state. |
| [ADR-036](036-application-service-composition-lifecycle.md) | Accepted | Compose application service graphs once at dependency-ready boundaries and bind them to the existing provider and Sync owners without adding a service container. |
| [ADR-037](037-roleplay-assistant-identity-and-persona-user-profile-separation.md) | Accepted | Separate assistant Personas from human User Profiles and persist authority-scoped character identity for trustworthy TTS authorship. |
| [ADR-038](038-file-notes-guarded-session-commit.md) | Accepted | Let File Notes create one reviewed local commit only when the complete staged delta exactly matches current Chatbook session ownership. |
| [ADR-039](039-global-and-studio-tts-settings-ownership.md) | Accepted | Keep application-wide TTS configuration in Settings, separately persist sparse Studio preferences, preserve character profile authority, and keep runtime operations in Lab. |
| [ADR-040](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md) | Accepted | Store Console block Prompts and Recipes as schema-v2 artifacts with a first-class discriminator, compiled compatibility text, sensitive provider calls, and segment-safe composer transactions. |
| [ADR-043](043-console-rail-compact-collapse-yields-to-explicit-toggle.md) | Accepted | Console rail compact-collapse rules are the responsive default rendering, not a hard block: explicit rail toggles are honored at any width with the main-column min-width waived, so manual toggles never silently no-op. |
| [ADR-045](045-qwencloud-dual-api-provider-boundary.md) | Accepted | Treat QwenCloud Responses and Chat Completions as two wire modes of one first-class provider while reusing the shared Console, native-tool, readiness, and model-catalog boundaries. |
| [ADR-046](046-roleplay-chat-display-identity-and-template-provenance.md) | Accepted | Keep human chat display identity separate from User Profiles and Personas, with safe resolved projections and explicit character-template provenance. |
| [ADR-049](049-local-prompt-retained-version-history.md) | Accepted | Expose local Prompt and Recipe sync snapshots as bounded retained history with indexed paging, atomic keyword capture, and conditional restore as a new current version. |
| [ADR-050](050-audio-cpp-generated-model-setup-ownership.md) | Accepted | Generate immutable audio.cpp launch artifacts from structured global settings and a built-in exact-package recipe registry while retaining the manual server.json path. |
| [ADR-051](051-private-tts-clone-reference-assets.md) | Accepted | Store canonical TTS clone references as private profile-owned assets with typed admission and separate explicit portability. |
| [ADR-052](052-console-conversation-memory-and-compaction-policy.md) | Proposed | Separate model capacity, mandatory request safety, conversation budgets, and branch-valid generated memory across their durable owners. |
| [ADR-053](053-mcp-unified-standalone-runtime-boundary.md) | Accepted | Replace FastMCP with the public `mcp-unified` strict stdio runtime while preserving Chatbook's standalone catalog, in-app Library boundary, permission policy, and bounded canonical mappings. |
| [ADR-054](054-deterministic-visual-transcript-compaction.md) | Accepted | Keep visual transcript pages deterministic, on-device, request-scoped, exactly accounted, capability-gated, and safely recoverable through text compaction. |
| [ADR-055](055-library-destructive-action-reversibility-rule.md) | Accepted | One reversibility rule for Library destructive actions: soft deletes owe a receipt + Undo with Trash as the durable story, hard deletes state permanence, draft discards confirm without receipts, and blank-note GC is the guarded named exception. |
| [ADR-056](056-context-use-visual-compaction-evaluation.md) | Accepted | Evaluate visual transcript compaction by using image history as context for downstream answers rather than requiring full transcript OCR. |
| [ADR-057](057-portable-chatbook-prompt-records.md) | Accepted | Add versioned portable Prompt records inside the existing Chatbook 1.0 Prompt content seam. |
| [ADR-058](058-thread-scoped-test-socketpair-exemption.md) | Accepted | Permit only same-thread connections made dynamically inside the real socketpair implementation while preserving the test network guard's process-wide default denial. |
| [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md) | Accepted | Use hierarchical ownership-aware Database Note folders plus device-private, journaled multi-root sync with explicit conflicts, deletion review, process coordination, and opaque server claims. |
| [ADR-073](073-notes-sync-round-trip-and-interoperability-constraints.md) | Accepted | Amend Notes lasting sync with single-binding ownership, representation-safe file writes, composite journaling, explicit Sync-v2 and backup boundaries, and per-mutation server fencing. |
| [ADR-061](061-library-ingest-parse-progress-channel.md) | Accepted | Carry local parse progress over a bounded, non-blocking, generation-fenced process channel while lifecycle results remain authoritative. |
| [ADR-062](062-hosted-chat-completions-provider-boundary.md) | Superseded by ADR-063 | Original hosted Chat wire decision with ephemeral-only reasoning continuation. |
| [ADR-063](063-hosted-provider-wire-and-durable-tool-continuation.md) | Accepted | Keep hosted Chat wire mechanics neutral and persist private provider continuation with message-owned sync, conflict, export, and model-specific replay semantics. |
| [ADR-064](064-deepseek-dual-api-provider-boundary.md) | Accepted | Treat DeepSeek Chat Completions and Responses as two strict modes of one provider with stateless explicit history and durable reasoning/tool continuation. |
| [ADR-065](065-active-ingest-source-admission-and-override.md) | Accepted | Refuse same-backend active-source ingest duplicates by default, with lexical identity, atomic folder admission, and an inline one-shot override. |
| [ADR-067](067-library-top-level-pagination-contracts.md) | Accepted | Keep top-level Library paging source-owned while standardizing exact bounded pages, stable-ID owning-page reads, complete facets/trust aggregates, and truthful recovery. |
| [ADR-068](068-console-project-instruction-context-boundary.md) | Superseded | Original Console project-instruction context boundary, replaced by ADR-069 after local-state and preflight ownership review. |
| [ADR-069](069-console-project-instruction-local-state-and-preflight.md) | Accepted | Keep project-instruction control state local-only, detect binding retargets, and prepare ephemeral tool context before unchanged security review. |
| [ADR-069](069-project-skills-folder-convention.md) | Accepted | Discover project-local `.SKILLS/` folders at startup and workspace creation, offering prompt-driven, fingerprint-gated import that copies content into the ADR-009 trust boundary quarantined rather than live-loading it. |
| [ADR-072](072-checkpoint-harness-process-ownership.md) | Accepted | Bound the suite-health checkpoint harness to retained cooperative process signals with PID-version-safe Darwin cleanup. |
| [ADR-074](074-portable-actor-packs-and-local-persona-visual-runtime.md) | Proposed | Keep Shared Visual Identity and Persona Visual as separate local runtimes inside one self-contained portable Actor Pack envelope. |
| [ADR-075](075-durable-character-emote-metadata.md) | Proposed | Match the pinned server emote grammar while persisting bounded final-expression metadata and immutable visual references. |
| [ADR-076](076-library-lifecycle-progressive-disclosure.md) | Accepted | Library uses destination-local, profile-persisted lifecycle composition rather than a global Beginner/Expert mode or a second onboarding wizard. |
| [ADR-077](077-server-offloaded-scheduled-agent-tasks.md) | Accepted | tldw_server is the execution authority for server-scoped scheduled agent work (single-owner execution, notifications pass-back, phase-1 side-effect-free runs), amending ADR-018's execution-unavailable clause. |
| [ADR-078](078-structured-agent-tool-outcome-provenance.md) | Accepted | Carry optional structured tool outcome provenance across the internal provider/runtime step boundary, with safe legacy fallback and no SQLite or external provider-wire migration. |
| [ADR-081](081-mcp-prompt-reduction-recommendations.md) | Accepted | Keep MCP prompt-reduction recommendations local-only, MCP-only, telemetry-free, and routed through the existing permission-store APIs. |
| [ADR-082](082-console-per-chat-private-scratch-space.md) | Accepted | Give each live Console chat private temporary scratch, remove implicit cwd/config authority, preserve explicit Workspace bindings, and defer cleanup safely around late tool threads. |

## Historical Decision Material

Some older decision material exists outside this directory. See [historical-index.md](historical-index.md). Historical entries are context, not canonical ADRs under the current immutability rules.

## When An ADR Is Required

Create or link an ADR when a task makes or changes a significant architectural choice, including:

- Storage, schema, migrations, sync, conflict policy, or data ownership.
- Provider/runtime boundaries, adapters, service contracts, or cross-module interfaces.
- Security, privacy, encryption, authentication, permissions, or data exposure.
- Dependency, framework, tooling, packaging, or runtime policy choices.
- Long-lived UX/application structure choices, such as navigation model or screen ownership.
- Decisions that reject a plausible alternative future contributors may ask about again.

Do not require an ADR for routine bug fixes, small UI polish, copy-only changes, mechanical refactors that preserve existing boundaries, test-only changes, or direct implementation of an existing ADR.

## Workflow

1. Read relevant ADRs before implementation planning.
2. Add an ADR check to every implementation plan:
   - `ADR required: yes/no`
   - `ADR path: backlog/decisions/NNN-short-title.md or N/A`
   - `Reason: brief explanation`
3. If an ADR is required, create it before implementation starts.
4. Link ADRs from Backlog task plans, Superpowers plans, and implementation notes when relevant.
5. At closeout, record ADRs created, superseded, or followed.

## Immutability

Accepted ADRs are immutable except for typo fixes, link repairs, or metadata that does not alter the decision. If a decision changes, create a new ADR and mark the old one as `Superseded by ADR-NNN`.

## Naming

Use numeric filenames:

- `000-template.md`
- `001-short-title.md`
- `002-short-title.md`

Do not reuse numbers.
