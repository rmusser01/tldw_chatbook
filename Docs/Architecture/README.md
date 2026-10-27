# Architecture docs

This folder is the **living internals reference** for tldw_chatbook: one document per subsystem, describing how it actually works — authoritative files, dataflows, boundaries, failure behavior, and verified gotchas — anchored to the current code rather than to point-in-time plans.

It complements the other doc families:

| Family | Question it answers |
| --- | --- |
| `Docs/Architecture/` (this folder) | How does it work inside, and where is the code? |
| `Docs/User_Guide/` | How do I use it? |
| `backlog/decisions/` (ADRs) | Why was it decided this way? |
| `Docs/superpowers/{specs,plans,qa}` | What was the design/plan/evidence for a specific change? |
| `Docs/Design/`, `Docs/Development/` | Historical design notes and implementation summaries (point-in-time) |

## Index

### Cross-cutting rules

- [layer-map.md](./layer-map.md) — the four layers (kernel, DB, domain, presentation), the import-direction rule, and the shrink-only allowlist its architecture test enforces

### Shell and surfaces

- [app-shell.md](./app-shell.md) — `TldwCli` composition root, startup phases, screen-based navigation, destinations, events/workers/reactives, CSS build and design tokens, splash, settings surface
- [console.md](./console.md) — the Console: screen layout, send lifecycle, run states, approvals, branching, compaction, crash recovery
- [chat-pipeline.md](./chat-pipeline.md) — streaming, per-send transforms, swipes/branching, tool-call presentation, attachments and vision
- [ccp.md](./ccp.md) — the legacy CCP route (Conversations/Characters/Prompts → Roleplay), its live handlers vs deleted modules, and the retired Coding tab

### Agents, tools, and permissions

- [agent-runtime.md](./agent-runtime.md) — the pure agent loop, `AgentService`, budgets and stop conditions, fenced vs native tool calls, fleet sub-agents, retries and fallback
- [tool-catalog.md](./tool-catalog.md) — `ToolCatalogRegistry` provider seam, namespacing and shadowing, per-run composition, local `fs_*` tools, skills, runtime tools
- [mcp-hub.md](./mcp-hub.md) — MCP hub, external stdio servers, permission store semantics, standalone server, gateway runtime
- [console-file-authority.md](./console-file-authority.md) — per-chat scratch, workspace bindings, admitted roots, AGENTS.md project instructions and the activation ledger, workspace assistant defaults
- [console-run-hooks.md](./console-run-hooks.md) — legacy and v2 hook lifecycle events, structured effects, permission/checkpoint ownership and consent

### Providers and retrieval

- [llm-providers.md](./llm-providers.md) — `chat_api_call` dispatch, provider handlers, streaming shapes, error taxonomy, model capabilities, model-catalog auto-refresh, local backends
- [project-indexing.md](./project-indexing.md) — explicit folder-bound keyword caches, management CAS, native retirement and run-bound search evidence
- [rag.md](./rag.md) — shared RAG service, hybrid retrieval and reranking, four-seam Library search, grounded answers, incremental indexing
- [character-chat.md](./character-chat.md) — character card import, card→prompt composer, world books/world info, chat dictionaries, personas vs characters vs buddies, expressions

### Data and pipelines

- [database-layer.md](./database-layer.md) — `BaseDB` infrastructure, ChaChaNotes schema and migrations, Media DB, cross-DB patterns, satellite DBs
- [notes-sync.md](./notes-sync.md) — review-first bidirectional notes sync, root leases, recovery-first executor, session file-notes and guarded git
- [media-ingestion.md](./media-ingestion.md) — parse-without-DB worker model, per-type extraction, single write seam, chunk stamping
- [media-generation.md](./media-generation.md) — image/video adapter registries, secrets precedence, validation choke point, chat integration, ephemeral video store

### Surfaces and supporting subsystems

- [home.md](./home.md) — Home destination: per-visit dispatch, triage builder, next-best-action engine, active-work adapter, deep links
- [library-shell.md](./library-shell.md) — Library destination: rail/canvas navigation, adaptive reader, read-it-later, collections, the durable ingest job queue
- [artifacts.md](./artifacts.md) — Artifacts after ADR-172: Library browse as canonical, the composite catalog, kept briefings, Daily Reports
- [subscriptions-watchlists.md](./subscriptions-watchlists.md) — source polling, single-flight run claims, items and briefings, the artifact chain
- [scheduling-workflows.md](./scheduling-workflows.md) — unified scheduler (reminders, watchlist checks, briefings, automations), server ownership/transfer, workflow authoring and session execution, meetings
- [evals.md](./evals.md) — classic evals orchestration, word bench, character probe engines, EvalsDB
- [speech.md](./speech.md) — TTS service and adapter registry, voice profiles, PCM playback, dictation, hands-free loop
- [notifications.md](./notifications.md) — dispatch path, local inbox, producer categories, server parity feed, settings state
- [canvas.md](./canvas.md) — sandboxed conversation artifacts: immutable revisions, in-turn persistence, QuickJS gateway, canvas tools
- [theme-system.md](./theme-system.md) — theme catalog, switch/persist engine, picker/editor, the `$ds-*` interaction rule
- [metrics.md](./metrics.md) — the three metrics paths, opt-in exposition, privacy and cardinality patterns
- [research-workspace.md](./research-workspace.md) — workspace authorities, three-stage source receipts, quick notes, research runs
- [server-sync.md](./server-sync.md) — `TLDWAPIClient` safety semantics, Sync v2 outbox, `sync_log` ledger, runtime identity, scope-service pattern
- [study.md](./study.md) — Study destination: SM-2 flashcards and quizzes, honest pane status, Library/Home handoffs, local/server asymmetries
- [backup-recovery.md](./backup-recovery.md) — ADR-126 backup/restore: admission fence, data groups, encrypted archives, journaled restore, rollback
- [web-server.md](./web-server.md) — served mode (TUI in a browser): auth and bind policy, canvas seam, the artifact-share child server
- [personal-context.md](./personal-context.md) — encrypted profile store, bounded snapshot injection, agent lessons and human-reviewed promotion
- [acp.md](./acp.md) — ACP runtime launcher, session payloads, Console-follow handoff, honest boundaries

## Conventions

- Each doc names its **authoritative files** — the modules that own the subsystem's behavior. When code and any other doc disagree, code wins and the doc should be fixed.
- Dataflows are numbered and verified against the code; **gotchas** are invariants or traps verified in source, not folklore.
- Decisions link to their ADRs in `backlog/decisions/`; user-facing behavior links into `Docs/User_Guide/`.
- These docs describe the current state. When a subsystem changes materially, update its doc in the same change.

## Coverage

The set now covers every subsystem tracked by this index (34 subsystem docs plus this README). New subsystems — or material changes to an existing one — should add or update a doc here in the same change; that is the convention that keeps this reference living rather than historical.
