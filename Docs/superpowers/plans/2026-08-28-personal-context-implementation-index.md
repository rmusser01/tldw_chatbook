# Personal Context Profile Implementation Plan Suite

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the approved unified Personal Context Profile as four independently reviewable, dependency-ordered implementation plans spanning Shared Core, Chatbook, tldw_server, and Sync V2.

**Architecture:** A standalone Shared Profile Core owns canonical schemas and serialization. Chatbook and tldw_server each own one encrypted repository and application service, while Sync V2 replicates identical canonical objects under negotiated capabilities. Interviews, Settings, and agent tools consume those services without becoming new authorities.

**Tech Stack:** Python 3.11+, Pydantic 2, Textual 8.x, SQLite, AES-256-GCM, scrypt, HMAC-SHA-256, FastAPI, pytest, Hypothesis, Sync V2.

**Spec:** `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md` — read it first; every child plan implements part of it.

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md
Reason: The work creates a new encrypted data authority, shared schema,
migration and conflict policy, service boundary, agent permission contract,
and long-lived setup/settings UX.
```

ADR-099 is provisional until its number is swept across remote refs, worktrees,
and open PRs immediately before creation. If occupied, the implementer renumbers
the ADR and updates the spec plus every plan in this suite before code begins.

## Scope decomposition

The approved design is intentionally larger than one atomic implementation
plan. Execute these child plans in order:

1. [`2026-08-28-personal-context-01-core-chatbook-local.md`](2026-08-28-personal-context-01-core-chatbook-local.md)
   - Shared Core contract and ADR.
   - Encrypted Chatbook repository and service.
   - My Profile Settings and read-only Console context.
   - Delivers a complete local-only profile.
2. [`2026-08-28-personal-context-02-interviews-agent-tools.md`](2026-08-28-personal-context-02-interviews-agent-tools.md)
   - Personal/workspace interviews.
   - Proposal review and controlled agent tools.
   - Delivers local learning with user review.
3. [`2026-08-28-personal-context-03-server-canonical-migration.md`](2026-08-28-personal-context-03-server-canonical-migration.md)
   - Encrypted tldw_server canonical store/API.
   - Fenced legacy migration and lossless compatibility routes.
   - Delivers one server authority without Sync.
4. [`2026-08-28-personal-context-04-sync-multidevice.md`](2026-08-28-personal-context-04-sync-multidevice.md)
   - Sync V2 domains, binding, reconciliation, conflicts, cleanup, device
     lifecycle, and purge barriers.
   - Delivers shared records across Chatbook devices and tldw_server.

Each numbered task inside a child plan maps to one atomic Backlog task and one
reviewable PR unless the repository owner explicitly approves a smaller docs-only
commit in the same PR. Do not create all Backlog tasks up front: task IDs are
concurrently allocated and must be swept immediately before each task is filed.

## Dependency contract

| Consumer | Required producer |
| --- | --- |
| Chatbook local repository | Shared Core `ProfileManifest`, `ProfileScope`, `ProfileRecord`, canonical bytes |
| Chatbook interview | Chatbook `PersonalContextService`; Shared Core interview/proposal schemas |
| Chatbook profile tools | Chatbook service, runtime authority, proposal store, current trusted user-message evidence |
| Server canonical repository | Released Shared Core version used by Chatbook |
| Legacy compatibility routes | Server canonical service and completed per-user migration |
| Personal-context Sync domains | Both canonical repositories, the same Shared Core schema range, server key custody |
| Multi-device linking | Sync capabilities, server canonical manifest, Chatbook provisional manifest journal |
| Global purge | All personal-context Sync domains, device registration/expiry, content-free generation barrier |

No later plan may change a Shared Core field or service signature silently. A
required contract change begins with a Shared Core version/fixture update and
then updates every supported consumer before release.

## Global constraints

- Read the spec, ADR-008, ADR-032, ADR-037, and ADR-099 before implementation.
- Use a fresh worktree at execution time; do not implement in the current dirty
  checkout.
- Before each Backlog task, sweep all remote refs and worktrees for task-ID
  collisions, search open PRs for in-flight duplicate work, and re-read the
  rendered task after creation.
- Add the implementation plan and ADR link to the Backlog task after moving it
  to In Progress and before editing code.
- One runtime has one write service. Compatibility endpoints, UI, tools,
  interviews, migrations, and Sync never write tables directly.
- One transaction is atomic only within one SQLite database. Cross-database or
  cross-runtime effects use durable outbox/journal state and idempotent recovery.
- Profile content, kinds, provenance, drafts, proposals, conflicts, Undo data,
  and outbox payloads are encrypted at rest. Clear metadata is restricted to
  routing/version/timing/size fields named in the spec.
- `device_only` content never enters an outbox. `user_only` content never enters
  agent context, agent tools, agent search, or agent-derived artifacts.
- Runtime read/propose/direct-write grants never synchronize.
- No embeddings, semantic retrieval, autonomous consolidation, federation, or
  multi-home-server support in these plans.
- UI work uses the F9 Settings Screen and production consolidated CSS; never add
  settings to the deprecated parallel settings surfaces.
- New screens follow ADR-031 keybindings and truthful footer rules.
- Tests touching SQLite use real temporary databases. Migration tests begin
  from truthful historical schemas and reopen them through the production path.
- Privacy evidence inventories every durable owner and scans decoded persisted
  bytes, not only the primary database.
- Live Chatbook checks use a scratch config and scratch data directory and prove
  the real profile was untouched afterward.
- Live server checks begin with capabilities and one real control request; do
  not infer behavior from schemas alone.
- Run targeted tests for each task. Ask the user before a local full-repository
  sweep, per `AGENTS.md`.
- Commit after every task and stage only explicit paths. Never use `git add -A`
  in the dirty source checkout.

## Release gates

- Shared Core publishes versioned JSON Schema and fixtures before either
  consumer claims support.
- Chatbook local-only behavior ships without requiring a server.
- Server migration ships behind a per-user fence with no dual writes.
- Sync linking stays disabled unless capabilities include every required
  personal-context domain, keyed integrity tags, cleanup acknowledgments, and
  purge generation.
- Global purge and privacy-reduction integration tests pass before multi-device
  linking is enabled outside a development feature flag.
- Legacy routes are removed only after the documented deprecation period and
  operator backup-retention work is complete.
