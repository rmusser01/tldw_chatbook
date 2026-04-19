# tldw_chatbook Server Parity Audit Design

**Date:** 2026-04-19  
**Primary Repo:** `tldw_chatbook`  
**Reference Repos:** `tldw_server`, `hermes-agent`

## Goal

Define a structured, audit-first method for bringing `tldw_chatbook` into better feature and data-shape alignment with `tldw_server`, while preserving `tldw_chatbook` as an offline-first standalone product. The audit must also identify high-value UX and agent-workflow gaps relative to `hermes-agent`, but only promote those gaps when they materially improve `tldw_chatbook`.

## Context

The work spans three related but distinct products:

- `tldw_chatbook` is a local-first Textual TUI with existing support for chat, notes, media, RAG, evals, tools, MCP, chatbooks, and related workflows.
- `tldw_server` is the interoperability source of truth. It exposes a broader and newer API surface across chat, characters, notes/workspaces, prompts/chatbooks, media/files, RAG, evals/study features, MCP, skills, tools, and personalization-oriented surfaces.
- `hermes-agent` is not the product target, but it is a strong reference point for agent UX, tool orchestration, session ergonomics, model switching, approvals, background task handling, and similar interaction patterns.

Current repo reconnaissance also shows that `tldw_chatbook` already has uncommitted UI changes in flight. The parity effort should therefore avoid ad hoc implementation until the audit and rollout order are explicit.

## Product Assumptions

- `tldw_server` is the source of truth for compatibility targets.
- `tldw_chatbook` remains a standalone product with local/offline-first operation.
- Future sync is important, but sync itself is not part of the first execution phase.
- Admin-only, billing, org, and platform-integration server surfaces are out of scope unless they directly affect local data compatibility or user-facing workflows in `tldw_chatbook`.
- `hermes-agent` parity is a secondary overlay. It should influence the backlog only when a Hermes-style pattern improves a concrete `tldw_chatbook` workflow.

## In Scope

The audit should cover the domains that matter for interoperability and user-facing capability in `tldw_chatbook`:

- Chat and conversations
- Character chat, dictionaries, and related prompt/persona workflows
- Notes and workspace-adjacent organization
- Media ingestion and file/document handling
- RAG, search, embeddings, and chunking
- Prompts and chatbooks
- Evals and study features such as flashcards and quizzes where relevant
- MCP, tools, and skills integration
- Companion or personalization surfaces only where they map to meaningful local workflows

## Out of Scope

These areas should be excluded from the main audit unless they directly affect compatibility or local workflows:

- Billing
- Multi-user admin
- Orgs and tenant management
- Server control-plane only features
- Platform messaging integrations
- Other operational/ops surfaces that do not map into `tldw_chatbook`

## Recommended Approach

### Option A: Big-bang parity sweep

Inventory everything and immediately start implementing parity changes, UI refreshes, and Hermes-inspired improvements from one mixed backlog.

Why not recommended:

- Blends incompatible goals into one queue
- Makes prioritization subjective
- Increases overlap with the dirty local `tldw_chatbook` UI work
- Creates rework risk because the compatibility model is not stabilized first

### Option B: Audit-first capability matrix, then phased vertical implementation

Treat the first sub-project as a formal parity audit. Build a capability matrix, score gaps by business priority, extract a data compatibility map, and then implement in phases.

Why recommended:

- Matches the user's priority order: user impact first, server alignment second, Hermes gap third
- Distinguishes missing feature work from compatibility work and from UX modernization
- Produces an explicit rollout order
- Reduces churn in an already dirty working tree

### Option C: UI-first modernization, then parity backfill

Refresh `tldw_chatbook` first and fix compatibility later.

Why not recommended:

- Risks polishing the wrong workflows
- Likely causes rework when server-aligned models and APIs are introduced later
- Obscures whether a gap is cosmetic or structural

**Chosen approach:** Option B, audit first.

## Audit Model

The audit is capability-based, not endpoint-based and not file-based. Each row in the resulting matrix should represent one user-visible or interoperability-relevant capability.

Each capability row should capture:

- Domain
- Capability name
- `tldw_server` source surface
- `tldw_chatbook` current state
- Gap type
- Interoperability requirement now
- Sync relevance later
- User impact score
- Server alignment score
- Hermes UX opportunity score
- Risk/complexity
- Dependency/blocking factor
- Recommended phase

### Gap Types

Each capability should be categorized as one of:

- Missing feature
- Missing interoperability support
- Outdated UI/workflow
- Already present but structurally incompatible
- Hermes-inspired UX enhancement only

## Required Audit Outputs

The audit should produce four primary artifacts plus the working spec/plan:

1. Capability matrix  
   The master crosswalk between `tldw_server` capability surfaces and `tldw_chatbook` support.

2. Phase backlog  
   A prioritized rollout list with implementation-ready batches and clear definitions of done.

3. Data compatibility map  
   A separate cross-domain document covering local entities, server resource shapes, import/export formats, identity/versioning strategy, timestamps, metadata, deletion semantics, and likely sync-safety later.

4. Hermes UX gap memo  
   A separate memo capturing high-value interaction patterns worth borrowing from `hermes-agent`, such as clearer tool execution states, session/model controls, long-running task visibility, approval affordances, and agent workflow ergonomics.

## Scoring Model

Use a weighted formula so prioritization is repeatable and explainable:

- User Impact: `0-5`, weight `5`
- Server Alignment / Interoperability Value: `0-5`, weight `4`
- Hermes UX Opportunity: `0-5`, weight `2`
- Risk / Complexity: `0-5`, subtract weight `3`
- Dependency Blocking Factor: `0-5`, weight `3`

Formula:

```text
priority = impact*5 + alignment*4 + hermes*2 + blocking*3 - risk*3
```

Interpretation:

- High-impact, high-alignment, low-risk capabilities rise first
- Hermes-only wins should not outrank core product/interoperability work
- Foundational work can rise early when it unblocks multiple later batches

## Rollout Phases

### Phase 0: Audit and stabilization

- Complete the capability matrix
- Identify unstable or duplicated `tldw_chatbook` surfaces
- Capture current dirty-tree overlap risks
- Define canonical local models needed for later compatibility work

### Phase 1: Core interoperability primitives

- Normalize server connection and auth expectations
- Define canonical local representations for conversations, notes, characters, prompts, and chatbooks
- Clean up or centralize API client seams where needed
- Prefer import/export and compatibility seams before any real sync work

### Phase 2: Highest-impact feature parity

- Chat and conversation parity
- Character chat and related prompt/dictionary behaviors
- Notes and workspace-adjacent flows
- Prompts and chatbooks where they materially affect everyday use

### Phase 3: Retrieval and advanced workflows

- Media ingestion compatibility
- RAG/search/embeddings/chunking alignment
- Evals, flashcards, quizzes, and related study workflows where useful locally
- MCP, tools, and skills parity relevant to standalone operation

### Phase 4: UX modernization and Hermes-inspired enhancements

- Tool progress and execution-state UX
- Session ergonomics
- Model/provider switching clarity
- Background and long-running task visibility
- Approval and safety patterns where appropriate in Textual

Guiding rule: do not modernize a surface before its compatibility model is stable enough to avoid immediate rework.

## Concrete Repo Method

### Working model

1. Treat the audit as a documentation-and-analysis deliverable first.  
   No feature edits should start before the matrix, compatibility map, and rollout backlog exist and agree.

2. Use `tldw_chatbook` as the execution anchor.  
   The design spec, implementation plan, and audit artifacts should live in the client repo.

3. Treat `tldw_server` as the canonical reference corpus.  
   For every in-scope domain, map:
   - server endpoint surface
   - server data model expectations
   - local UI/module/state/DB surface
   - current gap type
   - compatibility or UX implications

4. Keep `hermes-agent` as an overlay pass.  
   Hermes findings should be recorded separately and only promoted into the implementation backlog when they improve a concrete `tldw_chatbook` workflow.

5. Isolate implementation from the current dirty tree.  
   The audit may be written from the current workspace, but implementation should either move into a dedicated worktree or explicitly reconcile the current uncommitted UI work before touching overlapping files.

### Recommended artifact layout

- `docs/superpowers/specs/2026-04-19-chatbook-server-parity-audit-design.md`
- `docs/superpowers/plans/2026-04-19-chatbook-server-parity-audit.md`
- `Docs/Parity/2026-04-19-capability-matrix.md`
- `Docs/Parity/2026-04-19-data-compatibility-map.md`
- `Docs/Parity/2026-04-19-hermes-ux-gap-memo.md`
- `Docs/Parity/2026-04-19-rollout-backlog.md`

### Audit passes

#### Pass 1: Inventory `tldw_server`

- Group relevant endpoints into product domains
- Ignore server-only control-plane/admin domains unless they affect compatibility
- Focus on chat, characters, notes/workspaces, prompts/chatbooks, media/files, RAG, evals/study, MCP/tools/skills, and relevant companion/persona surfaces

#### Pass 2: Inventory `tldw_chatbook`

- Map active UI screens, modules, local DB entities, import/export formats, and config
- Identify duplicated or legacy surfaces
- Mark areas already partially aligned versus standalone-only implementations

#### Pass 3: Crosswalk and score

- Build one matrix row per capability
- Score using the approved priority order
- Mark prerequisites and dependency chains

#### Pass 4: Define rollout batches

- Convert the highest-value items into implementation-ready batches
- Separate foundation work from user-facing parity work
- Keep each batch small enough to verify independently

## Execution Rules After The Audit

When implementation starts, each batch should follow these rules:

- One vertical at a time unless two changes are clearly independent
- Foundation before polish
- Compatibility before sync
- UI refresh only where the underlying model is already settled
- Avoid touching current in-flight `tldw_chatbook` files unless the batch explicitly requires it
- Verify each batch with focused tests or smoke coverage before advancing

Likely first implementation batches, depending on audit results:

- Core chat/conversation compatibility
- Prompts/chatbooks compatibility
- Notes/workspace model alignment
- MCP/tools integration cleanup

## Risks

- `tldw_chatbook` appears to contain duplicated, backup, and evolving UI surfaces; parity work without an audit will likely touch the wrong layer.
- `tldw_server` has a much broader API surface than the local client needs, so a strict endpoint-to-screen parity approach would waste time.
- `hermes-agent` includes strong ideas, but pulling them in without product filtering would push the TUI toward a different product identity.

## Success Criteria

This design is successful if it produces:

- A defensible parity matrix grounded in actual code and API surfaces
- A prioritized backlog that matches the user's stated ordering
- A data compatibility view that reduces future sync friction
- A phased implementation sequence that can be executed without mixing foundational and cosmetic work

## Next Step

After user approval of this written spec:

- Write the implementation plan in `docs/superpowers/plans/2026-04-19-chatbook-server-parity-audit.md`
- Begin with the audit and matrix creation rather than direct feature edits
