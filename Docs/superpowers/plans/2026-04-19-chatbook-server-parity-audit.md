# Chatbook Server Parity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the parity audit artifacts and the prioritized rollout backlog needed to align `tldw_chatbook` with `tldw_server`, while capturing relevant `hermes-agent` UX improvements as a secondary overlay.

**Architecture:** This plan treats the audit itself as the first implementation deliverable. It builds a capability matrix, a data compatibility map, a Hermes UX memo, and a phased backlog before any broad feature work starts. The audit is executed from `tldw_chatbook`, uses `tldw_server` as the interoperability source of truth, and decomposes future implementation into separate vertical follow-on plans instead of one mixed mega-plan.

**Tech Stack:** Markdown, git, shell utilities (`rg`, `find`, `sed`), Python/Textual repo structure, FastAPI endpoint inventory

**Path Placeholder:** Use `"<path-to-tldw_server-repo>"` anywhere a command below needs the local checkout of the `tldw_server` repository.

---

### Task 1: Create The Audit Workspace

**Files:**
- Create: `Docs/Parity/2026-04-19-capability-matrix.md`
- Create: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Create: `Docs/Parity/2026-04-19-hermes-ux-gap-memo.md`
- Create: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Test: `Docs/Parity/2026-04-19-*.md`

- [ ] **Step 1: Create the artifact skeletons**

Use this capability matrix header:

```md
# tldw_chatbook Capability Matrix

| Domain | Capability | tldw_server Surface | tldw_chatbook Surface | Gap Type | Impact | Alignment | Hermes | Blocking | Risk | Priority | Phase | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
```

Use this compatibility map header:

```md
# tldw_chatbook Data Compatibility Map

| Entity | Local Source | Server Source | ID Strategy | Timestamp Strategy | Metadata / Versioning | Delete Semantics | Import/Export Format | Sync-Safe Later? | Notes |
|---|---|---|---|---|---|---|---|---|---|
```

- [ ] **Step 2: Verify the files and headings exist**

Run:

```bash
rg '^# ' Docs/Parity/2026-04-19-*.md
```

Expected: one top-level heading per artifact file.

- [ ] **Step 3: Commit the skeletons**

```bash
git add Docs/Parity/2026-04-19-capability-matrix.md Docs/Parity/2026-04-19-data-compatibility-map.md Docs/Parity/2026-04-19-hermes-ux-gap-memo.md Docs/Parity/2026-04-19-rollout-backlog.md
git commit -m "docs: add parity audit artifact skeletons"
```

### Task 2: Inventory `tldw_server` Capability Domains

**Files:**
- Modify: `Docs/Parity/2026-04-19-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Test: `tldw_server/tldw_Server_API/app/api/v1/endpoints/*.py`

- [ ] **Step 1: Inventory relevant endpoint domains**

Run:

```bash
find "<path-to-tldw_server-repo>"/tldw_Server_API/app/api/v1/endpoints -maxdepth 2 -type f | sed -n '1,200p'
```

Expected: endpoint files grouped enough to identify chat, characters, notes/workspaces, prompts/chatbooks, media/files, RAG, evals/study, MCP/tools/skills, and companion/persona domains.

- [ ] **Step 2: Record in-scope server capabilities in the matrix**

Create one row per capability, not one row per endpoint. Record the canonical server surface in the `tldw_server Surface` column using exact endpoint file names or endpoint families.

- [ ] **Step 3: Record excluded server-only domains**

Add a short excluded-domains section to the backlog document listing billing, orgs, admin/control-plane, and other out-of-scope server-only surfaces unless a compatibility dependency appears.

- [ ] **Step 4: Verify in-scope coverage**

Run:

```bash
rg '^## ' Docs/Parity/2026-04-19-rollout-backlog.md Docs/Parity/2026-04-19-capability-matrix.md
```

Expected: visible sections for each in-scope product domain.

- [ ] **Step 5: Commit the server inventory**

```bash
git add Docs/Parity/2026-04-19-capability-matrix.md Docs/Parity/2026-04-19-rollout-backlog.md
git commit -m "docs: inventory server capability domains for parity audit"
```

### Task 3: Inventory `tldw_chatbook` Local Surfaces

**Files:**
- Modify: `Docs/Parity/2026-04-19-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Test: `tldw_chatbook/UI/`, `tldw_chatbook/DB/`, `tldw_chatbook/Chatbooks/`, `tldw_chatbook/MCP/`, `tldw_chatbook/app.py`

- [ ] **Step 1: Map UI and module surfaces**

Run:

```bash
find tldw_chatbook/UI -maxdepth 2 -type f | sed -n '1,220p'
find tldw_chatbook/DB -maxdepth 2 -type f | sed -n '1,220p'
find tldw_chatbook/Chatbooks -maxdepth 2 -type f | sed -n '1,220p'
find tldw_chatbook/MCP -maxdepth 2 -type f | sed -n '1,220p'
```

Expected: enough local surface area to map current support or absence for every in-scope domain.

- [ ] **Step 2: Fill the `tldw_chatbook Surface` column**

For each server capability row, record the current local screen/module/DB surface using exact file paths or short grouped references.

- [ ] **Step 3: Mark legacy or duplicate surfaces**

In the `Notes` column, flag backup files, duplicated screens, partially migrated UI paths, or currently dirty files that could affect implementation ordering.

- [ ] **Step 4: Verify every server capability has a local status**

Run:

```bash
rg '\|[[:space:]]*$' Docs/Parity/2026-04-19-capability-matrix.md
```

Expected: no intentionally blank key cells for core parity rows.

- [ ] **Step 5: Commit the local inventory**

```bash
git add Docs/Parity/2026-04-19-capability-matrix.md Docs/Parity/2026-04-19-data-compatibility-map.md
git commit -m "docs: map chatbook surfaces for parity audit"
```

### Task 4: Capture `hermes-agent` UX Overlay

**Files:**
- Modify: `Docs/Parity/2026-04-19-hermes-ux-gap-memo.md`
- Modify: `Docs/Parity/2026-04-19-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Test: `hermes-agent/RELEASE_v0.8.0.md`, `hermes-agent/model_tools.py`, `hermes-agent/cli.py`, `hermes-agent/hermes_cli/`

- [ ] **Step 1: Inventory transferable UX patterns**

Review Hermes surfaces for:

```text
- tool progress and tool result presentation
- session and history ergonomics
- model switching and provider controls
- approvals and safety affordances
- background task visibility
- delegation/multi-step workflow support
```

- [ ] **Step 2: Write the UX memo**

Organize the memo into short sections:

```md
## Tool UX
## Session UX
## Safety And Approvals
## Background Tasks
## Model / Provider Controls
## Recommendations For tldw_chatbook
```

- [ ] **Step 3: Promote only justified items into the matrix/backlog**

Add Hermes-derived rows only when they materially improve `tldw_chatbook`, and mark them as `Hermes-inspired UX enhancement only` unless they also solve a parity problem.

- [ ] **Step 4: Verify Hermes items do not outrank core parity work without explanation**

Run:

```bash
sed -n '1,220p' Docs/Parity/2026-04-19-rollout-backlog.md
```

Expected: top backlog candidates are still driven by user impact and server alignment first.

- [ ] **Step 5: Commit the Hermes overlay**

```bash
git add Docs/Parity/2026-04-19-hermes-ux-gap-memo.md Docs/Parity/2026-04-19-capability-matrix.md Docs/Parity/2026-04-19-rollout-backlog.md
git commit -m "docs: add hermes UX overlay to parity audit"
```

### Task 5: Score And Sort The Capability Matrix

**Files:**
- Modify: `Docs/Parity/2026-04-19-capability-matrix.md`
- Test: `Docs/Parity/2026-04-19-capability-matrix.md`

- [ ] **Step 1: Assign gap types and numeric scores**

Populate:

```text
Impact: 0-5
Alignment: 0-5
Hermes: 0-5
Blocking: 0-5
Risk: 0-5
Priority = impact*5 + alignment*4 + hermes*2 + blocking*3 - risk*3
```

- [ ] **Step 2: Sort the matrix by practical implementation priority**

Keep the document grouped enough to read by domain, but ensure the top candidates are clearly visible in a summary section or ordered shortlist.

- [ ] **Step 3: Verify the top results match the agreed priority rule**

Check manually:

```text
1. User-facing impact in tldw_chatbook
2. Alignment with latest tldw_server capabilities
3. Hermes-agent UX/tooling gaps
```

- [ ] **Step 4: Commit the scored matrix**

```bash
git add Docs/Parity/2026-04-19-capability-matrix.md
git commit -m "docs: score and prioritize chatbook parity gaps"
```

### Task 6: Build The Data Compatibility Map

**Files:**
- Modify: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Test: `tldw_chatbook/DB/`, `tldw_chatbook/Chatbooks/`, related `tldw_server` endpoint/core files

- [ ] **Step 1: Map the core local entities**

At minimum include:

```text
conversations
messages
notes
characters
prompts
chatbooks
media
embeddings / retrieval artifacts
evaluations / study artifacts where relevant
```

- [ ] **Step 2: Record server counterparts and compatibility notes**

For each entity, capture ID shape, timestamps, metadata/versioning, delete semantics, import/export formats, and whether the current local model is likely sync-safe later.

- [ ] **Step 3: Flag structural mismatches**

Add explicit `Notes` entries for mismatches likely to force future migration or adapter work.

- [ ] **Step 4: Verify all top Phase 1 and Phase 2 items have map coverage**

Run:

```bash
sed -n '1,260p' Docs/Parity/2026-04-19-data-compatibility-map.md
```

Expected: every top backlog domain has corresponding compatibility coverage.

- [ ] **Step 5: Commit the compatibility map**

```bash
git add Docs/Parity/2026-04-19-data-compatibility-map.md
git commit -m "docs: add chatbook data compatibility map"
```

### Task 7: Build The Rollout Backlog And Choose The First Vertical

**Files:**
- Modify: `Docs/Parity/2026-04-19-rollout-backlog.md`
- Modify: `Docs/Parity/2026-04-19-capability-matrix.md`
- Test: `Docs/Parity/2026-04-19-rollout-backlog.md`

- [ ] **Step 1: Convert the highest-value matrix rows into phase deliverables**

Use phase sections:

```md
## Phase 0: Audit And Stabilization
## Phase 1: Core Interoperability Primitives
## Phase 2: Highest-Impact Feature Parity
## Phase 3: Retrieval And Advanced Workflows
## Phase 4: UX Modernization And Hermes-Inspired Enhancements
```

- [ ] **Step 2: Add a shortlist of first vertical candidates**

Include a recommendation for the first follow-on implementation plan, choosing from:

```text
- chat / conversations
- prompts / chatbooks
- notes / workspace alignment
- MCP / tools integration cleanup
```

- [ ] **Step 3: Record dirty-tree overlap risk**

Add a short section naming currently modified `tldw_chatbook` files that are likely to conflict with implementation, so follow-on work can decide whether to use a worktree or reconcile local edits first.

- [ ] **Step 4: Verify each backlog item maps back to matrix rows**

Run:

```bash
sed -n '1,260p' Docs/Parity/2026-04-19-rollout-backlog.md
```

Expected: every backlog item is concrete, scoped, and traceable to the audit.

- [ ] **Step 5: Commit the rollout backlog**

```bash
git add Docs/Parity/2026-04-19-rollout-backlog.md Docs/Parity/2026-04-19-capability-matrix.md
git commit -m "docs: create rollout backlog for chatbook parity work"
```

### Task 8: Write The Next Implementation Plan

**Files:**
- Create: `Docs/superpowers/plans/2026-04-19-chatbook-<first-vertical>.md`
- Test: `Docs/Parity/2026-04-19-rollout-backlog.md`

- [ ] **Step 1: Choose the top vertical from the backlog**

Do not pick a vertical based on preference alone. Use the scored matrix and compatibility map.

- [ ] **Step 2: Write a dedicated vertical implementation plan**

That follow-on plan should cover one subsystem only and should be the first plan that schedules code changes.

- [ ] **Step 3: Verify the vertical is small enough to ship independently**

Expected: the selected vertical has clear boundaries, a discrete verification path, and minimal dependence on unresolved audit items.

- [ ] **Step 4: Commit the follow-on plan**

```bash
git add Docs/superpowers/plans/2026-04-19-chatbook-<first-vertical>.md
git commit -m "docs: add first vertical implementation plan for chatbook parity"
```
