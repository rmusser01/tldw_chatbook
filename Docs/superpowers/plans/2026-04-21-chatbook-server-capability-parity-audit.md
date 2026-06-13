# Chatbook Server Capability Parity Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the revised 2026-04-21 parity-audit artifact set that maps `tldw_server` client-relevant capabilities to current `tldw_chatbook` coverage, target state, and rollout order.

**Architecture:** This plan treats the audit itself as the implementation target. It creates a fresh 2026-04-21 artifact set under `Docs/Parity/`, fills those docs from code-first evidence in both repos, explicitly crosswalks older Chatbook local surfaces to newer server capability names, and converts the resulting matrix into a gap ledger, target-state design, and execution roadmap. The audit remains documentation-only; it does not change product code.

**Tech Stack:** Markdown, git, shell utilities (`rg`, `find`, `sed`, `nl`, `sort`), Python/Textual repo structure, FastAPI endpoint inventory

---

## File Map

- Create: `Docs/Parity/2026-04-21-capability-matrix.md`
  Purpose: Master capability matrix with requirement class, client obligation, action coverage, evidence, scope fit, confidence, and scoring.
- Create: `Docs/Parity/2026-04-21-gap-ledger.md`
  Purpose: Flat list of parity gaps, grouped by severity and recommended tranche.
- Create: `Docs/Parity/2026-04-21-target-state-design.md`
  Purpose: Domain-by-domain local/remote operating model before sync exists.
- Create: `Docs/Parity/2026-04-21-execution-roadmap.md`
  Purpose: Tranche sequencing, dependency order, and follow-on vertical plan recommendations.
- Reference: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`
  Purpose: Source-of-truth spec for the audit contract.
- Reference: `Docs/Parity/2026-04-19-capability-matrix.md`
- Reference: `Docs/Parity/2026-04-19-data-compatibility-map.md`
- Reference: `Docs/Parity/2026-04-19-hermes-ux-gap-memo.md`
- Reference: `Docs/Parity/2026-04-19-rollout-backlog.md`
  Purpose: Older audit artifacts to mine for useful content, not to extend.
- Reference: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/`
  Purpose: Canonical client-facing server surfaces.
- Reference: `../tldw_server/tldw_Server_API/app/core/`
  Purpose: Contract-maturity and code-first evidence for thinly documented domains.
- Reference: `tldw_chatbook/tldw_api/client.py`
  Purpose: Primary Chatbook server client surface.
- Reference: `tldw_chatbook/Notes/`
- Reference: `tldw_chatbook/Media/`
- Reference: `tldw_chatbook/Study_Interop/`
- Reference: `tldw_chatbook/Evaluations_Interop/`
- Reference: `tldw_chatbook/Chatbooks/`
- Reference: `tldw_chatbook/RAG_Admin/`
- Reference: `tldw_chatbook/UI/Screens/subscription_screen.py`
- Reference: `tldw_chatbook/DB/Subscriptions_DB.py`
- Reference: `tldw_chatbook/Widgets/toast_notification.py`
- Reference: `tldw_chatbook/config.py`
  Purpose: Main local evidence sources for current Chatbook coverage.

### Working Assumptions

- Run all commands from the `tldw_chatbook` repo root.
- The sibling server checkout is available at `../tldw_server`.
- The 2026-04-21 audit replaces the 2026-04-19 artifact set as the active source of truth.
- This plan produces audit documents only. Follow-on code changes belong in later vertical implementation plans.

### Common Coverage Vocabulary

Use these values consistently in the matrix:

- Action coverage: `None`, `Partial`, `Substantial`, `Full`
- Confidence: `Low`, `Medium`, `High`
- Primary UI mode: `Separated Local/Server`, `Remote-only`, `Local-only`, `Optional mixed later`
- Client obligation: `Full CRUD`, `Discover / Configure / Trigger / Observe`, `Discover / Trigger / Observe`, `Observe-only`, `Unavailable offline with explicit fallback`

### Scoring Formula

Use the exact scoring model from the spec:

```text
priority = user*5 + interop*5 + standalone*4 + leverage*4 + ux*2 - risk*3
```

## Task 1: Create The Fresh 2026-04-21 Audit Artifact Set

**Files:**
- Create: `Docs/Parity/2026-04-21-capability-matrix.md`
- Create: `Docs/Parity/2026-04-21-gap-ledger.md`
- Create: `Docs/Parity/2026-04-21-target-state-design.md`
- Create: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Test: `Docs/Parity/2026-04-21-*.md`

- [ ] **Step 1: Verify the repo layout and sibling server checkout**

Run:

```bash
test -d ../tldw_server/tldw_Server_API/app/api/v1/endpoints
```

Expected: command exits `0` with no output.

- [ ] **Step 2: Create the capability matrix skeleton**

Use this exact starting content:

```md
# Chatbook Server Capability Matrix

Source spec: `Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md`

| Domain | Capability | Requirement class | Client obligation | Local target state | Remote target state | Primary UI mode | Browse | Detail | Create | Update | Delete | Execute/Launch | Observe/Status | Server evidence | Chatbook evidence | Verification evidence | User-scope / tenancy fit | Current gap summary | Authority policy now | Mirror / sync relevance later | Confidence | User | Interop | Standalone | Leverage | UX | Risk | Priority | Recommended tranche |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
```

- [ ] **Step 3: Create the gap ledger skeleton**

Use this exact starting content:

```md
# Chatbook Server Parity Gap Ledger

## Critical Gaps

## High-Value Partial Crosswalks

## Remote-Only Client Obligations

## Contract-Maturity Holds

## Deferred / Explicitly Out Of Scope
```

- [ ] **Step 4: Create the target-state and roadmap skeletons**

Use this exact starting content:

```md
# Chatbook Server Target-State Design

## Operating Rules

## Domain Decisions
```

```md
# Chatbook Server Execution Roadmap

## Tranche Summary

## Follow-On Vertical Plans
```

- [ ] **Step 5: Verify the files and headings exist**

Run:

```bash
rg '^# ' Docs/Parity/2026-04-21-*.md
```

Expected: one top-level heading per new artifact file.

- [ ] **Step 6: Commit the scaffold**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-target-state-design.md Docs/Parity/2026-04-21-execution-roadmap.md
git commit -m "docs: scaffold 2026-04-21 parity audit artifacts"
```

## Task 2: Inventory Server Capability Families And Contract Maturity

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Test: `../tldw_server/tldw_Server_API/app/api/v1/endpoints/`
- Test: `../tldw_server/tldw_Server_API/app/core/`

- [ ] **Step 1: Inventory the relevant server endpoint families**

Run:

```bash
find ../tldw_server/tldw_Server_API/app/api/v1/endpoints -maxdepth 2 -type f | sort | rg 'chat|character|persona|note|workspace|prompt|chatbook|reading|items|outputs|study|flashcard|quiz|evaluation|workflow|research|writing|mcp|sharing|web_clipper|watchlist|notification|reminder'
```

Expected: endpoint families for the in-scope domains listed in the spec.

- [ ] **Step 2: Add one matrix row per client-relevant capability family**

Populate rows for at least:

```text
Chat
Characters / Personas / CCP
Notes / Workspaces
Media / Reading / Ingestion Sources
Prompts / Chatbooks
Study Core
Study Packs
Study Suggestions
Collections: Reading List / Read-it-later
Collections: Outputs / Templates / Artifacts
Watchlists
Writing Suite
Research Sessions / Runs
Research Search / Provider Surfaces
Client Notifications
Server Reminders / Notification Feeds
Workflows
Scheduler Workflows
Chat Workflows
Local MCP Runtime
Remote MCP Control Plane / Governance
Sharing
Web Clipper
Evaluations
RAG / Embeddings / Chunking Admin
Cross-cutting Runtime Policy
```

- [ ] **Step 3: Verify contract maturity for the spec-flagged uncertain domains**

Run:

```bash
rg -n 'study_packs|StudyPacks|study_suggestions|research_runs|writing_manuscripts|mcp_catalog|mcp_hub' ../tldw_server/tldw_Server_API/app/api ../tldw_server/tldw_Server_API/app/core
```

Expected: evidence that lets you mark each uncertain row as one of:

```text
stable endpoint-backed
schema-backed but nested
core-only / immature client contract
present but low-confidence
```

Record that outcome in `Verification evidence` and `Confidence`, and add any blocked domains to `## Contract-Maturity Holds` in the gap ledger.

- [ ] **Step 4: Verify every matrix row has concrete server evidence**

Run:

```bash
awk -F '|' '
/^\|/ && $0 !~ /^\| ---/ && $0 !~ /^\| Domain / {
  if ($16 ~ /^[[:space:]]*$/) {
    print "Missing Server evidence at line " NR ": " $0
    bad = 1
  }
}
END { exit bad }
' Docs/Parity/2026-04-21-capability-matrix.md
```

Expected: no rows where `Server evidence` is blank.

- [ ] **Step 5: Commit the server inventory**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md
git commit -m "docs: inventory server capability contracts"
```

## Task 3: Inventory Chatbook Surfaces And Crosswalk Existing Local Names

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Test: `tldw_chatbook/tldw_api/client.py`
- Test: `tldw_chatbook/Notes/`
- Test: `tldw_chatbook/Media/`
- Test: `tldw_chatbook/Study_Interop/`
- Test: `tldw_chatbook/Evaluations_Interop/`
- Test: `tldw_chatbook/Chatbooks/`
- Test: `tldw_chatbook/RAG_Admin/`
- Test: `tldw_chatbook/UI/Screens/subscription_screen.py`
- Test: `tldw_chatbook/DB/Subscriptions_DB.py`
- Test: `tldw_chatbook/Widgets/toast_notification.py`
- Test: `tldw_chatbook/config.py`

- [ ] **Step 1: Inventory the main Chatbook service and UI surfaces**

Run:

```bash
rg -n 'server_|runtime_backend|scope_service|list_|create_|update_|delete_|watchlist|subscription|reminder|notification|workflow|research|writing|mcp' tldw_chatbook/tldw_api/client.py tldw_chatbook/Notes tldw_chatbook/Media tldw_chatbook/Study_Interop tldw_chatbook/Evaluations_Interop tldw_chatbook/Chatbooks tldw_chatbook/RAG_Admin tldw_chatbook/UI/Screens/subscription_screen.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Widgets/toast_notification.py tldw_chatbook/config.py
```

Expected: concrete evidence for what Chatbook already supports locally, remotely, or not at all.

- [ ] **Step 2: Fill the `Chatbook evidence` column for every server capability row**

Use exact file paths or tightly grouped file references such as:

```text
tldw_chatbook/tldw_chatbook/Media/media_reading_scope_service.py
tldw_chatbook/tldw_chatbook/Study_Interop/server_study_service.py
tldw_chatbook/tldw_chatbook/UI/Screens/subscription_screen.py
```

- [ ] **Step 3: Perform the explicit `Subscriptions -> Watchlists` crosswalk**

Add a short subsection to `Docs/Parity/2026-04-21-gap-ledger.md` with:

```md
## Local Name Crosswalks

- `Subscriptions` -> likely local precursor for server `Watchlists`
```

Then update the watchlists matrix row so its gap summary distinguishes:

```text
existing local monitoring and briefing behaviors
missing remote watchlist interoperability
missing exact server vocabulary / contract alignment
```

- [ ] **Step 4: Verify every matrix row has Chatbook evidence or an explicit absence**

Run:

```bash
awk -F '|' '
/^\|/ && $0 !~ /^\| ---/ && $0 !~ /^\| Domain / {
  if ($17 ~ /^[[:space:]]*$/) {
    print "Missing Chatbook evidence at line " NR ": " $0
    bad = 1
  }
}
END { exit bad }
' Docs/Parity/2026-04-21-capability-matrix.md
```

Expected: no rows where `Chatbook evidence` is blank.

- [ ] **Step 5: Commit the Chatbook inventory**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md
git commit -m "docs: map chatbook surfaces for parity audit"
```

## Task 4: Populate Requirement Classes, Client Obligations, And Action Coverage

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Test: `Docs/Parity/2026-04-21-capability-matrix.md`

- [ ] **Step 1: Add a legend section above the matrix**

Use this exact content:

```md
## Legend

- Action coverage values: `None`, `Partial`, `Substantial`, `Full`
- Primary UI mode values: `Separated Local/Server`, `Remote-only`, `Local-only`, `Optional mixed later`
- Client obligation values: `Full CRUD`, `Discover / Configure / Trigger / Observe`, `Discover / Trigger / Observe`, `Observe-only`, `Unavailable offline with explicit fallback`
```

- [ ] **Step 2: Fill `Requirement class`, `Client obligation`, `Local target state`, `Remote target state`, and `Primary UI mode` for every row**

Use the spec rules directly. In particular:

```text
remote-only domains still need discoverability and explicit offline fallback
dual-surface domains default to Separated Local/Server
local-first domains keep local write authority before sync exists
```

- [ ] **Step 3: Fill all seven action-coverage columns**

For each row, assign:

```text
Browse
Detail
Create
Update
Delete
Execute/Launch
Observe/Status
```

using only `None`, `Partial`, `Substantial`, or `Full`.

- [ ] **Step 4: Fill `User-scope / tenancy fit`, `Authority policy now`, and `Mirror / sync relevance later`**

Make the tenancy-fit column answer questions like:

```text
single-user local only
single-user client against multi-user server subset
server-owned shared resource surfaced to one user
```

- [ ] **Step 5: Verify there are no blank policy cells**

Run:

```bash
awk -F '|' '
/^\|/ && $0 !~ /^\| ---/ && $0 !~ /^\| Domain / {
  for (i = 4; i <= 23; i++) {
    if ($i ~ /^[[:space:]]*$/) {
      print "Blank policy/action cell at line " NR ", column " i ": " $0
      bad = 1
    }
  }
}
END { exit bad }
' Docs/Parity/2026-04-21-capability-matrix.md
```

Expected: no blank cells in the policy and action-coverage region of the table.

- [ ] **Step 6: Commit the policy-filled matrix**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md
git commit -m "docs: add parity obligations and action coverage"
```

## Task 5: Score The Matrix And Write The Gap Ledger

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Test: `Docs/Parity/2026-04-21-capability-matrix.md`
- Test: `Docs/Parity/2026-04-21-gap-ledger.md`

- [ ] **Step 1: Add the numeric scores for every row**

Populate:

```text
User
Interop
Standalone
Leverage
UX
Risk
Priority
```

Use the exact formula from the spec and keep the raw numbers visible in the matrix.

- [ ] **Step 2: Add a top-priority summary section above the matrix**

Use this exact heading:

```md
## Top Priority Rows
```

List the highest-priority rows in descending order with one-line justifications.

- [ ] **Step 3: Write the gap ledger entries from the completed matrix**

Use this entry template for each important gap:

```md
### [Domain]: [Capability]

- Requirement class:
- Client obligation:
- Current state:
- Gap:
- Evidence:
- Recommended tranche:
- Notes:
```

- [ ] **Step 4: Verify the top-ranked gaps match the agreed priorities**

Run:

```bash
sed -n '1,220p' Docs/Parity/2026-04-21-capability-matrix.md
sed -n '1,240p' Docs/Parity/2026-04-21-gap-ledger.md
```

Expected: the highest-ranked items are driven by user value, interoperability value, and standalone value before cosmetic UX work.

- [ ] **Step 5: Commit the scored matrix and gap ledger**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md
git commit -m "docs: score parity matrix and write gap ledger"
```

## Task 6: Write The Target-State Design

**Files:**
- Modify: `Docs/Parity/2026-04-21-target-state-design.md`
- Test: `Docs/Parity/2026-04-21-target-state-design.md`

- [ ] **Step 1: Write the operating-rules section from the spec**

Include explicit rules for:

```text
Separated Local/Server primary UI
remote-only discoverability
offline fallback behavior
local write authority before sync
future mirror relevance without implementing sync
```

- [ ] **Step 2: Add one short section per major domain family**

Use these headings:

```md
## Core Dual-Backend Domains
## Local-First Domains With Remote Interop
## Remote-Only Domains
## Cross-Cutting Runtime Policy
```

- [ ] **Step 3: For each domain, record the target state**

Each domain entry must answer:

```text
where data is authored now
where it is viewed from now
what Chatbook must do locally
what Chatbook must do remotely
how it behaves offline
whether mixed view is deferred
```

- [ ] **Step 4: Verify target-state decisions do not contradict the matrix**

Run:

```bash
rg '^## ' Docs/Parity/2026-04-21-target-state-design.md
```

Expected: sections exist for all major domain families and reflect the matrix decisions.

- [ ] **Step 5: Commit the target-state design**

```bash
git add Docs/Parity/2026-04-21-target-state-design.md
git commit -m "docs: add parity target-state design"
```

## Task 7: Write The Execution Roadmap

**Files:**
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Test: `Docs/Parity/2026-04-21-execution-roadmap.md`

- [ ] **Step 1: Write the tranche summary**

Use these exact headings:

```md
## Tranche 0: Runtime Policy And Capability Map
## Tranche 1: Strengthen Existing Dual-Backend Domains
## Tranche 2: Add Missing High-Value Local / Remote Surfaces
## Tranche 3: Remote-Only Surfaces And Convenience Layers
```

- [ ] **Step 2: Place domains into tranches using the matrix and gap ledger**

Bias the roadmap toward:

```text
existing partial dual-backend seams first
crosswalked local surfaces before from-scratch replacements
remote-only surfaces after core local/remote parity domains
```

- [ ] **Step 3: Add a follow-on vertical-plan list**

Use this exact heading:

```md
## Follow-On Vertical Plans
```

List candidate vertical plans such as:

```text
watchlists / subscriptions alignment
remote MCP control plane
writing suite parity
research sessions parity
remote workflows surface
study packs / study suggestions once contract maturity is confirmed
```

- [ ] **Step 4: Verify the roadmap does not skip tranche prerequisites**

Run:

```bash
sed -n '1,260p' Docs/Parity/2026-04-21-execution-roadmap.md
```

Expected: later tranches depend on earlier runtime-policy and evidence-building work, not the other way around.

- [ ] **Step 5: Commit the roadmap**

```bash
git add Docs/Parity/2026-04-21-execution-roadmap.md
git commit -m "docs: add parity execution roadmap"
```

## Task 8: Final Audit QA And Handoff

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-target-state-design.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Test: `Docs/Parity/2026-04-21-*.md`

- [ ] **Step 1: Verify the new artifact set is internally linked and dated consistently**

Run:

```bash
rg '2026-04-21|Docs/superpowers/specs/2026-04-21-chatbook-server-capability-parity-audit-design.md' Docs/Parity/2026-04-21-*.md
```

Expected: all four artifacts reference the correct date and the correct spec path where needed.

- [ ] **Step 2: Verify there are no leftover placeholder markers**

Run:

```bash
rg 'TODO|TBD|REPLACE ME|FIXME' Docs/Parity/2026-04-21-*.md
```

Expected: no placeholder markers remain.

- [ ] **Step 3: Verify the 2026-04-21 artifact set exists as a complete bundle**

Run:

```bash
ls Docs/Parity/2026-04-21-*.md
```

Expected: exactly four files.

- [ ] **Step 4: Review the full artifact set manually**

Run:

```bash
sed -n '1,260p' Docs/Parity/2026-04-21-capability-matrix.md
sed -n '1,240p' Docs/Parity/2026-04-21-gap-ledger.md
sed -n '1,240p' Docs/Parity/2026-04-21-target-state-design.md
sed -n '1,240p' Docs/Parity/2026-04-21-execution-roadmap.md
```

Expected: the docs agree with each other and with the spec.

- [ ] **Step 5: Commit the final QA pass**

```bash
git add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-target-state-design.md Docs/Parity/2026-04-21-execution-roadmap.md
git commit -m "docs: finalize 2026-04-21 parity audit artifacts"
```
