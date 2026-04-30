# Backend Parity Phase Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a live backend parity phase tracker that maps remaining Tranche 3-5 work to UX readiness, backend blockers, contract evidence, and verification status.

**Architecture:** This is a docs-only implementation. The live tracker lives separately from the roadmap and design specs, uses fixed status/source/stability vocabularies, and is seeded from current committed `dev` evidence without claiming UI implementation completion.

**Tech Stack:** Markdown, Git, focused shell verification with `git diff --check` and `rg`.

---

## File Structure

- Create: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`
- Reference: `Docs/superpowers/specs/2026-04-30-backend-parity-phase-tracker-design.md`
- Reference: `Docs/superpowers/specs/2026-04-29-backend-server-parity-handoff-roadmap-design.md`
- No production Python files should change for this tracker task.

## Task 1: Create Tracker Skeleton

**Files:**
- Create: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`

- [ ] **Step 1: Create tracker directory and file**

Use `apply_patch` to add `Docs/superpowers/trackers/backend-parity-phase-tracker.md`.

- [ ] **Step 2: Add header and vocabulary**

Include:

```markdown
# Backend Parity Phase Tracker

Status values: `done`, `ready-for-ux`, `in-progress`, `security-blocked`, `blocked`, `deferred`, `not-started`.
UX readiness values: `safe`, `partial`, `blocked`, `not-applicable`.
Source authority values: `local`, `server`, `workspace`, `mixed`, `remote-only`, `not-applicable`.
Stability values: `stable`, `provisional`, `experimental`, `blocked`, `not-applicable`.
```

- [ ] **Step 3: Add standard table schema**

Use the exact columns from the design:

```markdown
| Area | Item | Status | UX readiness | Source authority | Backend owner | Contract path | Contract version | Stability | Implementation evidence | Remaining work | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
```

- [ ] **Step 4: Verify skeleton fields exist**

Run:

```bash
rg -n "security-blocked|Source authority|Contract version|Stability" Docs/superpowers/trackers/backend-parity-phase-tracker.md
```

Expected: each term appears at least once.

## Task 2: Seed Phase Gates And Shared Foundations

**Files:**
- Modify: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`

- [ ] **Step 1: Add Phase Gate Summary**

Seed rows for:

- Connection/auth gate.
- Event/notification gate.
- Sync/mirror gate.
- Domain edge closure gate.
- UX handoff packet gate.

Connection/auth must be `security-blocked` if any credential storage, redaction, global sign-out, or server-switch security row is unresolved.

- [ ] **Step 2: Add Shared Foundations**

Seed rows for:

- Active server and capability authority.
- Credential storage and global sign-out.
- Event identity and principal scope.
- Event repository atomic persistence.
- Event observer lifecycle.
- Notification presentation projection.
- `ClientNotificationsDB` local inbox authority.
- `SyncStateRepository` dry-run mirror foundation.
- Provider migration audit.

- [ ] **Step 3: Attach committed evidence**

Use current commit evidence where applicable:

```text
35b9b19e Add server parity event and sync foundations
a68307df Wire server parity state repositories
4391e93a Wire durable server notification events
604c4b98 Scope server event state by credential principal
84b28848 Cover atomic event cursor rollback
af503adb Route research mirror reports through sync scope
```

- [ ] **Step 4: Verify no `done` row lacks evidence**

Run:

```bash
rg -n "\| done \|" Docs/superpowers/trackers/backend-parity-phase-tracker.md
```

Expected: every `done` row has non-placeholder `Contract path`, `Contract version`, `Stability`, and `Implementation evidence`.

## Task 3: Seed Domain Work Matrix

**Files:**
- Modify: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`

- [ ] **Step 1: Add primary domain rows**

Seed rows for:

- Chat.
- Media/Reading.
- Notes/Workspaces.
- Writing.
- Research.
- Study/Evaluations.
- RAG/Embeddings.
- Audio/Voice.

Each row must specify source authority and whether dry-run mirror reporting exists.

- [ ] **Step 2: Add individual remote-only utility rows**

Seed one row each for:

- Remote-only utility rollup.
- Sharing.
- Web clipper.
- Translation.
- Server tools.
- Text2SQL.
- Server skills.
- Claims.
- Meetings.
- Outputs.
- Kanban.
- Prompt Studio.

Do not collapse these into one handoff row.

- [ ] **Step 3: Add unsupported-action evidence**

For each remote-only row, include the relevant scope service or contract module if present. If evidence is not confirmed, mark the row `in-progress` or `blocked`; do not mark it `ready-for-ux`.

- [ ] **Step 4: Verify row coverage**

Run:

```bash
rg -n "Sharing|Web clipper|Translation|Text2SQL|Prompt Studio|Kanban" Docs/superpowers/trackers/backend-parity-phase-tracker.md
```

Expected: each named utility appears in its own row.

## Task 4: Add UX Handoff Readiness And Deferred Work

**Files:**
- Modify: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`

- [ ] **Step 1: Add UX Handoff Readiness section**

Seed rows for:

- Source authority and source selector contract.
- Unsupported-action report shape.
- Notification/event feed record contract.
- Sync dry-run mirror and conflict report contract.
- Domain capability matrix.
- Workspace isolation rules.
- Server unavailable/auth-expired/error presentation rules.

- [ ] **Step 2: Add Verification Ledger**

Include latest focused commands already used for this phase, including:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Research/test_research_scope_service.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Research_Interop/test_research_scope_service.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Sync_Interop/test_sync_scope_service.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_screen_navigation.py -q
```

Record known warnings without treating UI rebuild tests as blockers.

- [ ] **Step 3: Add Deferred Work section**

Include:

- Write sync and persisted local outbox.
- Background mutation replay.
- Conflict resolution beyond report-only dry-run.
- UI rebuild implementation details.
- Workflow orchestration and scheduler behavior.
- Local parity for remote-only utilities unless separately approved.

- [ ] **Step 4: Verify deferred items are explicit**

Run:

```bash
rg -n "Write sync|Background mutation replay|UI rebuild|remote-only utilities" Docs/superpowers/trackers/backend-parity-phase-tracker.md
```

Expected: all deferred items appear.

## Task 5: Final Verification And Commit

**Files:**
- Modify: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`

- [ ] **Step 1: Run Markdown whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit code 0.

- [ ] **Step 2: Review tracker diff**

Run:

```bash
git diff -- Docs/superpowers/trackers/backend-parity-phase-tracker.md
```

Expected: tracker contains all required sections and no production code changes.

- [ ] **Step 3: Commit tracker**

Run:

```bash
git add Docs/superpowers/trackers/backend-parity-phase-tracker.md
git commit -m "Add backend parity phase tracker"
```

- [ ] **Step 4: Report status**

Report:

- Tracker path.
- Commit hash.
- Verification commands and results.
- Whether branch is ahead of `origin/dev`.
