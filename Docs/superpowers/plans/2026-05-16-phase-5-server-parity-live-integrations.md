# Phase 5 Server Parity And Live Integrations Plan

> **For agentic workers:** execute this plan task-by-task. Use focused tests and durable QA evidence for every slice. Do not mark Phase 5 verified until the closeout replay passes.

**Goal:** Close high-value server parity and live-integration gaps that materially improve local Chatbook use while preserving local-first operation, source authority, and honest blocked states.

**Architecture:** Treat the current `dev` branch as the source of truth. The April server-parity plans are historical design inputs, not literal work queues: connection/auth, event state, dry-run sync, domain-edge contracts, and UX handoff contracts already exist. Phase 5 therefore starts with current-state reconciliation, then applies live workflow slices only where the existing backend-owned seams can support them.

**Tech Stack:** Python 3.11+, Textual, existing `RuntimeServerContextProvider`, runtime-policy/source authority contracts, `EventStateRepository`, `Sync_Interop`, `UX_Interop`, pytest, Backlog.md.

---

## Source Of Truth

- Product roadmap: `Docs/superpowers/trackers/product-maturity-roadmap.md`
- Phase owner: `backlog/tasks/task-12 - Product-Maturity-Phase-5-Server-Parity-And-Live-Integrations.md`
- Backend parity tracker: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`
- Backend handoff: `Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md`
- Server parity specs:
  - `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
  - `Docs/superpowers/specs/2026-04-29-backend-server-parity-handoff-roadmap-design.md`
- Migration audit: `Docs/Development/server-client-provider-migration-audit.md`
- Existing current-state seams:
  - `tldw_chatbook/runtime_policy/server_context.py`
  - `tldw_chatbook/runtime_policy/server_parity_models.py`
  - `tldw_chatbook/Notifications/event_state_repository.py`
  - `tldw_chatbook/Sync_Interop/`
  - `tldw_chatbook/UX_Interop/server_parity_contracts.py`

## Current Dev Inventory

Phase 5 must not recreate already-merged backend foundation work:

- Active server/auth foundation exists through `RuntimeServerContextProvider`, keyring-backed credential contracts, server switching invalidation, and UX-facing server connection contracts.
- Event/notification foundation exists through `EventStateRepository`, normalized event records, feed projection helpers, and replay-window metadata.
- Sync/mirror foundation exists as read-only/dry-run state through `Sync_Interop` repositories, readiness reports, mirror reports, and scope service wiring.
- Domain edge contracts exist for primary domains and remote-only utilities through runtime-policy source authority and unsupported-capability reports.
- UX handoff contracts exist through `server_parity_contracts.py` and the backend parity handoff packet.

Phase 5 product-maturity work should convert these foundations into usable running-app workflows. It should not enable write sync, hidden server fallback, plaintext credential storage, or broad workflow orchestration.

## Phase 5 Child Task Plan

### Task 1: Phase 5.1 Server Parity Current-State Inventory

Backlog: `TASK-12.1`

Purpose: reconcile the product-maturity roadmap with the current backend parity state before implementation.

Deliverables:

- Phase 5 QA index.
- Current-state inventory evidence.
- Child task split for Phase 5.
- Tracker regression covering plan, child tasks, evidence, and roadmap linkage.

Done when:

- `TASK-12.1` is Done.
- `TASK-12` is In Progress.
- Product roadmap says Phase 5 is in progress, not verified.

### Task 2: Phase 5.2 Active Server Auth Live Status

Backlog: `TASK-12.2`

Purpose: surface real active-server/auth/reachability state in the running app without requiring a server for local mode.

Deliverables:

- Mounted or running-app QA for missing server, configured server, auth-required, auth-expired, unreachable, and ready states.
- Focused regression proving local mode remains usable without credentials.
- UX evidence showing recovery target and authority owner for server blockers.

Non-goals:

- New credential backend.
- Plaintext fallback.
- Domain-specific server mutations.

### Task 3: Phase 5.3 Server Events And Notifications Live Feed

Backlog: `TASK-12.3`

Purpose: make server event/replay state visible and usable through Home/notification surfaces without merging server-owned read state into local notification authority.

Deliverables:

- Focused event/feed workflow replay using `EventStateRepository` fixtures or a local test server where available.
- Visible distinction between local notifications and server-owned event presentation.
- Recovery copy for replay gap, reconnect, requery, and unavailable server event states.

Non-goals:

- WebSocket feature expansion.
- Server reminder/feed write authority in local notification DB.

### Task 4: Phase 5.4 Sync Mirror Dry-Run Workflow Surfacing

Backlog: `TASK-12.4`

Purpose: expose dry-run sync/mirror readiness and conflict reports in product workflows without enabling write sync.

Deliverables:

- Library/Collections or Settings-visible dry-run status using existing `Sync_Interop` contracts.
- Focused regression proving mirror reports are read-only and do not enqueue mutations.
- QA evidence for conflict, unsupported, orphaned, and ready-to-mirror states.

Non-goals:

- Write sync.
- Mutation outbox replay.
- Automatic local/server merge.

### Task 5: Phase 5.5 High-Value Domain Parity Workflows

Backlog: `TASK-12.5`

Purpose: close the highest-value running-app domain workflows that are already supported by backend parity contracts and still materially affect the product loop.

Priority order:

1. Library/Search/RAG source authority and server availability where it affects Console handoff.
2. Personas, Skills, MCP, and ACP server-backed readiness where current Phase 4 residual risks block honest live use.
3. Schedules and Workflows run-control services only where backend contracts exist; otherwise keep them explicitly blocked with Phase 5 follow-up evidence.
4. Artifacts/Chatbook server-backed reopen/export/import behavior where it improves the core loop.

Non-goals:

- Workflow orchestration implementation from scratch.
- Broad endpoint-count parity.
- UI redesign for aesthetics.

### Task 6: Phase 5.6 Server Parity Live Integration Closeout

Backlog: `TASK-12.6`

Purpose: replay Phase 5 target workflows and decide whether the phase is verified or remains blocked by live-integration gaps.

Deliverables:

- Workflow matrix covering active server/auth, events/notifications, sync dry-run, and high-value domain parity workflows.
- Actual running-app screenshots for visible UI changes.
- Focused regression evidence.
- Explicit P0/P1 closure or accepted residuals with Phase 6/future owner.

## Risk Controls

- Local mode must stay fully usable without a server.
- Server mode must not silently fall back to local writes.
- Write sync remains deferred until a separate write-sync plan proves identity, conflict, cursor, and outbox safety.
- Server credentials must remain outside JSON/TOML/SQLite/plaintext storage.
- Runtime-policy source authority and unsupported-capability reports remain the central capability truth.
- Current backend contracts are authoritative; older April implementation plans must be reconciled before reuse.
- Visible UI changes require actual screenshot evidence before approval.

## Verification Commands

Focused planning regression:

```bash
python -m pytest -q Tests/UI/test_product_maturity_phase5_server_parity_plan.py --tb=short
```

Existing backend parity contract smoke:

```bash
python -m pytest -q Tests/RuntimePolicy/test_server_context_provider.py Tests/Notifications/test_event_state_repository.py Tests/Sync_Interop/test_sync_readiness.py Tests/UX_Interop/test_server_parity_contracts.py --tb=short
```

Diff hygiene:

```bash
git diff --check
```
