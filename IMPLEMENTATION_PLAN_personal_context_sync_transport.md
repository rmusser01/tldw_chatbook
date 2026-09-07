# TASK-24726 — Dispatch Personal Context Sync outbox

## Stage 1: Contract analysis and RED tests

**Goal:** Pin atomic profile-outbox, idempotent cross-database dispatch, and
service-owned inbound application behavior.

**Success criteria:** New focused tests fail because the dispatcher, transport
adapter, receipts, and five-domain integration do not exist.

**Status:** Complete

## Stage 2: Encrypted profile outbox lifecycle

**Goal:** Expose bounded pending, receipt, quarantine, and shredding operations
over the encrypted outbox already committed with canonical mutations.

**Success criteria:** All syncable object snapshots are atomic and encrypted;
device-only records remain local; acknowledged bodies are no longer readable.

**Status:** Complete

## Stage 3: Dispatcher and inbound adapter

**Goal:** Copy canonical snapshots idempotently into Sync state and apply pulled
objects only through `PersonalContextService`.

**Success criteria:** Crash replay is duplicate-safe, poisoned items quarantine,
HMAC/profile/scope/purge/lineage gates fail closed, and raw bodies stay private.

**Status:** Complete

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, task evidence,
and independent review.

**Success criteria:** Scoped Personal Context and Sync_Interop tests plus Ruff,
compilation, Bandit, diff hygiene, and review pass; TASK-24726 is complete.

**Status:** Complete

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs encrypted exact-wire outbox snapshots,
whole-object transport, local mutation authority, integrity, and purge fencing.
