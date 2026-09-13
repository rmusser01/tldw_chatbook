# TASK-24725 — Negotiate Personal Context Sync capabilities

## Stage 1: Contract analysis and RED tests

**Goal:** Pin strict Chatbook parsing and readiness behavior for the complete
server Personal Context Sync capability contract.

**Success criteria:** Focused tests fail only because the typed capability and
readiness result are absent.

**Status:** Complete

## Stage 2: Typed client capability contract

**Goal:** Parse the server capability object through the Sync v2 client schema
without weakening existing response validation.

**Success criteria:** Complete and forward-compatible contracts parse; missing,
malformed, downgraded, or incompatible required values fail closed.

**Status:** Complete

## Stage 3: Readiness integration

**Goal:** Expose one bounded Personal Context readiness result through the
existing server sync service.

**Success criteria:** Read and write readiness require every approved domain,
schema, policy, integrity, cleanup, purge, key-distribution, and quota gate,
while unrelated Sync domains retain their current behavior.

**Status:** Complete

## Stage 4: Verification and review

**Goal:** Complete targeted regressions, static/security gates, documentation,
and independent review.

**Success criteria:** Focused Sync tests, Ruff/format, compilation, Bandit,
diff hygiene, and independent review pass; TASK-24725 is complete.

**Status:** Complete

## ADR check

ADR required: no (existing)

ADR path: `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`

Reason: ADR-102 already governs Personal Context Sync domains, capability
gating, integrity, cleanup acknowledgments, and purge generations.
