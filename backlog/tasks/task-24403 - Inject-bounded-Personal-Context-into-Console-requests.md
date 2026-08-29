---
id: TASK-24403
title: Inject bounded Personal Context into Console requests
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-29 22:12'
labels:
  - personal-context
  - console
  - agents
dependencies:
  - TASK-24401
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide agents with one immutable, authorized, privacy-filtered Personal Context snapshot per Console turn while keeping live dispatch, Next Send preview, and child-agent propagation identical and bounded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Eligible global and mapped-workspace profile records are selected deterministically with workspace precedence, priority, expiry, visibility, authority, and unsupported-record rules
- [ ] #2 Serialized profile context is escaped, whole-record only, and bounded by both byte and token budgets
- [ ] #3 Locked, disabled, absent, and unlinked states fail closed without profile content
- [ ] #4 Console live dispatch and Next Send preview consume the exact same immutable per-turn snapshot
- [ ] #5 Agent child configurations preserve the same profile block and empty-profile behavior remains byte-identical
- [ ] #6 Targeted Personal Context, Agent, Console, and documentation verification passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED deterministic context-selection, budgeting, escaping, cache/conflict, parity, and propagation tests\n2. Implement the read-only ProfileContextService and immutable snapshot contract on top of PersonalContextService\n3. Thread the single per-turn block through AgentConfig, first-request planning, live dispatch, Next Send, and child configs\n4. Document enablement, precedence, privacy, budgeting, and exact preview behavior\n5. Run targeted tests, Ruff/format/diff checks, independent reviews, and close out\n\nADR required: no\nADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md\nReason: ADR-102 already governs authorized context injection, profile authority, privacy filtering, immutable per-turn snapshots, and Console integration.
<!-- SECTION:PLAN:END -->
