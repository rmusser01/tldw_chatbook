---
id: TASK-32670
title: Authenticate complete plugin authority snapshots
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:17'
updated_date: '2026-10-01 01:47'
labels:
  - plugins
  - implementation
  - foundation
dependencies:
  - TASK-32669
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-foundation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent package or registry tampering from changing activation, mappings or revocation without reviewed authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A separate plugin trust namespace authenticates complete registry snapshots, exact marker tuples and domain-separated prepared intents and commit certificates without resetting standalone skill trust.
- [x] #2 Activation overrides, selection, requirements, execution mappings, credential-binding generations, revocations and data-root fences are authenticated; missing or mismatched evidence blocks use.
- [x] #3 Snapshots are encrypted in the protected store outside package content and SQLite, and secrets remain credential references.
- [x] #4 Locked or unavailable trust follows ADR-009 posture rules; successful trust, offline tamper, rollback, reset and standalone-skill controls use real crypto with an isolated marker store.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (existing accepted authenticated authority and separate plugin trust namespace). ADR paths: backlog/decisions/162-managed-agent-plugins.md, backlog/decisions/163-expanded-console-hook-runtime.md and backlog/decisions/009-local-skill-trust-boundary.md. 1. Read F3 snapshot, posture, persistence and reviewed checkpoint contracts. 2. Reuse reviewed authority tests and establish the missing-module RED. 3. Integrate closed logical authority, purpose-separated keys, encrypted exact-marker snapshots/evidence and transactional plugin schema v2; preserve standalone trust and current private-write helpers. 4. Run real-crypto tamper, rollback, reset, unavailable marker, migration and standalone trust controls with isolated profile/marker storage. 5. Self-review, static and neighbor checks, record evidence/platform limits, finish ACs and commit task-owned files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reviewed F3 closed complete authority, encrypted marker-bound snapshots, exact prepared/committed evidence and separate plugin KDF namespace using existing crypto/private-write helpers. Plugin schema v2 migrates exact v1 transactionally with explicit review state and independent tombstones. Standalone skill trust is unchanged. Real-crypto/registry/owner controls: 172 passed; standalone trust/protected-path neighbors: 92 passed; no skips/warnings. Protected-path profile-binding failure reproduced on frozen pre-feature production; existing selected-profile fixture and test-owned collision cleanup repaired the harness without relaxing production guards. Six owned Python files pass full Ruff/format; shared lint unchanged, syntax/whitespace pass. Existing ADR009/162/163 linked; schema detail recorded in ADR162/spec. Evidence: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. Isolated marker backend/local macOS APFS qualified; real keyring, Windows/Linux, power loss and full sweep not claimed.
<!-- SECTION:NOTES:END -->
