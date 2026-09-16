---
id: TASK-32670
title: Authenticate complete plugin authority snapshots
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:17'
updated_date: '2026-09-16 07:31'
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
- [ ] #1 A separate plugin trust namespace authenticates complete registry snapshots, exact marker tuples and domain-separated prepared intents and commit certificates without resetting standalone skill trust.
- [ ] #2 Activation overrides, selection, requirements, execution mappings, credential-binding generations, revocations and data-root fences are authenticated; missing or mismatched evidence blocks use.
- [ ] #3 Snapshots are encrypted in the protected store outside package content and SQLite, and secrets remain credential references.
- [ ] #4 Locked or unavailable trust follows ADR-009 posture rules; successful trust, offline tamper, rollback, reset and standalone-skill controls use real crypto with an isolated marker store.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; extends backlog/decisions/009-local-skill-trust-boundary.md
Reason: Direct implementation of the accepted independent plugin trust namespace, complete authenticated authority and protected recovery evidence.
1. Read the completed F2 schema/transaction contracts, existing skill crypto/private-file helpers and precise trust seam research; establish RED for purpose separation and real isolated store behavior.
2. Define a closed complete authority schema/projection with strict typed activation, generations, mappings/credential references and root fences; add minimal registry projection/exact-operation reads only if needed, with focused registry regressions.
3. Reuse existing scrypt/HMAC/AES-GCM primitives with separate plugin keys, salt and marker account; implement explicit empty bootstrap, unlock/posture/reset and two-way standalone independence.
4. Persist durable encrypted snapshots, domain-separated prepared intents and post-commit certificate primitives; expose authenticated exact-transition recovery reads and compare exact old/new marker tuples. F4 owns certificate creation after real database commit.
5. Exercise real-crypto tamper/rollback/locked/reduced-posture controls and existing standalone trust tests, plus production-derived sensitive-path coverage; use isolated marker backends, never real keyring/profile data.
6. Run exact affected tests and static checks, update trust operation documentation, self-review and report. Controller commits exact paths and supplies independent review before AC/Done closeout. R8/R9 settle bootstrap/digest/ordering choices; add no optional key-cache convenience subsystem.
7. Schema detail R11: add Plugins/migrations/002_authority.sql, update PluginRegistry migrations/projection and targeted registry tests for transactional v1-to-v2 upgrade and failed-upgrade rollback. Store revision review state separately from immutable revision rows and retain independent installation tombstones. Define closed mapping/operation JSON contracts. Amend ADR-162 implementation detail and relevant schema documentation before schema code; this implements its accepted trust/revocation contract rather than adding another authority. Preserve v1 file, package migration wildcard, unrelated user intent and launch evidence.
8. Independent review fix R15: retain a minimal closed protected sibling reset receipt binding this plugin-store scope, exact archive leaf and original directory identity. Persist pending reset before rename, requalify archive/file-parent durability on retries including fresh instances, and record completed only after the archive is durable. Pending reset blocks bootstrap and reports recovery; exact marker retries republish or qualify durable backend state. Add real after-rename failure/retry regressions, preserving standalone reset and existing runtime-owner gates.
9. Reset identity R16: require an explicit stable operation_id on the new plugin reset primitive and bind it into the protected reset receipt. Same-ID retries complete/return only that reset; a new reviewed reset uses a new ID. Reject a competing ID while reset is pending, and never let an old completed retry reset a newly bootstrapped namespace. Update new F3 call sites/tests/docs; no standalone API changes.
<!-- SECTION:PLAN:END -->
