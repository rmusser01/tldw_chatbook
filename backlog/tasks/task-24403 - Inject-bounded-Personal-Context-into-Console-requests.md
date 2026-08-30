---
id: TASK-24403
title: Inject bounded Personal Context into Console requests
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 22:12'
updated_date: '2026-08-30 00:11'
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
- [x] #1 Eligible global and mapped-workspace profile records are selected deterministically with workspace precedence, priority, expiry, visibility, authority, and unsupported-record rules
- [x] #2 Serialized profile context is escaped, whole-record only, and bounded by both byte and token budgets
- [x] #3 Locked, disabled, absent, and unlinked states fail closed without profile content
- [x] #4 Console live dispatch and Next Send preview consume the exact same immutable per-turn snapshot
- [x] #5 Agent child configurations preserve the same profile block and empty-profile behavior remains byte-identical
- [x] #6 Targeted Personal Context, Agent, Console, and documentation verification passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED deterministic context-selection, budgeting, escaping, cache/conflict, parity, and propagation tests\n2. Implement the read-only ProfileContextService and immutable snapshot contract on top of PersonalContextService\n3. Thread the single per-turn block through AgentConfig, first-request planning, live dispatch, Next Send, and child configs\n4. Document enablement, precedence, privacy, budgeting, and exact preview behavior\n5. Run targeted tests, Ruff/format/diff checks, independent reviews, and close out\n\nADR required: no\nADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md\nReason: ADR-102 already governs authorized context injection, profile authority, privacy filtering, immutable per-turn snapshots, and Console integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a read-only, version-fenced `ProfileContextService` that selects authorized global and explicitly mapped workspace records, applies same-kind semantic-key overrides, excludes private/expired/conflicted content, and fails closed on lifecycle or authority doubt.
- Serialized escaped user-owned JSON with whole-record admission under the lower of the 12 KiB cap and 10% of provider-aware available input tokens. Unknown newer records expose only an opaque indicator.
- Pinned one immutable snapshot in each first-request plan and threaded its exact block through Next Send, live agent requests, and child-agent configs. Preview budgeting reserves the same prompt, skill, run-log, MCP, builtin, local project-root, and Library schema inputs as live dispatch; empty profiles preserve existing request bytes.
- Updated the Console guide. ADR-102 remains the governing architecture decision; no new ADR was required.
- Verification: 30 focused Task 6 tests, 133 Personal Context/App tests, and 394 post-review Agent/Console regression tests passed. Ruff checks, focused Ruff format checks, and `git diff --check` passed. Four unrelated project-instruction failures were reproduced unchanged at parent commit `60635a9d8f` and are not Task 6 regressions.
- Independent review completed with `SPEC APPROVED` and `CODE QUALITY APPROVED`.
<!-- SECTION:NOTES:END -->
