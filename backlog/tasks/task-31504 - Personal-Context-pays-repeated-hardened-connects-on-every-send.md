---
id: TASK-31504
title: Personal Context pays repeated hardened connects on every send
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 19:30'
updated_date: '2026-09-05 20:06'
labels:
  - performance
  - personal-context
  - chat
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing decision). ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md. Reason: document operation-scoped autocommit connection reuse and fail-closed negative-cache validity without changing authority. Add failing connection-count, invalidation, closure, thread isolation and authority-race regressions. Reuse one hardened connection per scoped operation while preserving short export snapshots and live fences; retain the Console double-build consistency fence with an owner note. Cache only unchanged absent state using content-free filesystem identity/change metadata and invalidate on setup, replacement, WAL change or errors. Never cache ready authority. Verify targeted Personal Context and agent/Console suites and record measured counts, not speculative latency.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added lazy, thread-local repository read operations and service/Console scopes. Nested reads reuse one autocommit connection and close it on success or failure; mutations keep their existing transactions. The export transaction and final live manifest, policy, binding, and revocation reads remain intact.
- Owner note: both Console authorized views remain because the second fences separately read scope identity and authority. Every reused read retains trusted-directory verification plus database identity/owner/mode/link and sidecar privacy checks.
- Cache only proven absent status using content-free DB/WAL/journal metadata. Setup, failed setup, Start Fresh, storage changes/errors and service replacement invalidate it. SHM ownership remains checked; empty WAL/journal creation/retirement cannot hide committed changes and is normalized to absent. Locked facades retain zero-connect behavior.
- Measured production composition on real SQLite: configured global **44 → 1** hardened opens; configured workspace **68 → 1**. Unchanged absent composition performs **0** opens after its first read, including WAL mode. These are connection counts, not latency claims.
- ADR required: yes, amended `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`; no schema/dependency change. Changed repository/service and Console composition; added `Tests/Personal_Context/test_send_performance.py`.
- Verification: targeted repository, service, context-service, export, durable-owner inventory, agent-provider, Console integration and import-provenance run: **169 passed, 2 existing dependency warnings**. New regressions cover count reduction, negative-cache invalidation (including external WAL-only setup), lifecycle/errors, replacement rejection, nesting, closure and threads. Ruff passes for repository/service/new tests; changed source ranges and new test formatting pass; `git diff --check` passes. Console whole-file lint has the same 27 pre-existing findings as HEAD.
- Self-review complete. Status/AC finalization awaits independent spec and quality review; no full suite, commit, or publication performed by the implementing subagent.

Independent spec review and code-quality review approved with no findings. Primary independently reran the nine-file targeted set: 169 passed, 2 dependency warnings (6.56s); quality reviewer separately ran 17 new regressions, all passed. Primary Ruff checks for repository/service/new tests, new-test formatting and diff check passed. Pytest emitted cleanup warnings for protected pre-existing garbage directories after exit 0; those unrelated directories were not modified. Implementation plan: Docs/superpowers/plans/2026-09-05-personal-context-send-performance.md. No schema or dependency changes, no full sweep, no PR or merge. Both authorized views remain intentionally per the documented owner note. Legacy unnumbered acceptance criteria were checked in the task source after CLI reported their indexes unavailable; Done status set through CLI.
<!-- SECTION:NOTES:END -->

## Description (the why)

`PersonalContextRepository._connect()` (`Personal_Context/repository.py:419`)
opens a fresh owner-checked connection per repository method (25 call sites);
each connect walks the directory path from `/` with dir_fd/O_NOFOLLOW checks
(`Utils/private_paths.py:1282`), prepares 3 sidecars, and issues extra
PRAGMAs. `Chat/console_chat_controller.py:2769`
(`_compose_profile_tool_provider`, default agent send path) builds
`authorized_context_view` TWICE per send plus `list_scopes`,
`get_scope_authority`, `get_manifest`; an unconfigured profile still pays
`status()`'s connects before failing closed, and nothing caches the negative
between sends. Measured: connect ~0.44 ms, unconfigured per-send floor
~1.75 ms off-thread; a CONFIGURED profile pays 6+ connects plus two full
export-snapshot decrypts per send, scaling with profile size. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 5.

## Acceptance Criteria (the what)

- [x] A send with Personal Context unconfigured/locked performs zero Personal Context DB connects after the first (the locked/absent status is cached with a correct invalidation story for setup/unlock)
- [x] A send with a configured profile builds the authorized view once and reuses one connection across the operation (or an owner note records why the double-build consistency check must stay)
- [x] Security semantics unchanged: fail-closed behavior and owner checks still hold (existing Personal Context authority tests stay green)
