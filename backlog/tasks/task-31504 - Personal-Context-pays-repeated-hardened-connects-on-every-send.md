---
id: TASK-31504
title: Personal Context pays repeated hardened connects on every send
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - personal-context
  - chat
dependencies: []
priority: medium
---

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

- [ ] A send with Personal Context unconfigured/locked performs zero Personal Context DB connects after the first (the locked/absent status is cached with a correct invalidation story for setup/unlock)
- [ ] A send with a configured profile builds the authorized view once and reuses one connection across the operation (or an owner note records why the double-build consistency check must stay)
- [ ] Security semantics unchanged: fail-closed behavior and owner checks still hold (existing Personal Context authority tests stay green)
