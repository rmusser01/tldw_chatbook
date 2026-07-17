---
id: TASK-254
title: Stop constructing unreachable RAG_Admin services at every launch
status: Done
assignee: []
created_date: '2026-07-12 14:12'
labels:
  - rag
  - cleanup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
app.py:2545-2559 instantiates server_rag_admin_service, local_rag_admin_service and rag_admin_scope_service on every startup, but every UI consumer of these services (Embeddings_Management_Window, chunking-template widgets) is only mounted by the dead legacy SearchWindow stack and is unreachable. This adds startup cost and implies a live admin surface that does not exist. Either construct them lazily when a reachable surface actually needs them or expose a real reachable surface. Filed from the 2026-07-12 RAG module audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RAG admin services are no longer eagerly constructed at startup unless a reachable UI surface consumes them
- [x] #2 Any surface that does consume them is reachable through shell navigation
- [x] #3 App startup and test suite pass unchanged otherwise
<!-- AC:END -->

## Implementation Plan

Re-scoping (2026-07-17): the landscape moved since this task was filed. PR #669 deleted the
legacy SearchWindow UI stack (Embeddings_Management_Window, chunking-template widgets) — the
consumers named in the description no longer exist at all. PR #677 repointed
LocalRAGAdminService's collection surface at the shared RAG vector store. A fresh
`git grep` inventory on current dev shows ZERO production readers of
`app.server_rag_admin_service` / `app.local_rag_admin_service` / `app.rag_admin_scope_service`
outside the construction site in `app.py` (now lines 2652-2670): the only other references are
the RAG_Admin module itself and tests that construct the services directly (never via app
attributes). No MCP/tools/event-handler consumer appeared. AC #2 is therefore satisfied
vacuously; the honest reading of AC #1 is "defer construction so launch pays nothing", while
keeping the trio available for future surfaces (e.g. task-251 indexing controls). Deleting the
services outright is a larger product decision and out of scope here.

1. Replace the eager construction in `TldwCli.__init__` with three private `None` slots plus
   a lock, and add cached lazy properties `server_rag_admin_service` /
   `local_rag_admin_service` / `rag_admin_scope_service` backed by a single
   `_build_rag_admin_services()` builder that preserves the exact prior constructor semantics
   (config-driven `ServerRAGAdminService.from_config` with `client=None` fallback on
   ValueError, `LocalRAGAdminService(media_db, media_service=...)`, scope service wiring with
   the policy enforcer). Build under a lock so a racing first access cannot produce a
   mixed trio.
2. Update `Docs/Development/server-client-provider-migration-audit.md`'s app.py row — the
   drift guard (`Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`) matches
   the normalized `ServerRAGAdminService.from_config(` source line semantically, and that
   line changes shape when it moves into the builder.
3. Tests (`Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py`): patch the three service
   classes in the `tldw_chatbook.app` namespace, construct `TldwCli()` (repo convention:
   bare construction under Tests/conftest.py's isolated TLDW_CONFIG_PATH), assert zero
   constructions after `__init__`; then access the properties and assert exactly-once
   construction, caching (same instance on second access), and correct wiring of the scope
   service to the cached local/server instances.
4. Verify: new tests, `Tests/RAG_Admin/`, `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`,
   `Tests/Performance/test_app_import_weight.py`, `Tests/test_smoke.py`,
   `python -c "import tldw_chatbook.app"`.

## Implementation Notes

Deferred the RAG admin trio from eager `TldwCli.__init__` construction to lazy, cached,
lock-guarded properties. Startup now only initializes three `None` slots plus a
`threading.Lock`; the first access to any of `app.server_rag_admin_service` /
`app.local_rag_admin_service` / `app.rag_admin_scope_service` runs
`_build_rag_admin_services()`, which reproduces the previous wiring byte-for-byte in
semantics (config-driven `ServerRAGAdminService.from_config` with `client=None` fallback on
`ValueError`, `LocalRAGAdminService(self.media_db, media_service=self.local_media_reading_service)`,
scope service over both with `self.service_policy_enforcer`) and caches the trio; the lock
prevents a racing first access from producing a mixed trio.

Re-scoping vs. the original description: the "unreachable UI consumers" (legacy SearchWindow
stack) were deleted entirely by PR #669, so on current dev there are ZERO production readers
of these three app attributes — the construction site itself was the only reference
(`git grep` evidence in the Implementation Plan). AC #2 is satisfied vacuously (no consuming
surface exists; nothing unreachable implies a live admin surface anymore). The lazy accessor
was deliberately chosen over deleting the services: the RAG_Admin module is still the
designated seam for a future Console-parity admin surface and task-251's indexing controls,
and removing it is a bigger product decision that is out of scope for this cleanup rider.
The eager construction had no observable side effects (no health checks, no attribute reads
at startup), verified by grep and by the full app-wiring test passing.

Modified files: `tldw_chatbook/app.py` (lazy slots + builder + three properties; module-level
imports kept — heavy transitive imports were already deferred by task-163/248 work),
`Docs/Development/server-client-provider-migration-audit.md` (the drift guard in
`Tests/RuntimePolicy/test_server_client_provider_migration_audit.py` matches the normalized
`ServerRAGAdminService.from_config(` source line semantically, and the line changed shape
moving into the builder), new `Tests/RAG_Admin/test_app_lazy_rag_admin_wiring.py` (3 tests:
zero construction after `__init__`, exactly-once build + caching + trio wiring on access,
`ValueError` fallback to clientless server service — real `TldwCli()` construction using the
proven heavy-init patch recipe from `Tests/UI/test_screen_navigation.py`).

Verification (local, CI intentionally cancelled): new tests 3 passed;
`Tests/RAG_Admin/ + Tests/RuntimePolicy/test_server_client_provider_migration_audit.py +
Tests/test_smoke.py` 58 passed / 1 skipped; `Tests/Performance/test_app_import_weight.py +
Tests/Utils/test_optional_import_deferral.py` 19 passed;
`Tests/UI/test_screen_navigation.py` 56 passed; `python -c "import tldw_chatbook.app"` OK.
Startup-perf linkage: this eager construction was also flagged in the 2026-07-16 performance
audit (import/startup lane); this change removes it from the launch path without renumbering
or editing any perf tasks.
