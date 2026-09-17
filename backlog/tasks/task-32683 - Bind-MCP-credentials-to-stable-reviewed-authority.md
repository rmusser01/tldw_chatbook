---
id: TASK-32683
title: Bind MCP credentials to stable reviewed authority
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:27'
updated_date: '2026-09-17 22:09'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32682
  - TASK-32670
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep normal token renewal usable while ensuring account, endpoint or scope changes cannot inherit stale plugin authority.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owning credential service exposes stable reference, principal, issuer, audience/endpoint and scope bindings with separate authority and storage revisions.
- [ ] #2 Verified renewal within unchanged authority resolves current credentials at dispatch without invalidating plugin trust; changed or unknown authority requires reconciliation.
- [ ] #3 Explicit header/token mappings and supported OAuth flows use existing host credential services; unsupported auth is a specific unready state and no vendor connector grant is imported.
- [ ] #4 Secret sentinels stay out of trust snapshots, catalogs, receipts, debug defaults and errors; origin changes, failed refresh, opaque replacement and store migration have successful and refusal controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Implement the accepted stable credential authority and R24 owning-service recovery contracts without adding another credential, permission or OAuth owner.

1. Read task16 brief, actual TASK32683, completed authentication/owner-seam research and final M1-M2/F3-F4 APIs. Inventory the actual host config/encryption/keyring boundary and supported auth adapters before exposure. Record exact additional recovery/coordinator/registry, migration, test and documentation paths via CLI before editing them. Unavailable generic MCP OAuth returns unsupported_authentication; do not reinterpret arbitrary-origin tldw_server credentials.
2. Establish behavioral RED with real CredentialBindingService and an isolated fake credential backend at the service/transport dispatch entry. Verify same-authority renewal and account-change controls using verified host callbacks, separate from token bytes or plugin claims. Missing imports alone are scaffolding evidence.
3. Add MCP/credential_bindings.py and wire local_store.py, local_control_service.py, streamable_http.py and config.py. Preserve frozen stable reference/authority-generation/method/issuer/audience/origin/principal/scopes with explicit unknown values; separate token bytes, expiry and storage revision. Resolve current usable secrets only at the selected-origin dispatch boundary and preserve complete typed MCP result provenance.
4. Verified renewal with unchanged authority retains generation and reviewed mappings. Changed account/issuer/audience/origin/scope/reference, revocation or unverifiable opaque replacement invalidates captured authority and requires reconciliation. Implement explicit host header/token mappings and only actually supported OAuth adapters; no vendor grant import, automatic uncertain-call replay or inferred identity from token content.
5. Complete R24 by integrating real credential-reference validation into F4 authenticated snapshot reconstruction against retained immutable material. Qualify supported current-reference recovery and refusal of missing/changed bindings. A MAC-covered raw mapping does not prove external authority, and recovery cannot create grants. Extend only the current owner seams recorded before code.
6. Cover pending review/live-run renewal, expired/failed refresh, origin redirect, unknown issuer/principal, changed scope, opaque replacement and explicit store migration. Assert secret sentinels absent from repr, snapshots, catalogs, receipts, statuses, logs, errors and provider context. Run the three exact brief files and recorded recovery/store neighbors under isolated profiles/fake backend, verified worktree provenance and frozen source/tests; no real credentials or full sweep. Record static baseline parity, commands/output, API/recovery evidence and future handoffs in task-16-report.md and recorded docs. Controller owns commit, independent review, four ACs/notes and Done.

Verification handoff from H2 review F3: any child bootstrap/network/profile fixture needed by committed tests must be repository-owned and use the actual pytest temporary root. A private implementation runner may isolate the parent, but cannot be a hidden prerequisite for the ordinary exact test command. Reuse the final reviewed H2 fixture where appropriate; controlled HTTP tests must explicitly scope their own loopback allowance rather than disable network checks globally. Record extra helper paths before edits.

7. Activation BASE is68ebe2accbd10bd49dd7ad88fc08d236bdbb0b8d after reviewed M2 Fix1/closeout. Read the final M2 handoff, m3-activation-precheck.md and task-16-paths.json;29 exact tracked snapshots and4 absent new paths are recorded before code. In addition to the eight original runtime/test paths, existing ADR162, the managed-plugin spec and MCP subplan may document this task's supported authentication/migration/owner boundaries. Recovery/coordinator/registry/authority, MCPClient, sensitive-config classifier and other seams remain read-only until the worker proposes the concrete adapter and root records exact edit paths. Preserve M2 raw-before-coercion validation, R60/R61, incomplete-cleanup retention and stdio ordering. Private M3 runner uses the unchanged committed bootstrap/default refusal; controlled peers permit only their owned loopback address/port. Establish the first real owner-entry RED before production edits; do not fabricate a credential mapping kind or use token/plugin claims as verified identity.

8. R62 resolves the public-mapping insertion boundary before owner integration. Authorize minimal additional edits to MCP/client.py (carry the real credential owner to HTTP), Plugins/coordinator.py and Plugins/recovery.py (concrete LocalMCPControlService validation of complete connection mappings before authenticated reconstruction). LocalMCPControlService.capture_connection_mapping binds the actual saved profile, retained MCP component definition/configuration and credential metadata; unsupported mapping kinds remain refused. Seed recovery tests through the existing guarded authenticated store/transaction fixture and invoke public coordinator recovery, without claiming a public mapping-edit workflow or grants. M4 must supply actual reviewed publication/registration/launch. Read m3-recovery-boundary-ruling.md. Preserve locked protected keyring record/tombstone revision semantics, reread current state at resolution and recheck async-refresh storage revision before writing; no token-derived identity or stale writer may restore prior authority. Relevant read-only neighbors include Tests/Plugins/test_recovery.py, test_coordinator.py, test_authority.py and test_authority_store.py plus MCP profile/store/lifecycle tests; record exact final covering selection and any additional edit path first.

9. Explicit credential-reference storage extends local MCP JSON schema2 to3. Authorize Tests/MCP/test_local_store.py to update existing migration expectations from2 to3 and future-schema refusal from3 to4, preserving their behavioral assertions. Old profiles migrate with no credential binding; binding fields supplied in an older schema cannot manufacture authority and must fail explicitly. Add new auth/migration/malformed-reference controls in the task-owned test_credential_bindings.py. The exact existing test path already has a BASE snapshot. Report further fixture/schema edits before changing them; do not weaken successful/reopen/unsupported-field controls to absorb the migration.

10. Apply R63 before finalizing credential creation: the host owner mints immutable UUID reference IDs for new records. A caller-supplied missing reference cannot be upserted at generation1; replacement/refresh/revocation must reread an existing protected record and refuse absence. Keep retained-record authority generations monotonic within the existing signed64-bit schema, tombstones persistent and exhaustion fail-closed; do not widen authority.py or hash/randomize token authority. Preserve a behavioral RED for deleted-record stale-binding resurrection and a successful independent new-reference creation control. Record the deliberate creation API change and record-loss rebind/review cost in the existing ADR162/spec/MCP plan and final report. No new source paths are authorized.

11. Apply R64 before final transport qualification: resolve potentially blocking protected-store work off the shared MCP event loop via one retained daemon-worker Future per credential service. Contending async callers wait only within their existing deadlines, then claim the slot for fresh work; no queued worker operations, shared secret-result cache or lock held over I/O. A cancelled waiter cannot issue HTTP later; actual worker completion alone releases capacity. Async authorize/renew storage segments use the same owner, with host callbacks outside storage locks and revision recheck. Audit actual connect/request/resume/notification/DELETE entries for blocking credential metadata calls. Add a blocked-backend RED, responsive unrelated anonymous peer, cancelled/deadline no-late-dispatch, ordinary short-contention and post-release fresh-lookup controls in existing task-owned test files. Document retained-thread/timeouts limitation in ADR162/spec/MCP plan and report; no new paths authorized.

12. Freeze and run the exact ten-file final covering set: Tests/MCP/test_credential_bindings.py, Tests/MCP/test_transport_auth.py, Tests/Plugins/test_credential_authority.py, Tests/MCP/test_local_store.py, Tests/MCP/test_streamable_http.py, Tests/MCP/test_protocol_profiles.py, Tests/MCP/test_control_plane_lifecycle.py, Tests/Plugins/test_recovery.py, Tests/Plugins/test_coordinator.py, Tests/Plugins/test_authority.py. All existing files have exact BASE snapshots. The config addition is a lazy owner factory only, without TOML secret-classification/encryption changes; test namespace composition and insecure-backend refusal in the task-owned credential tests rather than running unaffected encryption suites. Distinguish memory-backend semantics, real portalocker with fake keyring API across service/backend instances, and unqualified real OS keychain interoperability. Keep all source/test/doc/runner bytes fixed through covering and syntax/Ruff/Black baseline-parity/whitespace checks; preserve every earlier failing or pre-correction run as historical.

13. Fix independent review I1 before closeout. Repair the newly introduced schema2-to3 migration in MCP/local_store.py so malformed source entries or filtered authoritative records cannot be rewritten as sanitized/default state. Validate raw source shapes and preservation across serialized profile/governance/approval/activity and relevant dictionary collections before migration writes; preserve the explicit existing reserved-ID quarantine and legitimate legacy migrations. Add behavioral refusal plus exact-original-byte preservation controls alongside valid migration/reopen controls in Tests/MCP/test_credential_bindings.py and, if needed, Tests/MCP/test_local_store.py. No other runtime/test edits without a concrete scope request. Preserve the reviewer probe as RED evidence, establish committed regression RED, then freeze and run exactly Tests/MCP/test_credential_bindings.py, Tests/MCP/test_local_store.py, Tests/MCP/test_local_store_runtime_state.py, Tests/MCP/test_protocol_profiles.py, Tests/MCP/test_control_plane_lifecycle.py and Tests/Plugins/test_credential_authority.py. Compare static output to exact Fix1 BASE a9ea6f0ba0192ad37e3579bd4040fa537694ef2a; retain initial evidence, append the report and require a fresh scoped review. This is the existing malformed-storage preservation requirement, no new ADR or ruling.
<!-- SECTION:PLAN:END -->
