# Next Tranche Parallel Execution Design

Date: 2026-04-28

Status: Draft for review

Related docs:

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-connection-auth-foundation-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`

## Purpose

This design defines the next implementation tranche after the current server-parity planning and contract work.

The goal is to execute the next highest-value work with maximum safe parallelism while keeping the real dependency chain explicit:

1. connection/auth hardening must land first
2. capability/auth normalization and remaining provider migration can then fan out in parallel

This design deliberately stops at the next executable tranche. It does not pull in later remote event transport, sync expansion, or deferred domain rows.

## Goals

- Land connection/auth hardening as the next merge-gate tranche.
- Enable parallel follow-on work immediately after the hardening tranche lands.
- Keep one active-server authority and one provider-client construction seam.
- Preserve the migration-audit rules that now require semantic matching and disjoint test ownership.
- Minimize `app.py` contention and shared-test merge conflicts.

## Non-Goals

- Do not include later event transport work.
- Do not include sync dry-run expansion beyond what is already defined elsewhere.
- Do not include workflow implementation.
- Do not redesign UI/auth/server selection screens.
- Do not introduce any second active-server, capability, or credential authority.

## Recommended Approach

Use **Foundation Plus Immediate Dependents**.

This is preferred over a connection/auth-only plan because it preserves the critical-path correctness of the foundation work while still giving multiple contributors productive parallel follow-on lanes immediately afterward.

Rejected alternatives:

- **Connection/Auth only:** safest but underuses contributors and forces another planning round immediately after landing.
- **Mini-program planning for all remaining parity work:** higher forward visibility, but too broad for the next execution cycle and more likely to create vague ownership and hidden dependencies.

## Workstream Decomposition

### Lane A: Connection/Auth Hardening

This is the merge-gate lane.

Responsibilities:

- global sign-out across all stored server credentials, including orphaned entries
- active-profile-only legacy credential import
- provider cache invalidation on server switch, credential mutation, and sign-out
- typed secure-store and missing-credential failures
- strict target-store versus credential-store separation enforcement

This lane defines the final credential and invalidation behavior that later lanes consume.

### Lane B: Capability/Auth Normalization

This lane depends on the reviewed or landed `Lane A` behavior.

Responsibilities:

- capability snapshot invalidation on server switch and sign-out
- normalized auth-state transitions
- consistent unavailable, auth-required, session-invalid, and unreachable result mapping
- target-store last-known status alignment with runtime-policy state

This lane must remain derived-only. It must not become a second authority for selected server or auth state. Status alignment here means normalizing and persisting the projection between existing runtime-policy state and target-store last-known status metadata, not creating a new source of truth.

### Lane C: Remaining Provider Migration

This lane also depends on the reviewed or landed `Lane A` behavior.

Responsibilities:

- remaining compatibility-factory cleanup
- audited migration of holdout services
- migration-audit updates after each sub-batch, routed through one `Lane C` migration-audit owner
- enforcement that no new direct `tldw_api` config builders appear

This lane stays domain-sliced internally and uses the existing provider migration ownership model rather than inventing a new integration pattern.

## Merge Gates

### Gate 1

`Lane A` must land first.

Reason:

- it stabilizes credential lifecycle rules
- it stabilizes provider invalidation rules
- it defines the behavior that `Lane B` and `Lane C` must consume instead of reinterpreting

### Gate 2

After `Lane A` is reviewed or merged, `Lane B` and `Lane C` may proceed in parallel.

Rules:

- `Lane B` must not create a second status/capability authority
- `Lane C` must not weaken or bypass the provider and credential rules established by `Lane A`

## Ownership Boundaries

### Lane A Ownership

Owned files:

- `tldw_chatbook/runtime_policy/server_context.py`
- `tldw_chatbook/runtime_policy/server_credentials.py`
- any new credential-index or secure-store helper files under `tldw_chatbook/runtime_policy/`
- the exact `app.py` methods and direct caller blocks that initialize or tear down the provider/credential seam:
  - `_wire_server_context_provider()`
  - `_close_server_context_provider_cached_client()`
  - any logout/server-switch handler blocks that directly clear `server_credential_store`, call `server_context_provider` credential-clear methods, or invalidate provider-bound clients

App-wiring rule:

- `Lane A` is the only tranche lane allowed to edit those reserved `app.py` methods/blocks.
- `Lane B` and `Lane C` must treat all other `app.py` changes as out of scope for this tranche unless a separate integration-owner follow-up is declared.

Allowed tests:

- new lane-specific credential-store tests
- new lane-specific provider invalidation tests
- new lane-specific legacy-import tests

Forbidden edits:

- capability snapshot logic
- provider-migrated domain service modules
- migration-audit semantic rules except where required to register new baseline seams

Acceptance criteria:

- global sign-out clears all stored server credentials, including orphaned entries
- active-profile-only legacy import is enforced
- server switching invalidates provider clients and prevents old-profile credential reuse
- no secrets leak into JSON, config, log, export, or unsupported-report surfaces
- failures are typed and fail closed for server-authenticated actions

### Lane B Ownership

Owned files:

- `tldw_chatbook/runtime_policy/server_capabilities.py`
- auth-state classification helpers used by capability refresh
- `tldw_chatbook/runtime_policy/types.py` only for server reachability/auth-state type changes required by normalization
- `tldw_chatbook/runtime_policy/source_state.py` only for runtime-policy status normalization and persistence behavior required by freshness/alignment rules
- `tldw_chatbook/MCP/server_target_store.py` only for `update_target_status()` and related last-known status metadata handling needed to keep target-store projections aligned with runtime-policy state

Allowed tests:

- lane-specific capability invalidation tests
- lane-specific auth-state mapping tests
- lane-specific runtime-policy status tests

Forbidden edits:

- credential-store internals
- provider cache key or invalidation internals
- provider-migrated domain service modules
- reserved `Lane A` `app.py` methods/blocks

Acceptance criteria:

- capability snapshots invalidate on server switch and global sign-out
- auth-required, session-invalid, and unreachable states are normalized consistently
- target-store last-known status and runtime-policy status move together without authority drift

### Lane C Ownership

Owned files:

- only the service modules assigned to that migration sub-batch
- additive lane-specific migration tests

Shared-file rule:

- `Docs/Development/server-client-provider-migration-audit.md` is owned by one `Lane C` migration-audit integration owner for this tranche.
- Domain migration sub-batches must not edit the shared audit file directly.
- Each sub-batch must instead hand off a pending audit delta describing:
  - migrated modules
  - removed compatibility factories
  - new provider-backed seams
  - any remaining intentional holdouts
- The migration-audit integration owner is responsible for applying those deltas to the shared audit file after reviewing the sub-batch changes.

Allowed tests:

- sub-batch-owned service tests
- additive migration-guard tests

Forbidden edits:

- credential-store internals
- active-server authority seams
- capability snapshot ownership seams
- shared test files owned by another sub-batch

Acceptance criteria:

- no new raw `tldw_api` config builders are introduced
- all migrated services resolve clients through `RuntimeServerContextProvider`
- migration-audit updates continue to use stable semantic matching only
- compatibility holdouts remain explicitly listed
- shared audit updates land only through the `Lane C` migration-audit integration owner

## Test Ownership Rules

To avoid repeating the shared-test bottleneck:

- existing shared test files are owned by the lane that owns the production seam they validate
- other lanes add new lane-specific test files by default instead of editing shared owned test files
- cross-lane edits to an owned shared test file must route through that lane's owner

Concrete ownership:

- `Lane A` owns credential and provider invalidation test files
- `Lane B` owns capability and auth-state test files
- `Lane C` sub-batches own only their service-specific migration tests

If a provider migration needs a capability assertion, it should add a lane-local assertion file or coordinate through `Lane B` rather than directly editing a `Lane B`-owned shared test file.

## Sequencing

### Start Conditions

- `Lane A` starts immediately.
- `Lane B` may prepare docs, test scaffolding, and fixtures in parallel, but should not land behavior that depends on final invalidation semantics until `Lane A` is reviewed or merged.
- `Lane C` may audit holdouts and prepare sub-batch diffs in parallel, but migration merges should wait for `Lane A` to freeze provider and credential lifecycle behavior.

### Recommended Execution Order

1. Land `Lane A`.
2. Once `Lane A` is reviewed, branch `Lane B` and `Lane C` from that reviewed state.
3. Run `Lane B` and `Lane C` in parallel.
4. Within `Lane C`, preserve the existing provider sub-batch split and integrate through one migration-audit owner.
5. After `Lane B` and `Lane C` land, reassess whether the next tranche should be remote event transport or sync dry-run expansion.

## Risks

### Lane A Risks

- per-profile clearing works, but true global orphaned-secret cleanup is missed
- invalidation is partial and old-profile credentials remain reusable indirectly

Mitigations:

- require explicit orphaned-secret cleanup tests
- require explicit server-switch credential-isolation tests

### Lane B Risks

- capability normalization drifts into a second authority instead of staying a projection of runtime-policy plus provider outcomes

Mitigations:

- require lane-specific tests proving capability state is derived-only
- reserve active-server authority seams to `Lane A` and baseline runtime-policy owners only

### Lane C Risks

- small migration changes reintroduce direct config builders
- sub-batches edit shared tests opportunistically and recreate merge contention

Mitigations:

- fail CI on newly introduced unlisted builders
- enforce lane-specific additive tests for migration coverage

### Integration Risks

- `app.py` becomes the hidden contention point again

Mitigations:

- reserve the exact startup/logout/server-switch blocks up front
- route all shared invalidation wiring through the `Lane A` owner

## Definition Of Done

This tranche is complete when:

- connection/auth hardening is landed
- capability/auth normalization is landed
- the remaining provider migration batches in scope are landed
- migration-audit enforcement remains semantic and green
- no second server authority exists
- no credential leakage path exists in persisted metadata or logs

## Follow-On Boundary

This design intentionally stops before:

- remote event transport implementation
- sync dry-run/read-only mirror expansion
- later deferred parity rows outside this immediate dependency chain

Those should be planned only after this tranche lands and the next critical path is reassessed.
