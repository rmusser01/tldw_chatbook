# ADR-356: Native phase orchestration uses runtime ownership

- Date: 2026-11-07
- Status: Accepted; implemented and qualified by TASK-39147
- Task: [TASK-39147](../tasks/task-39147%20-%20Rebuild-PR3411-native-phase-orchestration-on-existing-runtime.md)
- Supersedes: ADR-285's implementation-identity/caller-attestation requirements and recovery rules that retain activity claims solely for uncertain historical effects. ADR-285's approved product behavior remains required.
- Design: [replacement design](../../Docs/superpowers/specs/2026-11-07-native-phase-orchestration-rebuild-design.md)
- Plan: [inline implementation plan](../../Docs/superpowers/plans/2026-11-07-native-phase-orchestration-rebuild.md)

## Context

PR3411 implements native coding-agent phases in one selected writable local
Workspace worktree. Its implementation accumulated function/code/default/closure
identity checks and caller-frame inspection across launch, tool use, settlement
and ordinary chat. These checks couple authority to Python implementation details.
They do not provide a security boundary against arbitrary in-process code.
Closed recovery can also retain activity protection indefinitely because uncertain
prior effects and live execution share the same active predicate.

The user authorized rebuilding this feature on the existing agent runtime,
selectively preserving useful data/UI and independent fixes. This is a replacement
for the PR's orchestration, not for the application.

## Decision

1. Keep one app-owned phase coordinator for ordering, explicit approvals,
   checkpoint transitions and result collection. It delegates model/tool execution
   to AgentService and FleetCoordinator. It owns no alternative executor.
2. Put typed child acceptance, provisional-launch cancellation, completion and
   resource-drain results at the native runtime/service boundary. A completed
   model or durable terminal row does not imply physical cleanup. Existing
   ExecutionOwner, resource leases, permission store and budget ledger remain
   authoritative for their actual resources.
3. Compose task narrowing through injected LoopDeps restrictions and the actual
   service/provider invocation boundary. Recheck current scope after hook argument
   mutation and permission waits. Preserve root, exclusion, revision, tool, budget
   and atomic file preconditions. A permission Allow never widens task scope.
4. Orchestration must not inspect function objects, code/defaults/closures,
   descriptors, caller stacks/locals or coroutine implementation to establish
   authority. A runtime-owned registration and current state identify ownership;
   persisted IDs, copied records and user/model input cannot recreate it.
   Planning freezes the admitted system-prompt text. Later session-settings saves
   affect future turns, not the native return path. The actual required prompt/
   seed remains checked at the provider boundary; current session-object identity
   is not evidence of resource return.
5. Ordinary chat participates only in shared-file activity coordination where
   required. Disabled or inactive workflows add no workflow self-attestation.
6. Separate worker liveness from uncertain historical file effects. Startup audits
   native owners; it does not resurrect workers from workflow rows. Live or
   unproven surviving workers retain their real resource claims. Historical
   uncertainty requires visible reconciliation and fresh evidence before resume,
   but does not create a permanent machine-wide or unrelated-scratch barrier.
   Explicit reconciliation closes reviewed uncertainty and releases only its own
   obsolete recovery protection; it never releases a live worker's claim.
   Required accounting/result publication can fail after the provider, tools and
   native scopes have positively returned. Such history uncertainty must not keep
   a physically returned native owner admitted. Durable native admission and known
   charges already exist before dispatch. Hand their unresolved accounting tail
   to a resource-return receipt and captured folder protection, then retire actual
   resources once. Recovery recording is joined but retryable; a second write or
   delivery failure cannot resurrect a returned owner. Explicit review retries
   recording with the displayed control revision, preserves committed completion
   and unknown charges, and never infers return from a terminal row.
   Shutdown may discard a proven nonlive coordinator entry without reviewing or
   modifying its durable history; the normal startup audit recovers that history.
   A failed or unproven native scope
   still retains its real owner; a terminal row alone never proves return.
   Clock/SQL refusals and errors delivering already-committed completion use this
   same actual-return path. Committed plans, completion and known charges survive;
   an incomplete settlement remains uncertain rather than inventing completion.
   A saved plan's directory-chain identity must match before projecting its
   historical fence. A replacement directory or symlink target inherits no fence.
   Planning-only history has no published root ownership or mutating task effects;
   its accounting stays reviewable without claiming a folder by pathname at
   startup. Installed overlays revalidate captured directory identity at admission
   and wave readiness. Retire only obsolete historical scopes, preserve any other
   current scopes and all real native claims, and notify waiters outside the lock.
   An unreadable directory chain is not proof of replacement. Preserve established
   historical scopes on transient I/O or incomplete metadata; retire them only on
   positive changed metadata or confirmed absence.
7. Preserve bounded history and accounting. Result/window removal releases actual
   payload reservations once and keeps useful history/tombstones. Do not add full
   private-history erasure. Legacy histories remain readable; old approvals cannot
   launch replacement execution without explicit fresh validation.
   Record explicit recovery review in one fixed-size revision field on the
   workflow row (additive AgentRuns v30), separate from payload quota. An immutable
   decision payload was considered, but a full quota would prevent the review
   required to remove results. The field records review, never physical drain;
   unknown charges and native retirement rows remain unchanged. Existing v25–29
   archives stay readable and upgrade transactionally without inferred review.
   A legacy `closed` status with the default zero marker is not explicit review;
   unsettled history keeps its scoped fence until the supported review completes.
   A delivered-again closing review acknowledges already-reviewed closed/removed
   history without another status/revision change or reopening removed payloads.
   Current revision and actual native-liveness checks still apply, including when
   accounting is known.
8. Implement in the existing checkout with scoped reviewable changes, targeted
   tests and one end-to-end native journey. No hidden source export, new worktree,
   automatic commit/reset/stash/push or publication is part of this task.

The Console runtime registers its native Library construction resource when it
is created. Bounded ordinary admission and construction require that same actual
resource and owning runtime/app, plus captured service/effect declarations. An
unregistered callable, including a partial with the app as its argument, requires
opaque protection before construction. Transparent implementation wrappers do not
change resource ownership; callback/code identity is not an authority check.

## Alternatives

- Continuing to patch the attestation layers preserves the coupling the user
  explicitly rejected and makes routine wrappers/refactors affect availability.
- Deleting every PR file loses useful validation, UI, history and real cleanup
  fixes. Selective replacement preserves behavior while removing the mechanism.
- A new workflow engine or external workflow product duplicates existing native
  execution and is outside the approved scope.

## Consequences

Functional tests must prove permission, budget, revision, concurrency, failure and
cleanup behavior; tests requiring counts of implementation-identity checks are not
replacement requirements. Physical cleanup remains a real runtime obligation.
The worktree is shared, not an OS sandbox; external writers can invalidate inputs.
The rebuild remains disabled by default until its acceptance evidence is complete.
This decision does not certify existing salvage candidates or authorize merging
or publishing a replacement PR.


## Current-dev integration allocation (PR3480, 2026-11-08)

The authorized rebase targets dev36a21dc4d270940e3576b49487a71b37c78896f8.
That shipped AgentRuns v25 belongs to review-boundary claims and the nullable
opaque wake-attempt handoff. The older standalone workflow branch reused v25 for
workflow tables and advanced through v30; it is not the shipped dev25 lineage.
Preserve the exact dev25 DDL/data, migration and recovery catalogs. Allocate the
workflow additions after that boundary (v26 onward, ending at v31 for the current
six-step feature chain), without rewriting an installed version or dropping the
review-boundary fields/claims. Current constructors, migration order and exact
recovery catalogs must agree; equal numeric labels are not compatibility proof.

Use real SQLite upgrades from the actual dev25 layout with populated native
review-boundary/wake data and fresh workflow operations, plus archive/catalog
controls and interrupted/failed migration checks. Standalone workflow25–30
qualification remains tied to its original source and does not certify shipped
current-dev25 recovery. Existing reviewed native resource, permission, revision
and history fences remain mandatory. This amends the existing additive storage
allocation decision; it introduces no new executor, service or trust boundary.
