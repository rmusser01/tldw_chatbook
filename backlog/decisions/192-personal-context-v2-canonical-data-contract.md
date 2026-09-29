---
title: Personal Context V2 canonical data contract
status: Accepted
date: 2026-09-26
---

# ADR-192: Specify an inactive V2 canonical Personal Context contract

Status: Accepted design direction after explicit user approval, 2026-09-26. No native schema/runtime rollout approved.
Task: TASK-25907.18
Design: [Concrete contract](../../Docs/superpowers/specs/2026-09-26-personal-context-v2-canonical-data-contract-design.md)

## Context

[ADR-201](201-versioned-profile-evidence-and-temporal-claims.md) accepts a
distinct V2 model, but field names, closed shapes and digest projections still
need a concrete contract. The [readiness audit](../docs/personal-context-v2-admission-readiness-audit.md)
found V1 canonical dispatch throughout the native owners. The existing complete
18-field binding is data-only and already published; widening it would alter
its digest and compatibility. [ADR-202](202-dependency-aware-personal-context-forgetting.md),
[ADR-203](203-personal-context-provider-disclosure-authority.md) and
[ADR-191](191-foreground-personal-context-source-inspection-authority.md)
remain independent lifecycle, disclosure and source-authority obligations.

## Decision

Freeze distinct manifest/record/proposal V2 wire shapes and a required V2
semantic dialect. Keep V1 scopes and typed payloads unchanged inside V2.
Compose the existing binding component without new fields, source forms or
excerpts in this first contract. Publish no runtime accept/migrate path as
part of an eventual data-only shared-core implementation.

Domain-separate a claim digest including profile, record and scope identity,
kind, payload, basis, validity and relations. Support additionally binds the
complete existing binding digest; approval binds claim digest and exact record
version. Serialized attribution is a peer assertion, not a capability or
proof of local user intent. Model policy defaults to deny and contains ceilings
only. Deleted records and resolved proposals retain no evidence, attribution
or payload; sanitized privacy successors have no transferable approval.

Manifest semantics are profile-wide, including effects on V1 records. Declared
requirements are not consumer acknowledgements. Activation still requires all
native consumers, old-grant retirement, metadata controls and companion-server
conformance under [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md)
and ADR-201. No new grant, source factory, provider enrollment or forgetting
receipt protocol is created here. Those protocols must be concretized and
qualified before this data can become usable.

## Alternatives

1. Distinct data-only V2 aggregates with existing binding composition: recommended;
   gives one testable contract without enabling unqualified consumers.
2. Optional V2 fields in V1 or a parallel evidence sidecar: rejected for this
   flow by ADR-201; silently changes bytes or splits canonical authority.
3. Ship local V2 storage and resolver first: deferred; a device-only flag does
   not qualify binding metadata, restore, old clients or disclosure gates.

## Consequences and approval boundary

The first implementation unit, after written review and planning, would be
shared-core data models, structural/semantic validation and fixed fixtures.
Native storage, migration, retirement receipts, activation and source inspection
remain separate reviewed units. Excerpts, Notes and captured representations
need explicitly versioned future components, not silent widening of binding v1.
The user explicitly approved this written contract on 2026-09-26; acceptance
approves design direction only.
No production/schema/fixture bytes change in this task. Number 192 was free
across available local object history and 82 worktrees at drafting; recheck at
integration, since no remote refresh or number reservation is implied.
