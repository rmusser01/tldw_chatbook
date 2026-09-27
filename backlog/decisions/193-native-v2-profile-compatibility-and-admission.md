---
title: Native V2 profile compatibility and admission
status: Accepted
date: 2026-09-27
---

# ADR-193: Gate native V2 profiles on current compatibility and atomic admission

Status: Accepted design direction after explicit user approval, 2026-09-27. No native schema/runtime rollout approved.
Task: TASK-25907.21
Design: [Native compatibility and admission](../../Docs/superpowers/specs/2026-09-27-personal-context-native-v2-admission-design.md)
Extends: [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md), [ADR-185](185-versioned-profile-evidence-and-temporal-claims.md), [ADR-192](192-personal-context-v2-canonical-data-contract.md)
Prerequisites: [ADR-186](186-dependency-aware-personal-context-forgetting.md), [ADR-187](187-personal-context-provider-disclosure-authority.md), [ADR-191](191-foreground-personal-context-source-inspection-authority.md)

## Context

TASK-25907.20 completed inactive V2 data validation, dialect and fixed fixtures.
Native owners still use V1. Strict context/export snapshots do not protect
individual Settings getters, mutation, interview, recovery and Sync seams by
themselves. A V2 exception or privacy restriction can affect a V1 record, so
omitting an unsupported row cannot establish profile compatibility.

## Decision

Use one profile-wide native compatibility barrier at both service and repository
boundaries, before ordinary reads and within commits. Compile an explicit route
registry at native composition; require current native owner qualification or
verified disablement/grant retirement for every active consumer. Bind encrypted
peer-local qualification controls to exact manifest/object/semantic requirements,
build/fixture identities, registry/cohort, authority, purge and retirement
revisions. Serialized requirements, imported receipts, installed libraries and
offline status are not qualifications or authority.

Separate compatibility from exact record/operation admission and from automatic
use. The authorized service creates non-exportable process-local admission
stamps for exact versions, relation heads/scopes, current authority and foreground
review. The repository checks them under its existing transaction and the
qualified retirement/publication coordinator. Sources/models/network callbacks
stay outside SQLite. Cross-owner races need owner fencing, not a stale pre-read.
Unchanged requirement/owner revisions permit atomic qualification rebind to a
manifest successor; changed requirements need fresh checks and prepared stamps
never auto-refresh. Existing immutable historical edges retain their original target; they confer
no new authority and cannot restore retired material. Native controls belong to
the existing encrypted profile repository, never a fourth fact store or portable
grant. Physical migration/control schemas require their own qualified unit.

A blocked V2 profile blocks all ordinary consumers, including V1 records within
it. Owner-only generic maintenance status is separate from content inspection.
Legacy V1 profiles continue existing behavior; under a V2 manifest unreviewed
V1 facts remain unavailable for automatic use/disclosure. Migration preserves
bytes/IDs and does not invent validity, support or approval. Startup, restore,
outboxes and reconnect apply current fences before replay; old backups cannot
restore native grants or lower retirement controls.

Stage implementation: first an integrated native barrier/codec with production
V2 permit paths closed; then qualified retirement/publication, restrictive
disclosure, atomic admission, server/cohort conformance, and reviewed cutover/
foreground source inspection. All accepted privacy, all-consumer and companion
server gates remain. No local-only exception, periodic job or new dependency is
selected. Mocked owner/server receipts are test inputs, never rollout evidence.

## Alternatives

- Per-object quarantine/unknown-row omission: rejected because V2 semantics can
  change the meaning or authority of a V1 record.
- Device-only storage/source inspection before lifecycle qualification: deferred
  because Sync exclusion is not metadata retirement or model consent.
- A second evidence/qualification fact database: rejected; native control state
  belongs to the existing profile owner and contains no duplicated user facts.

## Consequences and approval boundary

This makes cutover visibly unavailable while any active route or authority is
unqualified. New routes must register or deny governed profile access. Consumer
grants and peers need verified retirement; offline copies remain honestly
outstanding. Existing native storage schema 8/AAD marker 1, canonical V1 dispatch,
server_trusted_v1 custody, scope/payload/binding/meaning contracts and current
permissions remain unchanged in this design-only task.

The user explicitly approved the written design on 2026-09-27; acceptance
approves this design direction only. A native Unit A
implementation plan follows written review; it does not authorize V2 activation,
source access, provider enrollment, server writes or destruction. Number 193 was
free across available local object history and 76 worktrees at drafting; recheck
at integration. No remote refresh or global number reservation is implied.
