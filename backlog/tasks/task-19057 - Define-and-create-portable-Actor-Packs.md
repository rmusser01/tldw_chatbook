---
id: TASK-19057
title: Define and create portable Actor Packs
status: Done
assignee: []
created_date: '2026-08-20 18:13'
updated_date: '2026-08-22 21:19'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a secure, deterministic one-actor portable envelope and let users create pack-ready local Characters or Personas with a required portrait and stable portable identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `tldw.actor-pack/v1` defines exactly one local Character or Persona, required canonical actor JSON and portrait, optional typed visual sections, license/provenance declarations, required features, and no local IDs or external references.
- [x] #2 Internal paths, canonical JSON, per-file SHA-256/size inventory, non-self-referential top digest, deterministic ZIP metadata, and all actor/manifest/portrait limits match the approved spec.
- [x] #3 The profile-local registry is keyed by `(actor_kind, local_actor_id)`, stores a globally unique canonical lowercase RFC 4122 UUIDv4 as portable identity independent of names, content, and local IDs, enforces UUID uniqueness across both actor kinds without claiming cross-install coordination, survives soft deletion/restoration, and records copy provenance without reusing the source UUID.
- [x] #4 New Actor Pack uses the canonical local Character/Persona editors, admits only one operation at a time and rejects duplicate submits, requires a portrait, and fences source, editor, and portrait authority. Cancellation or declined navigation during portrait or commit work signals and drains owned work and leaves no actor, registry row, intent, or staged portrait; success creates only the actor plus portable identity, without writing an archive or requiring visual sections.
- [x] #5 Server-backed Personas cannot receive portable registry rows and expose Save Local Copy first.
- [x] #6 Persona actor/registry changes use a bounded profile-private intent durably written before the atomic Persona JSON replace; one SQLite transaction atomically writes registry/visual rows and committed intent status. Startup recovery runs before affected Persona or Actor Pack surfaces and is idempotent: prepared+old JSON+old SQLite cleans up; prepared+new JSON+old SQLite compensates to the old record or removes a newly created record; committed+new JSON+new SQLite retains the new record and cleans up. Old JSON+new SQLite, prepared+new SQLite, committed contradictory states, or any other unexpected digest/revision are quarantined without a destructive guess and require explicit recovery. Ordinary errors compensate; intents are never logged or exported; Character changes remain one SQLite transaction.
- [x] #7 Unknown required features, malformed/colliding paths, invalid actor kinds/payloads/portraits, concurrent registry assignment or UUID collision, and stale profile, editor, or portrait authority fail closed with no partial actor, registry row, intent, staged portrait, or other residue.
- [x] #8 This task is scoped to Actor Pack format, schema, canonicalization, digest, and pure-validator contracts plus actor creation and the Persona cross-store coordinator; export writer, import reader, extraction, staging, review, and activation implementation are absent and reserved for TASK-19058 and TASK-19059. Verification includes born-RED→GREEN tests, mutation proof for authority, safety, cancellation, and recovery guards, assigned-worktree provenance, real SQLite migration and crash-recovery tests in an isolated profile, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: ADR-074 already defines the Actor Pack schema, portable identity registry, local-only authority, and bounded Persona JSON/SQLite recovery protocol implemented by this task.

Executable plan: Docs/superpowers/plans/2026-08-22-task-19057-portable-actor-pack-foundation.md

1. Freeze the pure Actor Pack schema, canonical path/JSON, inventory, digest, portrait, actor payload, and typed optional-section contracts.
2. Add the v45 portable UUID registry and bounded Persona mutation intents with real migration and repository coverage.
3. Implement the purpose-built Persona JSON/SQLite coordinator, compensation, quarantine, and startup recovery matrix.
4. Add one transactionally atomic Character creation path and one coordinator-backed Persona creation path, both portrait- and authority-fenced.
5. Reuse the canonical Workbench editors for a distinct New Actor Pack action without adding ZIP export/import or visual-authoring scope.
6. Prove cancellation drain, stale authority, privacy, architecture, migration packaging, and isolated-profile behavior before closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the portable Actor Pack foundation governed by ADR-074: strict tldw.actor-pack/v1 contracts and deterministic metadata, v45 portable identity and bounded recovery persistence, atomic local Character creation, compensated Persona JSON/SQLite coordination, startup recovery, and a distinct canonical Workbench New Actor Pack flow. No ZIP writer/reader, extraction, staging, review, activation, visual-runtime merge, dependency, or second editor was added.

TDD and mutation evidence covered path/inventory/digest/portrait bounds, UUID and transaction ownership, every Persona recovery state, source/editor/portrait authority, duplicate admission, cancellation/drain, stale reconciliation, packaging inventory, and the architecture boundary. Final isolated evidence passed 134 Actor Pack/service/Workbench/ownership/architecture tests, 317 complete ChaChaNotes plus migration-package tests, 23 focused Workbench/non-regression tests, 81 branch-owned diagnostic/privacy tests, 5 Actor Pack architecture guards, and the assigned-worktree provenance test. The Impeccable detector ran once after the final visible change and returned an empty finding list. Ruff and compilation passed all touched Python files; 22 files are format-clean, while five exact files reproduce the same formatter baseline at pinned base 0da426e1e. Diff checks, licence/dependency review, Backlog governance, diagnostic inventory (five reviewed fixed-category calls), private SQLite ownership, and migration packaging passed.

A broad privacy run also reported six Client_Media_DB_v2 exception-chain failures; the exact pinned base reproduced the same six failures with 83 passes, and this task does not modify that owner. No new lesson was added because the diagnostic and profile-isolation incidents are already covered by the existing testing-evidence and live-verification lessons. Local specification/correctness and separate Ponytail reviews found no remaining task-owned issue. ADR required: no; ADR-074 remains authoritative.
<!-- SECTION:NOTES:END -->
