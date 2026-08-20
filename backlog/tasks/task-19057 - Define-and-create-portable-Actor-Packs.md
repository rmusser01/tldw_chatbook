---
id: TASK-19057
title: Define and create portable Actor Packs
status: To Do
assignee: []
created_date: '2026-08-20 18:13'
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
- [ ] #1 `tldw.actor-pack/v1` defines exactly one local Character or Persona, required canonical actor JSON and portrait, optional typed visual sections, license/provenance declarations, required features, and no local IDs or external references.
- [ ] #2 Internal paths, canonical JSON, per-file SHA-256/size inventory, non-self-referential top digest, deterministic ZIP metadata, and all actor/manifest/portrait limits match the approved spec.
- [ ] #3 The profile-local registry is keyed by `(actor_kind, local_actor_id)`, stores a globally unique canonical lowercase RFC 4122 UUIDv4 as portable identity independent of names, content, and local IDs, enforces UUID uniqueness across both actor kinds without claiming cross-install coordination, survives soft deletion/restoration, and records copy provenance without reusing the source UUID.
- [ ] #4 New Actor Pack uses the canonical local Character/Persona editors, admits only one operation at a time and rejects duplicate submits, requires a portrait, and fences source, editor, and portrait authority. Cancellation or declined navigation during portrait or commit work signals and drains owned work and leaves no actor, registry row, intent, or staged portrait; success creates only the actor plus portable identity, without writing an archive or requiring visual sections.
- [ ] #5 Server-backed Personas cannot receive portable registry rows and expose Save Local Copy first.
- [ ] #6 Persona actor/registry changes use a bounded profile-private intent durably written before the atomic Persona JSON replace; one SQLite transaction atomically writes registry/visual rows and committed intent status. Startup recovery runs before affected Persona or Actor Pack surfaces and is idempotent: prepared+old JSON+old SQLite cleans up; prepared+new JSON+old SQLite compensates to the old record or removes a newly created record; committed+new JSON+new SQLite retains the new record and cleans up. Old JSON+new SQLite, prepared+new SQLite, committed contradictory states, or any other unexpected digest/revision are quarantined without a destructive guess and require explicit recovery. Ordinary errors compensate; intents are never logged or exported; Character changes remain one SQLite transaction.
- [ ] #7 Unknown required features, malformed/colliding paths, invalid actor kinds/payloads/portraits, concurrent registry assignment or UUID collision, and stale profile, editor, or portrait authority fail closed with no partial actor, registry row, intent, staged portrait, or other residue.
- [ ] #8 This task is scoped to Actor Pack format, schema, canonicalization, digest, and pure-validator contracts plus actor creation and the Persona cross-store coordinator; export writer, import reader, extraction, staging, review, and activation implementation are absent and reserved for TASK-19058 and TASK-19059. Verification includes born-RED→GREEN tests, mutation proof for authority, safety, cancellation, and recovery guards, assigned-worktree provenance, real SQLite migration and crash-recovery tests in an isolated profile, and scoped Ruff, format, compile, diff, diagnostic, privacy, architecture, and governance checks.
<!-- AC:END -->
