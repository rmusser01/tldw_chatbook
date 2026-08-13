---
id: TASK-208
title: Guard active ingest submissions against duplicate sources
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-12 17:34'
updated_date: '2026-08-13 15:13'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent an accidental second submission of the same source while a matching
Library ingest job is still queued or running. The guard must keep intentional
re-ingestion available through an explicit two-press Start confirmation and
must remain separate from historical content deduplication and overwrite
behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A source matching a QUEUED, PARSING, or WRITING job for the same Local or Server backend is not submitted on the first Start press; terminal jobs never block re-ingestion.
- [ ] #2 Equivalent local path spellings and conservative HTTP(S) URL spellings share a stable lexical admission key without filesystem, network, database, or content-hash work.
- [ ] #3 Local and Server jobs use separate admission scopes, so an active Local import does not block submission of the same source to Server and vice versa.
- [ ] #4 A blocked first press preserves the complete form and displays a compact, fully visible one-row instruction at the minimum supported Library geometry; the existing dead zone prevents double-clicks and key repeat from counting as consent.
- [ ] #5 One deliberate second press applies a one-shot duplicate override only when the unchanged armed snapshot included an active-source reason; a combined active-source and preflight-warning confirmation still requires two presses total, never three.
- [ ] #6 Consent fingerprints stable matching job IDs, not their ordinary QUEUED/PARSING/WRITING transitions. Source/form/backend edits, changed warning identity, active-match membership changes, canvas reset/exit, or Escape disarm; identical preflight refresh and focus movement alone do not.
- [ ] #7 Folder admission is atomic: one outer check runs before any member is queued, and confirmed members use an already-admitted child seam so recursive re-entry cannot partially submit or re-block the batch.
- [ ] #8 The app repeats the guard immediately before local queue creation or remote submission. Expected refusal carries only bounded job ID/state references, is safe to stringify, creates no generic failure receipt, and releases retained external-model resources.
- [ ] #9 Focused registry, coordinator, screen, folder, URL/path-normalization, privacy, keyboard-parity, constrained-geometry, resource-lifecycle, and regression tests plus scoped static checks and documentation pass.
<!-- AC:END -->

## Design

- Detailed design: `Docs/superpowers/specs/2026-08-13-task-208-active-ingest-source-admission-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/065-active-ingest-source-admission-and-override.md`.
- Reason: TASK-208 establishes a durable idempotency scope, canonical source identity, batch admission rule, cross-module contract, and explicit override policy while rejecting persistent uniqueness and historical content hashing.
