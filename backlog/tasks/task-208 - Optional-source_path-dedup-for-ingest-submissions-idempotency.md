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
- [ ] #4 A blocked first press preserves the complete form and displays an inline, non-modal instruction to press Start again; the existing dead zone prevents double-clicks and key repeat from counting as consent.
- [ ] #5 One deliberate second press applies a one-shot override to the exact current source/backend/form snapshot and covers both an active-source warning and any existing preflight-warning consent without requiring a third press.
- [ ] #6 Editing the source or options, changing backend, leaving/resetting the canvas, or changing the active-match fingerprint disarms pending consent.
- [ ] #7 Folder admission is atomic: if any expanded member is already active, the first press queues none; the confirmed second press queues the original batch normally.
- [ ] #8 The app repeats the guard immediately before local queue creation or remote submission, expected duplicate refusal does not produce a generic failure receipt, and retained external-model resources are released when admission is refused.
- [ ] #9 Focused registry, coordinator, screen, folder, URL/path-normalization, keyboard-parity, resource-lifecycle, and regression tests plus scoped static checks and documentation pass.
<!-- AC:END -->

## Design

- Detailed design: `Docs/superpowers/specs/2026-08-13-task-208-active-ingest-source-admission-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/065-active-ingest-source-admission-and-override.md`.
- Reason: TASK-208 establishes a durable idempotency scope, canonical source identity, batch admission rule, cross-module contract, and explicit override policy while rejecting persistent uniqueness and historical content hashing.
