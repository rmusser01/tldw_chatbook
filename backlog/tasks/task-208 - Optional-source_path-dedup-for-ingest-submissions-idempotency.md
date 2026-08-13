---
id: TASK-208
title: Guard active ingest submissions against duplicate sources
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 17:34'
updated_date: '2026-08-13 22:16'
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
- [x] #1 A source matching a QUEUED, PARSING, or WRITING job for the same Local or Server backend is not submitted on the first Start press; terminal jobs never block re-ingestion.
- [x] #2 Equivalent local path spellings and conservative HTTP(S) URL spellings share a stable lexical admission key without filesystem, network, database, or content-hash work.
- [x] #3 Local and Server jobs use separate admission scopes, so an active Local import does not block submission of the same source to Server and vice versa.
- [x] #4 A blocked first press preserves the complete form and displays a compact, fully visible one-row instruction at the minimum supported Library geometry; the existing dead zone prevents double-clicks and key repeat from counting as consent.
- [x] #5 One deliberate second press applies a one-shot duplicate override only when the unchanged armed snapshot included an active-source reason; submit-time candidate identity is exact and every current active match is covered, while a combined active-source and preflight-warning confirmation still requires two presses total, never three.
- [x] #6 Consent fingerprints candidate-set identity/count, tooling affected-file count, and stable matching job IDs, not their ordinary QUEUED/PARSING/WRITING transitions. Source/form/backend edits, candidate mutations, changed warning identity or affected count, active-match membership changes, canvas reset/exit, or Escape disarm; an identical full preflight refresh and focus movement alone do not.
- [x] #7 Folder admission is atomic: one outer check runs after current expansion and before any member is queued, changed members re-arm, and confirmed exact members use an already-admitted child seam so recursive re-entry cannot partially submit or re-block the batch.
- [x] #8 The app repeats the guard immediately before local queue creation or remote submission. Expected refusal carries only bounded job ID/state references plus an opaque candidate digest/count, is safe to stringify without source metadata, creates no generic failure receipt, and releases retained external-model resources.
- [x] #9 Focused registry, coordinator, screen, folder, URL/path-normalization, privacy, keyboard-parity, constrained-geometry, resource-lifecycle, and regression tests plus scoped static checks and documentation pass.
<!-- AC:END -->

## Design

- Detailed design: `Docs/superpowers/specs/2026-08-13-task-208-active-ingest-source-admission-design.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/065-active-ingest-source-admission-and-override.md`.
- Reason: TASK-208 establishes a durable idempotency scope, canonical source identity, batch admission rule, cross-module contract, and explicit override policy while rejecting persistent uniqueness and historical content hashing.

## Implementation Plan

- Detailed plan: `Docs/superpowers/plans/2026-08-13-task-208-active-ingest-source-admission.md`.
- ADR required: yes.
- ADR path: `backlog/decisions/065-active-ingest-source-admission-and-override.md`.
- Reason: ADR-065 is accepted and directly governs the active-state scope, lexical identity, app-boundary authority, atomic folder admission, privacy-safe refusal, and one-shot override implemented by this plan; no additional ADR is required.

1. Add and verify the pure canonical source key, active-state registry query, and bounded privacy-safe refusal references.
2. Refactor submission into one authoritative outer admission check plus a private already-admitted child seam, preserving Local/Server routing and atomic folder behavior.
3. Extend the existing inline two-press Start grammar with stable request fingerprints, reason-specific override forwarding, late-refusal recovery, and external-scope release.
4. Prove the exact warning copy through painted `72x18` compositor tests, update the Library user guide, run the affected regression/static matrix, and complete task/ADR hygiene.
5. Final review fix wave: replace the Boolean override with a privacy-safe exact candidate/membership scope, add mutation-sensitive screen/app regressions, rerun TASK-208 plus TASK-15742 evidence, and restore Done only after ACs 5–8 are green.

## Implementation Notes

Implemented ADR-065 across the registry, app coordinator, and Library screen: pure lexical Local/Server admission keys; active-only matching; one authoritative outer guard with an already-admitted folder-child seam; bounded privacy-safe refusal references; immutable consent fingerprints keyed by stable matching job IDs; reason-specific, one-shot duplicate override; and release of retained external-model resources on expected refusal. The existing inline confirmation now handles active-only and combined active/preflight warnings in two deliberate presses while edits, membership changes, reset/exit, and Escape disarm it. No preference, schema, dependency, persistent-uniqueness, or historical-content-dedup behavior was added. ADR-065 is the governing accepted ADR; ADR-014 remains unchanged and no additional ADR is required.

Task 4 added painted compositor proofs for all three exact instruction sentences at `72x18`, plus a mounted full-screen proof that the consent repaint preserves widget identity, input focus and cursor, canvas scroll, and the Start-region geometry. The Library user guide now explains active duplicate admission for Local and Server files and atomic folders, combined warnings, terminal-job behavior, and disarming edits.

Final review fix: replaced the bare Boolean bypass with `ActiveIngestConsentScope`, an immutable origin/candidate-digest/count/active-ID carrier. The app rebuilds it after authoritative folder expansion, requires exact candidate identity and coverage of every current active job ID, permits ordinary lifecycle shrink, and fails closed if bounded active-ID references truncate. Refusal returns only the opaque scope and bounded safe references; the screen re-arms candidate changes, keeps tooling affected-count in its fingerprint, and preserves external-scope release and one-shot consumption. Added app mutation cases, affected-count and late-refusal screen cases, and a mounted real screen-to-app regression proving a member added between presses queues nothing.

Final evidence: registry 90 passed; app 120 passed plus 7 inherited parse-pool assertion-shape failures; state 234 passed plus the documented Windows `WinError 1314` symlink privilege failure; inline 51 passed plus 4 inherited unmounted-Textual-worker harness failures; canvas 135 passed; integration 12 passed plus the inherited options-default assertion; runner 134 passed plus the inherited logger-placeholder assertion; selected shell 7 passed (566 deselected). Mutation-sensitive app selection passed 6, real screen-to-app passed 1, and the remaining consent/refusal selection passed 7 before its one inherited unmounted-worker failure. TASK-15742's combined portability matrix passed 33 with 4 capability skips. Scoped Ruff, Python compilation, and diff checks passed. Self-review confirmed Local/Server partitioning, atomic pre-side-effect refusal, privacy-safe payloads, active-ID overflow refusal, terminal/lifecycle behavior, external resource release, and no schema/dependency/persistent-state change. ADR-065 remains the governing accepted ADR; no new ADR was required.
