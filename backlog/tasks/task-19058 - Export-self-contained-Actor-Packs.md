---
id: TASK-19058
title: Export self-contained Actor Packs
status: In Progress
assignee: []
created_date: '2026-08-20 18:27'
updated_date: '2026-08-22 21:56'
labels: []
dependencies:
  - TASK-19053
  - TASK-19056
  - TASK-19057
references:
  - >-
    Docs/superpowers/specs/2026-08-20-actor-pack-persona-buddy-and-emote-programme-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users export one eligible local Character or Persona as a deterministic, self-contained Actor Pack whose actor, portrait, and active visual versions come from one consistent authority snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Export validates an eligible local Character or Persona and its portrait before assigning a missing portable UUID; one-time assignment is durable and harmless and remains assigned if later archive writing or publication fails, while server-backed Personas remain disabled.
- [ ] #2 The snapshot captures and, after every await and immediately before publication, revalidates exact local source/profile identity, actor revision, portable UUID, portrait, active visual bindings/versions/assets, canonical content digests, and pinned source filesystem identity.
- [ ] #3 Every included visual section is self-contained and preserves its typed manifest/license/provenance; a missing declared asset fails rather than emitting a thin reference.
- [ ] #4 Export consumes TASK-19057 canonical JSON, canonical path, inventory, `actor-pack.json` self-exclusion, and non-self-referential digest contracts; output uses `ZIP_STORED`, fixed metadata/order, bounded streaming, and byte-identical bytes for identical canonical inputs, with archive, hash, decode, and file work off the event loop.
- [ ] #5 Publication uses a same-directory temporary file, file fsync then atomic replacement then parent-directory fsync where supported, no-follow pinned identities, and a capability-limited verified fail-closed fallback; cancellation shields and drains uncancellable work before cleanup or serialization release, removes only the owned temporary file, and leaves the destination untouched on stale authority, failure, or cancellation.
- [ ] #6 Local IDs, chats, deletion state, provider settings, credentials, paths, session/UI preferences, and private diagnostics never enter the archive.
- [ ] #7 Real export-to-independent-pure-validator/readback round trips, without import activation, cover minimal actor+portrait, Character, Persona, and both-visual-section exports alongside independent golden deterministic byte and digest oracles.
- [ ] #8 Verification includes born-RED-to-GREEN evidence, mutation proof for authority, path, cancellation, and privacy guards, assigned-worktree provenance, isolated HOME/XDG/config/data roots, focused race/package/licence/privacy tests, scoped Ruff/format/compile/diff checks, and diagnostic/privacy/architecture/governance gates.
<!-- AC:END -->

## Implementation Plan

1. Add immutable export snapshot contracts and capture exact local actor, portrait,
   portable identity, active visual graph, and source-file authority.
2. Project each snapshot into the existing canonical Actor Pack document contract,
   including self-contained typed visual manifests, assets, licences, and provenance.
3. Write deterministic `ZIP_STORED` archives with the metadata and ordering already
   frozen by TASK-19057, then validate them through an independent readback oracle.
4. Publish beside the selected destination with pinned no-follow authority, file and
   directory syncing, atomic replacement, owned-temporary cleanup, and stable errors.
5. Add one app-owned asynchronous export controller that owns admission, blocking
   work, results, cancellation, repeated-cancellation drain, and shutdown; screens
   submit immutable requests and apply only identity-fenced outcomes.
6. Expose one labelled Workbench `Export Actor Pack` action for eligible local
   Characters and Personas; server Personas continue to offer Save Local Copy first.
7. Run the focused Actor Pack, repository, visual, Workbench, privacy, architecture,
   packaging, provenance, static, and mutation gates documented in the detailed plan.

ADR required: no

ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`

Reason: ADR-074 already decides the self-contained deterministic envelope, local-only
eligibility, portable identity, separate visual sections, snapshot authority, and
atomic publication boundary implemented by this task.

Detailed plan: `Docs/superpowers/plans/2026-08-22-task-19058-actor-pack-export.md`
