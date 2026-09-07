---
id: TASK-13206
title: Add explicit clone voice bundle portability
status: Done
assignee: []
created_date: '2026-08-09 17:39'
updated_date: '2026-08-13 07:33'
labels:
  - tts
  - audio-cpp
  - profiles
  - security
  - portability
dependencies:
  - TASK-13205
references:
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
  - >-
    Docs/superpowers/specs/2026-08-11-audio-cpp-clone-voice-bundle-portability-design.md
  - >-
    Docs/superpowers/plans/2026-08-11-task-13206-clone-voice-bundle-portability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit warning-gated export and hostile-input-safe import for portable clone voice bundles while ordinary exports remain sanitized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ordinary profile export keeps reference-free profiles on wire v1 and exports reference-bearing profiles as sanitized wire v2 with an explicit reference-omitted marker and no audio, transcript, local-oracle digest, path, assignment, endpoint, generated config, or private runtime state.
- [x] #2 Explicit voice-bundle export requires a plaintext/sensitive-data warning and creates only the versioned allowlisted ZIP entries manifest.json, profile.json, reference.wav, and reference.txt with bounded metadata, exact sizes, and canonical SHA-256 checksums.
- [x] #3 Bundle data contains the sanitized profile selection and generic user declaration only—never model weights, character/persona data, assignments, defaults, credentials, origins, recipe code, process state, or unnecessary timestamps.
- [x] #4 Import validates the unchanged source archive before storage and rejects encryption, duplicate or normalized-name collisions, absolute/traversal/separator paths, symlinks/special files, unknown or missing entries, unsupported compression, size/count/ratio limit breaches, malformed content, and checksum mismatch.
- [x] #5 Validation and extraction use a bounded owner-private staging directory without trusting archive paths, clean up on every terminal path, and describe checksums only as byte integrity rather than authenticity, speaker identity, signature, or consent proof.
- [x] #6 Import displays exact UUID, name, recipe, and model dependency conflicts for explicit user resolution; it never overwrites, assigns, changes a default, or retargets automatically, and may store a valid unresolved bundle only as inactive Needs compatible model.
- [x] #7 Hostile-archive, source-mutation, quota, collision, old-reader, missing-model, cleanup, privacy, and deterministic round-trip tests pass, and manual UAT proves ordinary sanitized export versus explicit warned bundle transfer.
- [x] #8 The profile store advances from every supported older schema to v4 through validated private candidates with exact optional recipe provenance and retained pre-v3/pre-v4 backups; pre-publication failure leaves the active store unchanged, while the non-cancellable publication protocol completes a valid v4 store or durably restores the prior store and retains bounded recovery artifacts if total storage failure prevents both.
- [x] #9 Newly saved and imported clone references carry exact recipe/model requirements through availability and service-owned runtime admission; recipe/model/config mismatch blocks before provider work and post-ready generation drift blocks before private materialization or synthesis, while migrated legacy references remain usable with a visible provenance-unavailable advisory and cannot be bundled until regenerated.
- [x] #10 An app-owned bounded inspection service retains private single-use source authority while the UI receives only safe review facts; commit revalidates source/dependency evidence and performs exact reuse or conflict recheck plus profile/recipe/reference creation in one serialized repository transaction.
- [x] #11 Import/export/migration work is retained and joined across cancellation and shutdown, performs no adapter acquisition, audio.cpp launch, network, or Settings mutation during inspection, fails closed without verified owner-private containment, and exposes bounded accessible recovery without leaking private archive/reference values.
- [x] #12 Provenance-bearing profile edits preserve exact recipe/model invariants, bundle export never overwrites an existing destination, and inactive blockers plus provenance-unavailable advisories expose truthful immutable action projections across the profile library and assignment consumers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/028-character-tts-generation-profile-ownership.md`, `backlog/decisions/029-local-private-data-boundary.md`, `backlog/decisions/051-private-tts-clone-reference-assets.md`

Reason: ADR-051 already owns clone-reference storage, migration, privacy, runtime admission, and portability and has been amended for this protocol. ADR-028 keeps assignment explicit and ADR-029 governs owner-private local data.

1. Preserve ordinary wire v1 while adding strict sanitized omission wire v2 and immutable recipe provenance.
2. Migrate isolated profile stores to v4 through private candidates, retained backups, and recoverable non-cancellable publication.
3. Require provenance for new reference writes and add one atomic repository-lane exact-reuse/create/copy decision.
4. Implement the deterministic four-entry codec and hostile archive rejection without general extraction.
5. Add the app-owned retained inspection/export service and close it before the repository.
6. Derive pure dependency truth and enforce pre-provider, adapter-preflight, and post-ready generation gates.
7. Extend the existing Voice Profile library and reuse inactive/advisory action truth in Personas assignment.
8. Complete privacy/lifecycle testing, static/full verification, two-launch isolated UAT, docs, review, and closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented sanitized wire-v2 export, strict warning-gated four-entry voice
  bundles, hostile-input validation, inactive conflict recovery, exact
  dependency admission, schema-v4 migration/recovery, and service-owned
  lifecycle integration under ADR-028, ADR-029, and ADR-051.
- Added cross-cutting privacy/lifecycle regressions for legacy-reader behavior,
  repository rollback, runtime collaborator failures, UI error surfaces, and
  composite shutdown; the runtime regression exposed and fixed an unsanitized
  adapter-preflight exception boundary.
- Updated the user/developer speech documentation and recorded both the limited
  service-layer setup and an isolated two-launch production-mounted Pilot UAT
  on commit `6eab86144`. Production warnings gated chooser/publication, the
  imported inactive profile survived restart with `Needs compatible model`
  under a deterministic non-launching missing-dependency observer, no
  assignment was created, and composite shutdown left no portability
  sessions/tasks or staging/output residue.
- The complete scoped matrix produced 2,278 passes and three sandbox-blocked
  real-child cases; the parent host reran those exact node IDs on `d3d60abcb`
  with `3 passed in 2.33s`, closing scoped automated coverage at 2,281 passed
  and 2 skipped. This does not constitute clone-model or audible UAT.
- A separate exact real-model run on `3583343d1` used the production Guided
  `TTSService` path with the previously reviewed audio.cpp executable and
  official PocketTTS English bf16 package. Dependency admission was exact,
  generation returned a valid PCM16 mono 24 kHz WAV, both reference and result
  playback exited 0, the user confirmed the expected voice was audible, and
  shutdown left no owned process/materialization/generated artifact.
- A terminal full repository run on feature `8821c3293` / exact base
  `a4b16b1e2` completed with 41,064 passed, 241 skipped, 4 xfailed, 383 failed,
  and 59 errors in 1:54:00. Exact-base reruns and focused corrections removed
  every base-green/feature-red node. The 411 baseline-red node IDs and artifact
  hashes are preserved in the QA evidence for a separate follow-up task/PR;
  this is not represented as a green full suite.
- Rebased final code/test commit `77a5e22b2` has clean merge-base
  `f7fe006ca`. All 11 intervening upstream commits were inspected. Their 10
  changed test files passed 804 tests; the only shared production path was
  `app.py`, with non-overlapping library-ingest versus TTS ownership/shutdown
  hunks. The rebased feature matrix closed at 2,281 passed / 2 skipped across
  sandbox and exact host rerun. Changed-file Ruff, planned format, 17-source
  mypy, normalized merge-base mypy comparison (zero new diagnostics), CSS
  sync, diff-check, and task-ID uniqueness all passed. Relevant inventories
  passed 88 tests; the remaining two persistent-diagnostic census failures
  reproduced on exact `f7fe006ca`.
- Independent final review approved exact evidence HEAD `21673c09f`; all 12
  acceptance criteria and the repository Definition of Done are supported by
  the linked automated, static, privacy, lifecycle, migration, isolated UAT,
  audible UAT, rebase, and exact-base classification evidence. The remaining
  repository-wide baseline failures are preserved for follow-up rather than
  hidden or relabeled.

Detailed test-first steps, file ownership, commands, review checkpoints, and commit boundaries are in `Docs/superpowers/plans/2026-08-11-task-13206-clone-voice-bundle-portability.md`.
<!-- SECTION:NOTES:END -->
