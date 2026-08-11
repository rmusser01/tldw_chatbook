---
id: TASK-13206
title: Add explicit clone voice bundle portability
status: In Progress
assignee: []
created_date: '2026-08-09 17:39'
updated_date: '2026-08-11 22:06'
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
  - Docs/superpowers/specs/2026-08-11-audio-cpp-clone-voice-bundle-portability-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit warning-gated export and hostile-input-safe import for portable clone voice bundles while ordinary exports remain sanitized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ordinary profile export keeps reference-free profiles on wire v1 and exports reference-bearing profiles as sanitized wire v2 with an explicit reference-omitted marker and no audio, transcript, local-oracle digest, path, assignment, endpoint, generated config, or private runtime state.
- [ ] #2 Explicit voice-bundle export requires a plaintext/sensitive-data warning and creates only the versioned allowlisted ZIP entries manifest.json, profile.json, reference.wav, and reference.txt with bounded metadata, exact sizes, and canonical SHA-256 checksums.
- [ ] #3 Bundle data contains the sanitized profile selection and generic user declaration only—never model weights, character/persona data, assignments, defaults, credentials, origins, recipe code, process state, or unnecessary timestamps.
- [ ] #4 Import validates the unchanged source archive before storage and rejects encryption, duplicate or normalized-name collisions, absolute/traversal/separator paths, symlinks/special files, unknown or missing entries, unsupported compression, size/count/ratio limit breaches, malformed content, and checksum mismatch.
- [ ] #5 Validation and extraction use a bounded owner-private staging directory without trusting archive paths, clean up on every terminal path, and describe checksums only as byte integrity rather than authenticity, speaker identity, signature, or consent proof.
- [ ] #6 Import displays exact UUID, name, recipe, and model dependency conflicts for explicit user resolution; it never overwrites, assigns, changes a default, or retargets automatically, and may store a valid unresolved bundle only as inactive Needs compatible model.
- [ ] #7 Hostile-archive, source-mutation, quota, collision, old-reader, missing-model, cleanup, privacy, and deterministic round-trip tests pass, and manual UAT proves ordinary sanitized export versus explicit warned bundle transfer.
- [ ] #8 The profile store advances transactionally from v3 to v4 with exact optional recipe provenance, a retained owner-private pre-v4 backup, no guessed legacy provenance, bounded downgrade guidance, and rollback that leaves v3 authoritative on failure.
- [ ] #9 Newly saved and imported clone references carry exact recipe/model requirements through availability and service-owned runtime admission; mismatch blocks before provider work, while migrated legacy references remain usable with a visible provenance-unavailable advisory and cannot be bundled until regenerated.
- [ ] #10 Import/export work is retained and joined across cancellation and shutdown, performs no audio.cpp launch/network/Settings mutation during inspection, fails closed without verified owner-private containment, and exposes bounded accessible recovery without leaking private archive/reference values.
<!-- AC:END -->
