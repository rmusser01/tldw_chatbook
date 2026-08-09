---
id: TASK-13203
title: Add private clone-reference profile storage and migration
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - profiles
  - privacy
dependencies:
  - TASK-13200
references:
  - backlog/decisions/028-character-tts-generation-profile-ownership.md
  - backlog/decisions/029-local-private-data-boundary.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend TTS profiles with canonical private clone-reference assets, safe migration, quotas, backup, restore, and damage isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The profile repository advances from v2 to v3 with one optional reference row per profile containing immutable reference UUID, canonical bounded WAV BLOB, bounded transcript, digest, validated audio metadata, and timestamps, while existing profile and character-assignment semantics remain unchanged.
- [ ] #2 Reference ingest accepts only a regular bounded WAV, validates and canonicalizes supported audio, strips arbitrary RIFF metadata, rechecks the source before commit, persists no source path, and commits reference plus profile revision atomically.
- [ ] #3 Per-reference size/duration/transcript limits and aggregate byte/count quotas are enforced transactionally, BLOB reads and writes are bounded/streamed, and list/open paths load metadata summaries rather than reference bytes.
- [ ] #4 Migration takes and retains an owner-private v2 online backup, validates the source and migrated schema/integrity/domain equivalence transactionally, publishes no partial v3 store on failure, and older builds refuse v3 with documented lossy v2-restore downgrade steps.
- [ ] #5 Repository backup and restore include full reference data, validate digests/WAV structure/quotas before replacement, and isolate a safely attributable damaged reference to its profile while structural or ambiguous corruption keeps the repository unavailable.
- [ ] #6 Editing, replacing, and deleting a reference follow profile revision and cascade ownership rules without silently retargeting a profile or character, and existing v2 profiles migrate with no reference and identical effective selection.
- [ ] #7 Storage, errors, diagnostics, logs, and tests expose no source path, transcript, audio bytes, digest-derived identity, or raw database detail; user-facing copy truthfully states local plaintext, filesystem-not-encryption, backup/export sensitivity, and best-effort deletion.
- [ ] #8 Migration, quota, canonicalization, concurrency, backup/restore, damage-isolation, downgrade-refusal, privacy, and unchanged v2-profile behavior are covered using isolated temporary repositories only.
<!-- AC:END -->
