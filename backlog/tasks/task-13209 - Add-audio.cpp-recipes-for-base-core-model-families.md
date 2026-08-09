---
id: TASK-13209
title: Add audio.cpp recipes for base core model families
status: To Do
assignee: []
created_date: '2026-08-09 17:39'
labels:
  - tts
  - audio-cpp
  - recipes
  - compatibility
dependencies:
  - TASK-13208
references:
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
  - backlog/decisions/051-private-tts-clone-reference-assets.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add exact release-0.5.1 recipes and evidence for chatterbox, dramabox, miotts, vibevoice, and moss_tts_nano.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 11 declared release-0.5.1 package entries across chatterbox (3), dramabox (1), miotts (3), vibevoice (2), and moss_tts_nano (2) have exact immutable recipe identities and no unreviewed package gap.
- [ ] #2 Each recipe declares only its upstream-supported tts/clone capabilities, exact required and optional files, safe model fields, model/voice/reference rules, lazy-load behavior, backend/platform tuples, compatibility state, and pinned recipe revision.
- [ ] #3 Exact positive package fixtures are recognized deterministically, while missing, extra-conflicting, ambiguous, renamed, mismatched-version, symlink/reparse, and near-match fixtures are rejected or left explicitly unclassified.
- [ ] #4 Generated server.json and catalog cross-check tests prove the exact model IDs/tasks for every recipe and continue excluding VC, dialogue-only, ASR, Music, and other non-TTS task entries from native TTS admission.
- [ ] #5 Every tuple labeled Verified has provisioned real-process evidence for configuration acceptance, health/catalog identity, structurally valid text or clone WAV output as declared, and definitive zero-leak shutdown; all other tuples remain visibly Untested, Unsupported, or Blocked.
- [ ] #6 The support/accounting UI and documentation name this exact family/package/platform/backend subset without changing existing Supertonic, PocketTTS, External, or user-provided server.json behavior.
<!-- AC:END -->
