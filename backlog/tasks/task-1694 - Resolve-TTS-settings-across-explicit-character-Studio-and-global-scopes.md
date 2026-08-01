---
id: TASK-1694
title: Resolve TTS settings across explicit character Studio and global scopes
status: To Do
assignee: []
created_date: '2026-08-01 06:02'
labels:
  - tts
  - settings
  - roleplay
dependencies:
  - TASK-1692
  - TASK-1693
references:
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every admitted TTS request use one coherent and explainable selection across explicit caller intent, assigned character profiles, saved Studio preferences, global defaults, and provider fallbacks. This prevents roleplay and Studio behavior from depending on mutable widgets or silently falling through after an exact selection becomes invalid.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Normal, roleplay, media, and other non-Studio requests resolve each applicable axis in the exact order explicit caller value, assigned character TTS profile for an authoritative assistant CharacterRef, global default, then provider-declared fallback (STATE-001).
- [ ] #2 Studio requests resolve each applicable axis in the exact order current validated Studio controls or explicit preview, persisted Studio preference, global default, then provider-declared fallback; an unrelated selected character is never injected implicitly (STATE-002).
- [ ] #3 Resolution produces one immutable snapshot containing provider, model mode and value, voice mode and value, format, speed, validated provider options, source for each axis, and relevant preference and provider revisions before adapter admission (STATE-003).
- [ ] #4 The effective snapshot contains no credential, endpoint secret, submitted synthesis text, mutable widget reference, character payload, or adapter instance (SEC-001 through SEC-004).
- [ ] #5 Absence inherits, but an invalid, missing exact, unsupported, or revision-incoherent higher-precedence value blocks the affected request without falling through or changing providers (STATE-005 and CFG-011).
- [ ] #6 First available resolves exactly once at request admission and Server default remains an omitted voice; neither resolved ephemeral identifier is written to global, Studio, or character storage (STATE-004).
- [ ] #7 audio.cpp remains constrained to WAV, speed 1.0, and no arbitrary options at every scope, while unknown provider options fail closed and legacy providers retain their existing supported request behavior.
- [ ] #8 Resolution never mutates, repairs, deletes, or reassigns a character TTS profile, and a preview is distinguishable from an adopted Studio preference (OWN-003 and CFG-025).
- [ ] #9 Table-driven deterministic tests cover every precedence layer, presence and absence combination, authoritative and missing character identity, exact and dynamic modes, invalid higher layers, provider constraints, source metadata, and unchanged legacy/global-only behavior.
<!-- AC:END -->
