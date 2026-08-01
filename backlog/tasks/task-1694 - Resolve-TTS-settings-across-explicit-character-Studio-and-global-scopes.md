---
id: TASK-1694
title: Resolve TTS settings across explicit character Studio and global scopes
status: Done
assignee: []
created_date: '2026-08-01 06:02'
updated_date: '2026-08-01 09:52'
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
- [x] #1 Normal, roleplay, media, and other non-Studio requests resolve each applicable axis in the exact order explicit caller value, assigned character TTS profile for an authoritative assistant CharacterRef, global default, then provider-declared fallback (STATE-001).
- [x] #2 Studio requests resolve each applicable axis in the exact order current validated Studio controls or explicit preview, persisted Studio preference, global default, then provider-declared fallback; an unrelated selected character is never injected implicitly (STATE-002).
- [x] #3 Resolution produces one immutable snapshot containing provider, model mode and value, voice mode and value, format, speed, validated provider options, source for each axis, and relevant preference and provider revisions before adapter admission (STATE-003).
- [x] #4 The effective snapshot contains no credential, endpoint secret, submitted synthesis text, mutable widget reference, character payload, or adapter instance (SEC-001 through SEC-004).
- [x] #5 Absence inherits, but an invalid, missing exact, unsupported, or revision-incoherent higher-precedence value blocks the affected request without falling through or changing providers (STATE-005 and CFG-011).
- [x] #6 First available resolves exactly once at request admission and Server default remains an omitted voice; neither resolved ephemeral identifier is written to global, Studio, or character storage (STATE-004).
- [x] #7 audio.cpp remains constrained to WAV, speed 1.0, and no arbitrary options at every scope, while unknown provider options fail closed and legacy providers retain their existing supported request behavior.
- [x] #8 Resolution never mutates, repairs, deletes, or reassigns a character TTS profile, and a preview is distinguishable from an adopted Studio preference (OWN-003 and CFG-025).
- [x] #9 Table-driven deterministic tests cover every precedence layer, presence and absence combination, authoritative and missing character identity, exact and dynamic modes, invalid higher layers, provider constraints, source metadata, and unchanged legacy/global-only behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: TASK-1694 implements ADR-039's accepted precedence, dynamic-selection, revision, preview, and fail-closed admission boundary; no new ADR is required.

Detailed plan: Docs/superpowers/plans/2026-08-01-task-1694-effective-tts-resolution.md

1. Add failing tests for immutable text-free snapshots, source metadata, canonical providers, and provider constraints.
2. Implement normal explicit/character/global/fallback resolution with provider isolation and bounded failures.
3. Implement Studio draft/preview/saved/global/fallback resolution, stale-draft rejection, and one-time dynamic model resolution.
4. Route existing default and native-exact request admission through the shared snapshot without changing the temporary legacy bridge contract.
5. Run focused and full TTS regressions, static checks, independent review, and record ADR conformance before marking Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an immutable effective-selection resolver for explicit, authoritative character, Studio, global, and provider-fallback scopes, including per-axis provenance and preference/configuration/catalog revisions. Routed default, exact, character Console, and Studio admission through the coherent snapshot; added authoritative audio.cpp exact model/voice validation, fixed WAV/speed/options enforcement, preserved dynamic model/server-default semantics, and kept submitted text outside the snapshot. Wired the application service to a lazy non-migrating Studio preference reader and preserved exact legacy model, voice case, format, speed, and supported Chatterbox options through the temporary bridge.

ADR: Implements backlog/decisions/039-global-and-studio-tts-settings-ownership.md; no new ADR was needed.

Verification: focused resolver/admission/bootstrap/Console/Studio/UAT sweep passed (280 tests); full Tests/TTS run completed with 2040 passed and 14 skipped. Its sole failure is the unchanged pre-existing package-export baseline in test_tts_logging_privacy.py, where three portability symbols from the earlier portability slice remain exported. Ruff check, Ruff format check for changed formatted files, compileall, and git diff --check passed. Independent review concluded READY with no remaining actionable findings.

Plan deviation: removed legacy bridge model/format/voice-case rewriting after review showed it made the frozen effective snapshot disagree with the admitted request; routing remains behind the existing temporary legacy bridge.
<!-- SECTION:NOTES:END -->
