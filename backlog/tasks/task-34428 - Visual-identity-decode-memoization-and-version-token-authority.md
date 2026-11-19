---
id: TASK-34428
title: Visual-identity decode memoization and version-token authority
status: Done
created_date: 2026-10-07 02:42
updated_date: 2026-10-07 20:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 6 / F13: visual identity resolution fully decodes every animation frame plus sha256 per state change and playback decodes the same frames again while persona authority re-hashes a 25MB portrait up to four times per resolve
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Second resolution performs zero frame decodes,Exactly one full decode per asset per stat signature,Persona revalidation compares version not bytes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 16 (T16)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Visual-identity decode memoization: 32-entry facts-only LRU keyed (path, size, mtime_ns) / manifest-sha for builtins / content digest for blobs — repeat resolutions perform zero Image.open/seek/load (corruption-check decode once per stat signature, mtime invalidation, failures never cached); shared prepared frames consumed by playback via a byte-bounded retention store (8 entries / 16 MiB aggregate, line-for-line parity with the legacy decode path pinned by a tobytes golden test; assets over the cap honestly fall back to the old second decode). Persona authority: memo keyed (service-instance, character_id, revision) with a short-circuiting byte-equality memcmp guard on the cheap path (version-only tokening failed the existing ABA test — guard preserves that contract while keeping zero sha256 on unchanged cards); portraits capped 4 entries / 64 MiB. Three independent short-held locks, decodes/memcmps outside locks. Evidence: true 512-frame GIF, resolve+prepare x2: 4 full-decode passes -> 1. 9 new TDD tests; targeted failure lists byte-identical to baseline. Files: Character_Chat/visual_identity.py, Chat/character_expression_playback.py, Character_Chat/persona_visual_identity.py, Tests/Persona_Visual/test_visual_identity_decode_memoization.py. Report: .superpowers/sdd/task-16-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
