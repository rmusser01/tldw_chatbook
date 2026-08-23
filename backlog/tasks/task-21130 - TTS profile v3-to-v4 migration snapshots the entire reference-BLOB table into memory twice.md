---
id: TASK-21130
title: >-
  TTS profile v3-to-v4 migration snapshots the entire reference-BLOB table into memory twice
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - tts
  - migrations
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21130).

`TTS/profile_schema.py:1300-1316` (`_migration_reference_snapshot`) pulls `wav_bytes` for every
row, and is called at :1439 and :1468 with the first snapshot still held - up to ~1 GB peak at
profile-store open under the 512 MiB store bound (profile_reference_types.py:21), against the
module's own 256 KiB streaming norm. Swap-thrash on constrained hardware at exactly the moment
TTS opens.

## Acceptance Criteria

- [ ] The migration compares projections without wav_bytes (sibling profile_migration_candidate.py:320 already does) using hashes; peak memory during v3->v4 is bounded and asserted by a test with synthetic large references
- [ ] Migration outcomes byte-identical for the existing fixtures
