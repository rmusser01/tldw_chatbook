---
id: TASK-32921
title: Skill-trust keyring reads no longer run once per skill on every send
status: Done
assignee:
- '@claude'
created_date: 2026-09-23 15:43
labels:
- performance
- linux
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every send and every Chat visit computes a trust status for each installed skill on the UI loop, and each status re-read the rollback marker (and, when locked, the key cache) from the OS keyring. On Linux each read is a SecretService D-Bus round trip, and a locked gnome-keyring can block on an unlock prompt -- N round trips/prompts per send that scale with installed skills. Most likely Linux-specific cause of the reported lag.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated marker and key-cache reads within a pass share one keyring round trip
- [x] #2 A write or clear through the store is visible to the very next read
- [x] #3 A failed read is shared only briefly, so a locked keyring raises once per pass while a later Retry still reads fresh
- [x] #4 Rollback protection is unchanged: a stale cached marker can only cause a false mismatch, never a false accept
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Short-lived read cache in the two keyring stores' read methods (the one place every caller routes through)
2. 30 s success / 2 s failure windows; each store's own writes clear it
3. Tests with a counting fake keyring and a patched clock
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`_cached_keyring_read` in `Skills_Interop/skill_trust_store.py` sits under both `KeyringSkillTrustGenerationMarkerStore.load_marker` and `KeyringSkillTrustKeyCache.load_keys` -- the one place every per-skill trust check routes through. Successful reads (including "absent") are shared for 30 s; failures for 2 s, so a locked keyring raises once per pass instead of once per skill while `trust_posture()`'s Retry contract (skill_trust_service.py:240-258) still holds. Each store's own save/clear invalidates its cache AFTER the write and bumps a generation counter, so a read racing the write (trust setup runs on a worker thread while sends read on the loop) cannot re-cache the replaced marker -- pinned by a race test that fails when the bump is removed.

Security: `load_manifest` requires an exact generation+digest match against the marker, so a stale cached marker can only yield a false mismatch (fail-closed), never accept a rolled-back manifest. Cross-instance staleness is bounded by the TTL (marked `ponytail:`).

Not changed: `_scan_skill` still hashes skill files per skill on the loop; the mount/resume skill-context fetch remains a non-threaded worker.

Files: `Skills_Interop/skill_trust_store.py`, `Tests/Skills/test_skill_trust_store.py`.

Qodo review fixes: expiry now starts when the read RETURNS, not before it; a locked keyring slower than the 2 s failure window used to store an already-expired failure. Integration test: the real `SkillTrustService.status_for_skill` over 10 skills does one keyring read (10 before).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
