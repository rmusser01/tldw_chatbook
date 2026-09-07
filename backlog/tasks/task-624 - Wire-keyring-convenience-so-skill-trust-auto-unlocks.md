---
id: TASK-624
title: Wire keyring convenience so skill trust auto-unlocks
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 11:20'
updated_date: '2026-07-25 17:10'
labels:
  - skills
  - trust
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`SkillTrustService.unlock_from_keyring_convenience()` exists, is unit-tested, and has **zero callers in the application**. The feature it implements is therefore dead: enabling "keyring convenience" persists the derived trust keys to the OS keychain specifically so the passphrase does not have to be re-entered, but nothing ever loads them back.

The user-visible result is that every launch shows *"Skill trust is locked for this session"* in Library ▸ Skills and demands a manual Unlock, even for a user who explicitly enabled convenience. The keys sit in the keychain unused.

`app.py` constructs the trust service with a `key_cache` wired **and** `keyring_convenience_enabled=False` hardcoded, then never attempts the cached-key unlock, so the flag can only ever become True inside a process that just called `enable_keyring_convenience()` itself.

Found by live TUI verification (task-579). Unit tests cannot catch this class of defect: the method passes its own tests in isolation precisely because those tests call it directly — the missing thing is the call site.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With keyring convenience previously enabled, a fresh app launch reaches a usable trust state without the user entering a passphrase
- [x] #2 The Library ▸ Skills trust header does not report "locked for this session" when cached keys are available and valid
- [x] #3 A failed or absent cached-key load still falls back cleanly to the existing locked/Unlock path — never a crash and never a silent unlocked-looking state
- [x] #4 A stale cache (salt no longer matching the manifest) does not unlock, matching the existing salt-bound behaviour
- [x] #5 Whatever surface enables convenience persists it, so the setting survives a restart rather than being reset to False on every construction
- [x] #6 A regression test asserts the auto-unlock is actually invoked at startup, not merely that the method works when called
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wired the cached-key unlock that existed but was never called.

Root cause: unlock_from_keyring_convenience() was unit-tested yet had ZERO production callers, so every lockedness decision saw self._keys is None and reported 'locked for this session' while the derived keys sat unused in the keychain. The method passed its own tests precisely because those tests supplied the call site production lacked.

Fix: a single memoized _try_cached_unlock() called at the four points that decide lockedness — trust_posture(), status_for_skill(), trusted_file_paths(), _require_keys(). Putting it behind the choke points rather than at app startup means every consumer benefits (Library UI, agent worker thread, Console skill paths) with no startup keychain cost, and it stays lazy.

Design notes: the attempt is latched to once per trust-state generation because keychain reads are not free and posture is queried on every Library render (pinned by a counting-cache test asserting exactly 1 call across 10 queries); the latch clears in reset_trust so a later re-bootstrap can cache again. AC#5 needed no config key — the keychain entry IS the persistence, and unlock_from_keyring_convenience already flips keyring_convenience_enabled on success. Failure is silent by design: a missing, stale, or exploding cache leaves the session locked so the existing Unlock path remains the fallback. Also added the module's missing loguru import.

LIVE-VERIFIED in a real app (the point, since unit tests could not catch this): same QA profile, before = 'Skill trust is locked for this session.' with warning glyphs on both skills; after = 'Skill trust: ready.' with checkmarks, no unlock and no passphrase. Capture at Docs/superpowers/qa/skills-script-execution-2026-07-25/after-fix-trust-ready.txt.

Tests: Tests/Skills/test_skill_trust_keyring_autounlock.py (8 tests) RED-first — 5 failed against the old code, 3 (locked/stale/absent-cache fallbacks) passed throughout, proving no false baseline. Skills 357, Agents 261, ruff clean.

Files: tldw_chatbook/Skills_Interop/skill_trust_service.py, Tests/Skills/test_skill_trust_keyring_autounlock.py
<!-- SECTION:NOTES:END -->
