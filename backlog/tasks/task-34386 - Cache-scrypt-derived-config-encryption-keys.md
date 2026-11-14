---
id: TASK-34386
title: Cache scrypt-derived config-encryption keys
status: Done
created_date: 2026-10-11 21:42
labels:
- performance
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-11 21:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConfigCrypto.encrypt_value/decrypt_value run scrypt per value with a fresh random salt, so one config load with N encrypted secrets costs N full KDF runs (~0.3-1.5s CPU) and a save cycle runs ~3N. Memoize derived keys on (password, salt, N) so repeat operations over stable salts become pure AES-GCM. Memoization only: no ciphertext format change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repeated encrypt/decrypt with the same (password, salt, N) derives the key at most once (call-counted),Round-trip encrypt/decrypt behavior and ciphertext format unchanged,Cache is bounded (no unbounded growth across random per-value salts),Derivation is thread-safe (config writes happen under locks from multiple threads)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read Utils/config_encryption.py and its tests
2. Add a bounded, thread-safe LRU memo on (password, salt, N) key derivation
3. Add call-count test proving one derivation for repeated ops; keep round-trip tests green
4. Write ADR (encryption-adjacent change: in-memory key retention)
5. Update task ACs/notes, close
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Memoized scrypt via bounded lru_cache on (password,salt,N); lifecycle clears wired into config.py; ADR-228; 43+58 targeted tests green
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

- Added a module-level `functools.lru_cache(maxsize=128)` (`_derive_key_cached`) around
  scrypt in `Utils/config_encryption.py`, keyed on the full derivation input
  `(password, salt, N, key_len, r, p)`; `ConfigEncryption._derive_key` now routes
  through it, and a `clear_derived_key_cache()` static method exposes explicit invalidation.
- `config.py` `set_encryption_password` / `clear_encryption_password` call the clear so
  keys derived under an old password do not survive a password change. Encrypt keeps its
  fresh random salt per value (no format change); decrypt-after-encrypt and all repeat
  decrypt cycles (the save pipeline's ~3N scrypt per save) now hit the memo.
- ADR: `backlog/decisions/228-config-encryption-derived-key-memoization.md` (in-memory
  key retention tradeoff, alternatives incl. deferred master-key v3 format, no-new-oracle reasoning).
- Tests: `Tests/Utils/test_config_encryption.py::TestDerivedKeyMemoization` (5 tests,
  call-counted scrypt across round-trips, clears, v1/v2 separation); full file 43 passed;
  `Tests/test_config_app_config_encryption.py`, `test_config_raw_snapshot.py`,
  `test_config_private_bootstrap.py`, `test_config_encryption_effective_path.py`,
  `test_config_persistence_owner.py` — 58 passed.
- Deviation: none from ACs.
