# ADR-212: Memoize scrypt-derived config-encryption keys in process memory

Date: 2026-10-11
Status: Accepted
Task: [TASK-34386](../tasks/task-34386%20-%20Cache-scrypt-derived-config-encryption-keys.md)

## Context

`ConfigEncryption.encrypt_value` generates a fresh random salt per value and derives a
key with scrypt (N=16384, r=8) on every encrypt **and** every decrypt. Config saves
decrypt the whole tree, encrypt changed values, and re-walk the tree several times
(see the save pipeline in `config.py`), and every config load decrypts all values
again. With N encrypted secrets a single load costs N full KDF runs (~0.3–1.5 s of
CPU for realistic N), and a save cycle costs roughly 3N. This class of pain is
documented in the module itself: v1 used N=2^20 and "made normal config saves
unusably slow", so v2 lowered the cost parameter — trading KDF strength to recover
save latency, because the *repeat* derivations were never addressed.

The salt is embedded in each ciphertext and stable between saves, so decrypt
re-derives byte-identical keys on every cycle. Derivation is a pure function of
`(password, salt, N, key_len, r, p)`.

## Decision

1. Memoize scrypt key derivation in `Utils/config_encryption.py` behind an
   LRU cache keyed on the full derivation input `(password, salt, N, key_len, r, p)`,
   bounded at 128 entries. Ciphertext format, salts, nonces, and the v1/v2 version
   scheme are unchanged; this is memoization of a pure function, not a format change.
2. The cache is cleared on the encryption-password lifecycle: `set_encryption_password`
   and `clear_encryption_password` in `config.py` call
   `ConfigEncryption.clear_derived_key_cache()`, so keys derived under an old password do
   not outlive a password change beyond the call site.
3. Repeat derivations of the same `(password, salt, N)` become AES-GCM-only (~µs).
   First-time derivations (fresh salt on encrypt; first decrypt of each stored value)
   still pay one KDF each — unavoidable without a format change.

## Alternatives considered

- **Master-key format (v3)**: derive one key per password+file-salt, encrypt values
  under it with AES-GCM. Eliminates per-value KDF entirely and is the cleaner end
  state, but requires a ciphertext migration, cross-version reading of old configs,
  and a re-encryption prompt. Deferred; the memo captures most of the win with zero
  migration. If v3 is adopted later, the memo becomes harmless and removable.
- **Per-file fixed salt**: changes the stored format for a smaller win; rejected for
  the same migration cost with less benefit.
- **Do nothing / lower N again**: v2 already paid strength for latency; repeating
  that trade erodes the KDF's purpose.

## Security consequences

- Derived keys and the password string are retained in process memory for up to
  128 cache entries. The password is already held in process memory for the whole
  session (`_ENCRYPTION_PASSWORD`), so the cache does not extend the trust domain —
  it extends retention within it, and `cache_clear()` bounds it on password change.
- Wrong-password decrypt attempts derive a *different* key (different cache key) and
  pay full KDF every time; a repeated identical guess is served faster but learns
  nothing new (it already failed). No new timing oracle over the status quo.
- Cache entries are process-local (never persisted, never logged). `functools.lru_cache`
  is thread-safe; races degrade to duplicate derivation, never corruption.

## Verification

- `Tests/Utils/test_config_encryption.py::TestDerivedKeyMemoization` counts scrypt
  invocations across encrypt→decrypt, repeated config round-trips, cache clears, and
  v1/v2 key separation.
- The pre-existing 38 encryption tests and the 58 config-integration tests
  (`Tests/test_config_*.py`) pass unchanged.
