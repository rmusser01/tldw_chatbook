---
id: TASK-40133
title: "Personal Context's repository names its key's keyring account"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 17:13
updated_date: '2026-11-13 17:30'
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Personal_Context/repository.py
- tldw_chatbook/Personal_Context/key_protector.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-39148
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** The profile key lives in the OS keyring under the account `"personal-context:" + sha256(absolute db path)` (`Personal_Context/repository.py:364-367`, `key_protector.py:28`). The account comes from where the file sits; nothing in the database names it.
- A database at another path finds no key and is locked (`repository.py:380-387`). That covers a moved profile, and a restore onto another machine, OS account or data path.
- Backup can capture and remap a keyring key only through a reference stored in the data. The citation key is the precedent (`Backup_Recovery/credential_policies.py:16-31`, `credentials.py:758-772`).
- ADR-126 decision 8 forbids a restore overwriting a keyring entry (`backlog/decisions/126-complete-local-backup-and-recovery.md`, "Never overwrite shared keyring entries"). So a restored database must be able to point at a fresh account (spec Q2 and Q4).

**Approved design (decision 2; spec Q2, "What re-wraps the key").** Schema v9 adds a non-secret `profile_meta.key_account`; empty means the legacy path-derived account.
- **Stamping.** Migrations run before the key is loaded (`repository.py:373-378`, `:630-733`), so the migration only adds the empty column. The repository writes the reference the first time the path-derived account's key authenticates this database. Consequences:
  - A v8 database found at a moved path stays unstamped and locked, and it opens again once moved back.
- **Lookup.** The repository looks the key up by the stored account, or by the path-derived account while the reference is empty.
- **Authenticate before use.** It uses a key only after that key authenticates this database's manifest head. It never compares `key_version` to decide, because `key_version` starts at 1 (`key_protector.py:48`) and only a link bumps it, so unrelated keys usually share it.
- **Missing or failing account.** A missing, malformed or non-authenticating account leaves the profile locked, and nothing creates a key. ADR-102: "Missing or changed key protectors lock the profile and never trigger silent replacement."
- **New account per profile.** Today two paths call `load_or_create` on the path account, which adopts any existing entry (`key_protector.py:144-148`): creating a database (`repository.py:374-378`) and Start fresh (`reinitialize_destroyed_profile`, `:1703`, from `service.py:606-635`).
  - After a replace restore, that lets a new profile adopt the account the retained original names. A later link, Remove, or failed re-initialisation would then break the rollback. A failed re-initialisation deletes the path account even when it only loaded it (`repository.py:1752-1754`).
  - Instead, creating or re-creating a profile writes its key to a **new, unused** account (set-if-absent), recorded in the same `profile_meta` row. It never adopts an existing entry, and failure cleanup deletes only an account it created.
  - A database with no profile yet holds no key, and has nothing to authenticate.
- **Authenticate before change or delete.** Two operations change or delete an account: the first link's key replace (`repository.py:2612-2618`) and Remove local profile (`:4994-4997`), which today deletes whichever account was loaded. Both act only on the account this database names, and only after its key has authenticated this database.
  - Remove records the account it removes in `profile_meta`. The secure-removal retry (`service.py:1952-1963` → `repository.py:4995`) then deletes exactly that account.
  - Restores keep retained copies on separate accounts.
  - Known ceiling: a user's own manual copy of the database file shares its key account.

**Catalog caveat.** The repository migrates in place with `ALTER TABLE` (`repository.py:636-733`), so a fresh v9 database and a migrated one can differ textually. Backup validation compares the exact SQL catalog, so every reachable v9 layout has to be known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A v8 database migrates to v9 and, the first time its path-derived account's key authenticates it, records that account in `key_account`; it lists the same records before and after, and no keyring entry is created, changed or deleted
- [x] #2 A v8 database whose key is not found at its current path stays locked and unstamped, and opens once it is moved back to its original path
- [x] #3 Creating or re-creating a profile (the first profile, or Start fresh) writes its key to a new, unused keyring account (set-if-absent) and records it in `profile_meta.key_account`; it never adopts an existing entry, a database with no profile holds no key, and failure cleanup deletes only an account it created
- [x] #4 A v9 database moved to another path on the same keyring opens and lists its records; the same move of a v8 database stays locked (red first)
- [x] #5 The repository uses a stored account's key only after it authenticates this database's manifest. An account that is malformed, names no keyring entry, or holds a key that does not authenticate (including a different key with the same `key_version`) leaves the profile locked, with no key created, changed or deleted
- [x] #6 The first link's key replace and Remove local profile change or delete only the account this database names, and only after that account's key authenticated this database; Remove records the account it removes, and the secure-removal retry deletes only that account; a test shows another database's account is untouched
- [x] #7 With the path-derived account holding another database's key, Start fresh, then link, then Remove leaves that entry byte-for-byte unchanged
- [x] #8 Every v9 schema layout the repository can produce (fresh, and migrated from each earlier version) is listed in one place, and a test pins each list entry against what the repository actually builds
<!-- AC:END -->

## Implementation Plan

1. `key_protector.py`: add a set-if-absent `create(account)` to the protector protocol, the in-memory protector and the keyring protector (the passphrase protectors stay out, decision 11).
2. `repository.py`: schema v9. Fresh `profile_meta` gains `key_account TEXT NOT NULL DEFAULT ''`; the v8 -> v9 step adds the same column (guarded, inside the open transaction). No key is touched by the migration.
3. Open: a database with no profile loads no key. A live profile looks up its key by `key_account`, or by the path-derived account while it is empty; a malformed account, a missing entry or a key that does not authenticate the manifest head leaves it locked. Only after the key authenticates does an empty reference get stamped with the path-derived account.
4. New profiles (provisional, with global scope) and Start fresh write fresh material to a new random account through `create`, record it in `key_account` in the same transaction, and delete only that account if the transaction fails.
5. The first link's key replace and Remove act on the account the database names, read in the same transaction that authenticated its key. Remove keeps the account in `key_account` until the keyring delete succeeds, then clears it; the secure-removal retry deletes only a recorded account.
6. List every reachable v9 catalog (fresh, and migrated from each historical fresh layout v1, v2, v4-v8; v3 never had a constructor) in one module, captured from the real constructors, with a test that rebuilds each from the historical DDL and compares.
7. Tests in `Tests/Personal_Context/test_repository.py` (red first), fakes updated where the protocol changed; run Personal_Context, Sync_Interop PC, and affected UI/App tests.

## Implementation Notes

Schema v9 gives the repository a stored key reference, and every key operation goes through it.

- **Schema.** Fresh `profile_meta` gains `key_account TEXT NOT NULL DEFAULT ''`; the v8 -> v9 step adds the same column (guarded, so re-enterable). Migrations now commit in their own transaction before any key is read (`__init__`), so a locked database is still brought current, just never stamped. Personal Context's migrations are inline in `_inspect_schema`, outside `DB/migrations/` (the README lists the eight schemas it covers), no table or index is added, and `sql_validation.py` has no Personal Context allowlist, so the README / `VALID_TABLES` / index-plan rules have nothing to apply to here; the behavioural rule (one guarded, re-enterable transaction per step) holds.
- **Open.** A database with no profile loads no key (the old open-time `load_or_create` on the path account is gone). A live profile reads its key from `key_account`, or from the path-derived account while it is empty (`RepositoryKeyCustody._named_key_account`); anything else (a `recovery:setup_required:...` marker, garbage) locks without touching the keyring. The key is adopted only after it authenticates the manifest head (`_authenticate_keys`, new native consumer `repository._authenticate_keys`); only then is an empty reference stamped with the path account. A handle opened before its profile existed adopts the named key lazily on first use (`_require_keys` -> `_adopt_named_keys`, read-only, same authentication), which keeps the existing absent -> ready transition across handles (`test_absent_status_invalidates_on_external_wal_setup`).
- **New profiles.** `create_provisional_profile`, `create_profile_with_global_scope` and `reinitialize_destroyed_profile` provision fresh material in `personal-context:profile_<32 hex>` through a new set-if-absent `create()` on the protector protocol (in-memory and keyring; the passphrase protectors stay out, decision 11), record it in the same transaction, and on any failure delete only that account (`_new_profile_key`).
- **Link and Remove.** The first link reads the account inside the transaction that authenticated its key and replaces only that one; the interrupted-link recovery replace does the same. Remove authenticates the loaded key against the manifest inside its transaction, keeps the account in `key_account` on the removed row, deletes it, then clears the record; the secure-removal retry deletes only a recorded account, so a completed removal never deletes whatever later holds that name.
- **Catalogs (AC#8).** `Personal_Context/recovery.py` lists the six reachable v9 catalogs (fresh; migrated from v1/v2, v4, v5/v6, v7, v8), captured from the real constructors. Historical fresh schemas came from the commits that shipped them (v1 ca75629011 ... v8 fd8e1ec4b9; v3 never had a constructor, it is only a migration step). The test rebuilds each historical schema, pins it to those commits by digest, migrates it with the current repository and compares the catalog set with the list.

**Keyring isolation.** Every new test uses the in-process `SecureKeyring` dict backend from `test_key_protector.py` (or `InMemoryProfileKeyProtector`) passed explicitly to `KeyringProfileKeyProtector`; `Tests/conftest.py` pins `PYTHON_KEYRING_BACKEND` to the null keyring, which the protector refuses, so no test can reach the OS keyring.

**Evidence.** On base source (repository/key_protector/native_compatibility restored from HEAD) 15 of the 16 new nodes fail. Most fail on assertions (`assert None == <path account>`, `DID NOT RAISE ProfileLockedError`, keyring dict changed, `assert 8 == 9`); the four stored-account cases, the occupied-account case and the link/Remove case error instead, because the column or `create()` does not exist on base, so those invariants carry the negative controls below. Negative controls on the new code, each red on an assertion and then reverted: no open-time authentication (2 nodes `DID NOT RAISE`), any stored string accepted as an account (2), `create` adopting an occupied account (1), Remove deleting the path account (3), Remove without in-transaction authentication (1), link replacing the path account (2). Contract changes in existing tests: two first-open tests now pin "no key at open" (`test_simultaneous_first_openers_create_no_key_and_one_profile_wins`, `test_schema_initialization_failure_creates_no_key`); `test_bootstrap_does_not_claim_an_empty_profile_is_present` (an empty database is absent, not locked; creation fails instead); the secure-removal retry count is 2, not 3; the barrier-route test reads the stored account.

**Files.** `tldw_chatbook/Personal_Context/{repository.py,repository_keys.py,key_protector.py,native_compatibility.py,recovery.py}` (the key-custody helpers live in the new `repository_keys.py` mixin so `repository.py` stays at its 4,997-line ratchet; `_authenticate_keys` stays in `repository.py` with the other protected native routes); `Tests/Personal_Context/{test_repository.py,test_runtime_policy.py,test_service.py,test_native_profile_barrier_routes.py}`.

**Review r14pca-rev1 fixes (2026-11-13).**
- Remove and the first link now act on the account the row names at that moment only if that account's own material equals the key that authenticated the transaction (`RepositoryKeyCustody._authenticated_account`). A reference rewritten after open (to another database's account) refuses with `ProfileLockedError` and leaves the keyring unchanged. Test: `test_a_reference_rewritten_after_open_is_never_acted_on[remove|link]`; on the previous commit, both cases fail with `DID NOT RAISE`.
- The v8 -> v9 step stamps removed rows with the path-derived account, so a v8 secure removal still pending at upgrade is still owed. Test: `test_upgrade_keeps_a_pending_v8_secure_removal` (on the previous commit, `'' == path account` fails).
- `_new_profile_key` deletes its fresh account only while the database does not name it, so a failure after commit keeps the key. Test: `test_a_failure_after_commit_keeps_the_new_profile_key` (on the previous commit, the key is gone).
- AC#8: the v7 build's own 6 -> 7 step created the rebaseline table without `target_key_record_id`, so a v1-v6 database it migrated has three more layouts (nine in all). The catalog test now also emulates that step, patching the rebaseline constant to the v7 text while migrating each historical fresh schema; on the previous commit, the listed-versus-built comparison fails.

**Review r14pca-rev2 fix (2026-11-13).** The secure-removal retry had become a regression from dev. It deleted whatever well-formed account a removed row named, so a removed row rewritten to another live database's account deleted that profile's key.
- **Removal record.** Remove now stores `removal:<account>:<tag>` in `key_account`, where the tag is HMAC-SHA256(serialized key material, `b"personal-context-removal:" + account`).
  - The tag lives in the existing column, not a new one: `key_account` already means "the account this row's removal still owes", and a new column would change the nine v9 catalogs (AC#8) and need another schema step.
  - The tag is a MAC, never the key, and nothing logs it or puts it in an issue string.
- **Retry.** `_delete_removed_account` deletes only if the entry exists and `load(account)` reproduces the tag.
  - A bare path account (`== _profile_ref`, from the v8 step) keeps dev's behaviour.
  - An absent entry, invalid material or any other value is never deleted; the record is just cleared.
  - The keyring protector gains `exists()`, so a missing entry is told apart from an unavailable keyring, which still raises and keeps the record.
- **Test.** `test_a_removed_row_rewritten_to_another_account_deletes_nothing[bare|retargeted]`: on ab4bdf751d the bare case deletes the other database's key and the retargeted case leaves its record. The honest retry stays covered by `test_link_and_remove_act_only_on_the_account_this_database_names`, now asserting the record shape, and the v8 path case by `test_upgrade_keeps_a_pending_v8_secure_removal`.

