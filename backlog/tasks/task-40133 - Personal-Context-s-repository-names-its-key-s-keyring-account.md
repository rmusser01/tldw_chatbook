---
id: TASK-40133
title: "Personal Context's repository names its key's keyring account"
status: To Do
created_date: 2026-11-10 17:13
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
  - New databases record their account at creation.
- **Lookup.** The repository looks the key up by the stored account, or by the path-derived account while the reference is empty.
- **Authenticate before use.** It uses a key only after that key authenticates this database's manifest head. It never compares `key_version` to decide, because `key_version` starts at 1 (`key_protector.py:48`) and only a link bumps it, so unrelated keys usually share it.
- **Missing or failing account.** A missing, malformed or non-authenticating account leaves the profile locked, and nothing creates a key. ADR-102: "Missing or changed key protectors lock the profile and never trigger silent replacement."
- **Authenticate before change or delete.** Two operations change or delete an account: the first link's key replace (`repository.py:2612-2618`) and Remove local profile (`:4994-4997`), which today deletes whichever account was loaded. Both act only on the account this database names, and only after its key has authenticated this database.
  - Restores keep retained copies on separate accounts.
  - Known ceiling: a user's own manual copy of the database file shares its key account.

**Catalog caveat.** The repository migrates in place with `ALTER TABLE` (`repository.py:636-733`), so a fresh v9 database and a migrated one can differ textually. Backup validation compares the exact SQL catalog, so every reachable v9 layout has to be known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A v8 database migrates to v9 and, the first time its path-derived account's key authenticates it, records that account in `key_account`; it lists the same records before and after, and no keyring entry is created, changed or deleted
- [ ] #2 A v8 database whose key is not found at its current path stays locked and unstamped, and opens once it is moved back to its original path
- [ ] #3 A newly created profile records its key account in `profile_meta.key_account`
- [ ] #4 A v9 database moved to another path on the same keyring opens and lists its records; the same move of a v8 database stays locked (red first)
- [ ] #5 The repository uses a stored account's key only after it authenticates this database's manifest. An account that is malformed, names no keyring entry, or holds a key that does not authenticate (including a different key with the same `key_version`) leaves the profile locked, with no key created, changed or deleted
- [ ] #6 The first link's key replace and Remove local profile change or delete only the account this database names, and only after that account's key authenticated this database; a test shows another database's account is untouched
- [ ] #7 Every v9 schema layout the repository can produce (fresh, and migrated from each earlier version) is listed in one place, and a test pins each list entry against what the repository actually builds
<!-- AC:END -->
