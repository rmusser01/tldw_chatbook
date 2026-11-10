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

**Approved design (decision 2).** Schema v9 adds a non-secret `profile_meta.key_account`.
- Migration stamps each existing database with the account that already holds its key, so nothing changes for today's users.
- New databases record their account at creation.
- The repository looks the key up by the stored account instead of by path.
- A missing or malformed account leaves the profile locked and never creates a key. This is ADR-102: "Missing or changed key protectors lock the profile and never trigger silent replacement."

**Catalog caveat.** The repository migrates in place with `ALTER TABLE` (`repository.py:636-733`), so a fresh v9 database and a migrated one can differ textually. Backup validation compares the exact SQL catalog, so every reachable v9 layout has to be known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A v8 database migrates to v9 with `key_account` set to the account that already holds its key; it lists the same records before and after, and the migration creates, changes or deletes no keyring entry
- [ ] #2 A newly created profile records its key account in `profile_meta.key_account`
- [ ] #3 A v9 database moved to another path on the same keyring opens and lists its records; the same move of a v8 database stays locked (red first)
- [ ] #4 A stored account that is malformed, or names no keyring entry, leaves the profile locked with no key created; a test shows it
- [ ] #5 Every v9 schema layout the repository can produce (fresh, and migrated from each earlier version) is listed in one place, and a test pins each list entry against what the repository actually builds
<!-- AC:END -->
