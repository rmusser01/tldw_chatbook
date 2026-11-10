---
id: TASK-40140
title: "Re-encrypt the Personal Context profile after a restore so an old archive's key cannot read the live database"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- security
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Personal_Context/repository.py
- tldw_chatbook/Personal_Context/crypto.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40135
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** Decision 4 installs a restored profile key as it is, and files re-encryption as this separate follow-up.
- The profile `encryption_key` never rotates. Even the first link keeps it (`Personal_Context/repository.py:2051-2055`).
- So anyone holding an old archive and its password, plus a later copy of the live database, can read the later records.
- The only full-rewrite path today is the first-link rebaseline (`repository.py:1824-2621`).

**Approved shape (spec Q2 option B).**
- Re-encrypt every row under a fresh `encryption_key` and bump `key_version`.
- Keep the integrity key and link state. Option C, replacing the server-shared integrity key too, was rejected.
- Store the new key without overwriting the old account (ADR-126 decision 8).
- Delete the old account only after the new one verifies, and only if no other database or retained restore copy names it. Restores stamp fresh accounts (TASK-40135), but a user's own manual copy of the file can still share one.

**Open question.** Whether re-encryption runs automatically after a restore or as a Settings action is for planning. Bring it to the owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a restore that installs a profile key, the profile can be re-encrypted under a fresh encryption key; afterwards every record reads, and the restored key no longer decrypts any row of the live database (red first)
- [ ] #2 An interruption at any point leaves a database that one stored key opens (old rows with the old key, or new rows with the new key), never one that no stored key opens; a test interrupts at each step
- [ ] #3 The integrity key, `key_account` semantics and link state are unchanged by re-encryption
- [ ] #4 The new key goes into a new keyring account without overwriting the old one, and the old account is deleted only after the new one verifies and only if no other database or retained restore copy names it
<!-- AC:END -->
