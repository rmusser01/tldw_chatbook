---
id: TASK-40140
title: "Re-encrypt the Personal Context profile after a restore so an old archive's key cannot read the live database"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-14 23:18'
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
- Delete the old account only after the new one verifies, and only if no retained restore copy names it. Restores stamp fresh accounts (TASK-40135). A user's own manual copy of the file can still share one, which is the spec's known ceiling.

**Open question.** Whether re-encryption runs automatically after a restore or as a Settings action is for planning. Bring it to the owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a restore that installs a profile key, the profile can be re-encrypted under a fresh encryption key; afterwards every record reads, and the restored key no longer decrypts any row of the live database (red first)
- [x] #2 An interruption at any point leaves a database that one stored key opens (old rows with the old key, or new rows with the new key), never one that no stored key opens; a test interrupts at each step
- [x] #3 The integrity key, `key_account` semantics and link state are unchanged by re-encryption
- [x] #4 The new key goes into a new keyring account without overwriting the old one, and the old account is deleted only after the new one verifies and only if no retained restore copy names it
- [x] #5 Record versions that no key can read are counted before the restored-profile review is confirmed ("N damaged records can't be read and will be removed"); on confirm, re-encryption rewrites the readable rows and removes the damaged ones in the same transaction, crash-safe at the same four points; a damaged head version and a damaged historical version are each tested (owner decision 2026-11-14)
<!-- AC:END -->

## Implementation Plan

1. `RepositoryKeyCustody.reencrypt_profile(keep_old_account=...)` (the key-custody mixin; `repository.py` is at its line budget): refuse mid-link; authenticate the current key and account; put new material (fresh encryption key, same integrity key, `key_version` + 1) into a new, unused account; in one transaction decrypt-authenticate every `encrypted_objects` row, re-encrypt it under a fresh DEK and the new key, name the new account, and authenticate the manifest with the new key; on failure before commit delete only the unnamed new account. After commit, reopen and authenticate with the new account, then delete the old account unless the caller keeps it.
2. `Personal_Context.recovery.reencrypt_profile(repository)`: keeps the old account whenever any recovery copy exists (a retained copy may name it); no trigger calls it until the owner decides (open question below).
3. Tests in `Tests/Personal_Context/test_repository.py`: readable afterwards, old key opens no row, integrity key/tags and other columns unchanged, new account; interruption at each step; retained copy keeps the old account.

## Implementation Notes

**2026-11-13 (round 14, stream pcc).** The re-encryption mechanism is in; what triggers it is still the owner's open question.

- **Mechanism.** `repository_keys.RepositoryKeyCustody.reencrypt_profile` (option B of spec Q2). Every row is decrypted and authenticated under the old key, then sealed under a fresh DEK wrapped by a fresh `encryption_key` with `key_version + 1`; the integrity key, every `integrity_tag` (HMAC over AAD and plaintext) and every other `profile_meta` column stay; link state lives in the sync-state database and is not touched. The new material goes into a new, unused `personal-context:profile_<hex>` account (`install`, set-if-absent; added to `InMemoryProfileKeyProtector` and the protocol), and the one rewrite transaction also names it, so the database always names the key its rows need. A first link in flight (freeze or rebaseline row) refuses. A corrupt row (failing authentication) aborts the whole rewrite, so a profile with such a row cannot be re-encrypted until it is repaired.
- **Old account (AC#4).** Deleted only after a fresh connection authenticates the committed manifest with the new account, and only when `keep_old_account` is false. `recovery.reencrypt_profile` sets it whenever any recovery copy is listed under the default control root (or the listing fails): rollback copies are encrypted, so whether one names the old account cannot be read, and a later rollback to such a copy would otherwise name a deleted account. A user's own manual copy of the file stays the spec's known ceiling.
- **Open question for the owner (NEEDS_CONTEXT, not decided here).** Whether re-encryption runs automatically after a replace restore that installed a key, or as a Settings > Personal Context action (or both). Nothing calls `recovery.reencrypt_profile` yet.

**Evidence.** The six new tests fail on base with `AttributeError` (no `reencrypt_profile`). Negative controls, each red and reverted: naming the new account in its own transaction before the rewrite (`before_rewrite` and `mid_rewrite` interruption cases failed: the database named a key its rows did not use), and skipping the row rewrite (`test_reencryption_leaves_the_restored_key_reading_no_row` failed: the old key still opened rows). `Tests/Personal_Context` in full: 673 passed (`-n 4`).

Files: `tldw_chatbook/Personal_Context/{repository_keys.py,key_protector.py,recovery.py}`, `Tests/Personal_Context/test_repository.py`.

**Owner decision, 2026-11-13 (AskUserQuestion, relayed by the coordinator), on the open question:** "As part of the restored-profile review (Recommended)". The option as put to the owner: "When the user confirms 'Review restored profile', it re-encrypts in the background. One step, and it covers every restore that someone reviews."

**2026-11-13 (round 14, pcc round 2): wired.**
- **Service.** `PersonalContextService.reencrypt_after_restore` runs `recovery.reencrypt_profile` on the open repository under the lifecycle lock.
- **Panel.** Confirming **Review restored profile** in Settings > Personal Context notifies "Re-encrypting the restored profile under a new key…". On the panel's mutation worker it then re-encrypts first and approves the review second. On success it shows "Restored profile reviewed and re-encrypted under a new key."
- **Failure.** It approves nothing, so the review stays offered and can be run again. The rewrite is one transaction, so the profile stays readable. The panel's content-free failure copy is shown.
- **Tests.**
  - `Tests/UI/test_settings_personal_context.py::test_restored_profile_review_reencrypts_before_it_approves[reencrypted|reencrypt_failed]`. Red with the panel at the previous head: the first notice was "Restored profile reviewed.".
  - `Tests/Personal_Context/test_repository.py::test_the_review_trigger_reencrypts_through_the_service`, which uses the real service and repository. Red before the method existed: `AttributeError`.
- **Docs.** The User Guide's restore paragraph says what confirming does and what a failure leaves.

**Owner decision, 2026-11-14 (AskUserQuestion, relayed by the coordinator), on review r14pccB's finding that one damaged row blocks the review for good:** "Drop them, after showing the count (Recommended)". The option as put to the owner: "Before confirming, the review says 'N damaged records can't be read and will be removed'. Re-encryption then rewrites only the readable ones. Nothing readable is lost, since the damaged records already can't be read by any key." AC#5 was added for it.

**2026-11-14, review r14pccB fixes.**
- **Damaged records (AC#5).**
  - **Count.** `RepositoryKeyCustody.damaged_record_versions()` counts stored record versions that fail authentication. The count is read-only and exposed as `PersonalContextService.damaged_record_count()`.
  - **Display.** The panel's load worker shows "N damaged records can't be read and will be removed" under the review. N counts stored versions, so a damaged older version counts too.
  - **Confirm.** Confirming passes N as `expected_damaged`.
  - **Rewrite.** `reencrypt_profile(expected_damaged=...)` skips unreadable record rows during the rewrite and refuses if their number differs from N. In the same transaction it removes them with their head (so the record goes, as it already read under no key), queued envelopes and quarantine notes. Readable history stays.
  - **Other objects.** A damaged manifest, scope, proposal or local body still refuses.
  - **Tests.** `test_reencryption_removes_the_damaged_versions_the_review_counted[head|history]`, which also checks that a wrong count refuses and changes nothing. The four interruption cases now also run with a damaged version (8 cases): before the commit the damaged row remains, after it the row is gone, and one stored key always opens the database. Panel test: `test_restored_profile_review_shows_and_removes_the_damaged_count`.
  - **Negative control.** Skipping the removal turned 4 of those red; reverted.
- **Old account (minor).** Neither of the two suggested paths is taken: the old account is kept while any recovery copy exists, and is never deleted later.
  - **Why the invariant is not relied on.** "A retained copy never names the live account" holds for copies taken during the restore that stamped the live account. It breaks after a rollback: rolling back restores the original's bytes, which name the original account, and that restore's own rollback copy names it too.
  - **Why deleting on copy removal was not built.** It would add a new durable record and a hook in copy deletion.
  - **Why keeping it is safe.** The kept account holds only the old encryption key, which reads no row once the profile is re-encrypted, so keeping it does not weaken this task's goal.
  - **Proposed follow-up:** delete kept accounts when the last recovery copy is deleted.

