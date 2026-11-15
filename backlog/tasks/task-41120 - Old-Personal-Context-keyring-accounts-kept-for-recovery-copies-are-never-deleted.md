---
id: TASK-41120
title: "Old Personal Context keyring accounts kept for recovery copies are never deleted"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
- security
priority: low
references:
- tldw_chatbook/Personal_Context/repository_keys.py
- tldw_chatbook/Personal_Context/recovery.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Each restore and re-encryption can leave an old encryption key in the keyring for good. It reads no row of the re-encrypted profile, but it does read the recovery copies, and nothing ever removes it once those copies are gone.

**Evidence.** `repository_keys.py:336-337`: re-encryption deletes the old account only `if not keep_old_account`, and `recovery.reencrypt_profile` (`recovery.py:667-685`) passes `keep_old_account=retained` while any recovery copy exists. TASK-40140's notes (review r14pccB, minor): "the old account is kept while any recovery copy exists, and is never deleted later. … Proposed follow-up: delete kept accounts when the last recovery copy is deleted."
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Deleting the last recovery copy that names a kept Personal Context account deletes that account, and only when no live database names it
- [ ] #2 A test re-encrypts with a recovery copy kept, deletes the copy and checks the account is gone, red before the change
<!-- AC:END -->
