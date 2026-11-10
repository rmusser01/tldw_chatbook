---
id: TASK-39148
title: "Personal Context's repository database cannot be captured by a backup, and no key custody design lets a restore read it"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:18
labels:
- follow-up
- backup
- recovery
- personal-context
- design
priority: medium
references:
- tldw_chatbook/Backup_Recovery/config_adapter.py
- tldw_chatbook/Backup_Recovery/detached_layout.py
- tldw_chatbook/Personal_Context/repository.py
- tldw_chatbook/Personal_Context/key_protector.py
- tldw_chatbook/Personal_Context/link_key_custody.py
- tldw_chatbook/Personal_Context/interview_launch.py
- backlog/docs/backup-recovery-owner-inventory.md
- Docs/User_Guide/settings/personal-context-profile.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-39110
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (backup stream, TASK-39110 on round-12 branch fu/r12-backup). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08. Line numbers are dev's; round-12 code is named by symbol.

**The ruling.** Owner decision (2026-11-06), recorded in TASK-39110's notes for its AC#2 on round-12 branch fu/r12-backup: "Capture it: back up the database and design key custody so a restore can read it — a larger feature that needs its own design task." TASK-39110's AC#2 stays open for this task.

**Where it stands.**
- Round 12 (`_LocalSQLite` in `Backup_Recovery/config_adapter.py`, fu/r12-backup) declares `personal_context.repository` (`tldw_chatbook_personal_context.db` and its `-wal`, `-shm` and `-journal`) as an explicit `unsupported` row. A present file keeps the preview incomplete, and capture is refused with `personal_context_repository_capture_unsupported`. The interview drafts are intentionally excluded.
- The repository's records are encrypted with a profile key. By default `PersonalContextRepository` takes it from `KeyringProfileKeyProtector` (`Personal_Context/repository.py:368`), a verified secure OS keyring entry. An archive that holds only the database cannot be read.

**Open questions for the design.** These are context, not a prescribed design.
- Does the archive carry a wrapped copy of the profile key, under the backup's own credential? Or does a restore need the same keyring entry or passphrase? `PassphraseProfileKeyProtector` (`key_protector.py:209`) writes its salt, nonce and ciphertext into one bundle file at a path its caller chooses (`_store`, `:234-248`). Where that bundle lives, and whether the archive holds it, is open.
- Restore onto another machine or another OS account: the keyring entry does not travel. What re-keys or re-wraps the records? What does the user see when the key is missing?
- Link and sync key custody. The device's RSA wrapping key lives only in the OS keyring (`KeyringPersonalContextWrappingKeyProvider`, `link_key_custody.py:205`). So do staged integrity keys (`KeyringPersonalContextLinkKeyCustodian`, `:401`). Are these keys, and the server-trusted sync state, restored as they are, re-enrolled, or excluded? A restored device must not be able to impersonate the old one.
- Rollback and replacement: a restored database whose key no longer matches the keyring.
- Should the interview-drafts exclusion also join the detached-layout preview's fixed owner list (`Backup_Recovery/detached_layout.py:931-946`, beside `notes.sync_state`)? Today a detached source that holds drafts is refused as unknown content.
- TASK-39116 (To Do) reviews the Personal_Context producer rows that sit in the `unsupported` default; its decisions and this design must agree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design (spec or ADR) answers each open question above, and the owner approves it before any capture code lands
- [x] #2 Every owner decision the design needs is recorded verbatim with its date
- [x] #3 The implementation is filed as tasks that each deliver one testable outcome: capture of the repository; a restore that reads its records; a restore where the key is unavailable, with no data loss and copy that says what to do; and the detached-layout treatment of the drafts file
<!-- AC:END -->

## Implementation Plan

1. Read the custody code (`Personal_Context/key_protector.py`, `repository.py`, `link_key_custody.py`, `link_service.py`), the backup and restore machinery (`Backup_Recovery/capture.py`, `credentials.py`, `replacement.py`, `isolated_restore.py`, `activation.py`, `detached_layout.py`), ADR-102, ADR-126, the owner inventory, and TASK-39110, TASK-39116 and TASK-40132.
2. Write `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. For each open question, give the code facts with file:line, two or three options with trade-offs, and a recommendation. End with "Decisions for the owner" and "Filed implementation tasks (proposed)".
3. The coordinator brings the decisions to the owner. Record each ruling verbatim with its date (AC#2).
4. After approval, file the implementation tasks listed in the spec (AC#3). Then fold the approved decisions into ADR-102 and ADR-126 notes. No capture code lands before approval (AC#1).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** This was a design-only task; it changed no product code. I read the custody code, the backup and restore machinery, ADR-102, ADR-126, and TASK-39110, 39116 and 40132.
- The spec `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md` answers each open question in the description (Q1–Q6). Each answer gives facts from code with file:line, two or three options with trade-offs, and a recommendation.
- The spec was drafted on branch fu/r13-pcdesign from base dev ae9f9e8d21, and the owner then approved it.

**Owner rulings, verbatim:**
- 2026-11-06 (TASK-39110 AC#2): "Capture it: back up the database and design key custody so a restore can read it — a larger feature that needs its own design task."
- 2026-11-10: "Draft it now."
- 2026-11-10 (AskUserQuestion): "Approve all 11". The option text was: "Record approval of the 11 recommendations as written and file the implementation tasks: key reference (schema v9), capture, readable restore, restore without a key, no link resume, detached-layout treatment."

**The 11 approved decisions, as the spec words them:**
1. Carry the Personal Context profile key as a credential record inside the encrypted archive (only when a backup includes credentials, which already forces encryption), rather than a separate Personal Context passphrase or no key at all.
2. Give the repository a stored key reference (schema v9 `profile_meta.key_account`), so restore installs the key into a fresh keyring account and never overwrites an existing one (ADR-126 decision 8), rather than overwriting the path-keyed entry after escrowing it (which needs an ADR-126 amendment) or refusing replace-restore.
3. When a backup does not include credentials, capture the database without its key and say so in the preview and restore plan, rather than refusing Personal Context capture.
4. Install the restored key as it is and file re-encryption after restore as a separate follow-up, rather than re-encrypting every record during restore.
5. Never capture the device wrapping key, staged integrity keys or dataset staging keys, and never resume a restored link; Personal Context sync stays off until a fresh reviewed first link, rather than restoring or re-wrapping link keys.
6. Keep Personal Context locked in isolated restored profiles (they run with a null keyring), with its key retained in the encrypted restore copy, rather than giving isolated profiles keyring access.
7. Refuse Personal Context capture while a first-link freeze or rebaseline marker exists, rather than capturing mid-link state.
8. Include device-only records in the backup, as Export recovery copy already does, and record that ADR-102's "never leave Chatbook" means never synced or sent to a model, rather than filtering them out of the capture.
9. Put the repository in a new selectable "Personal Context" backup group, rather than in the existing "Settings" group.
10. Add the interview drafts to the detached-layout fixed owner list as excluded, and refuse the repository there by name, rather than refusing both as unknown content.
11. Leave `PassphraseProfileKeyProtector` unshipped and out of this work (it has no production caller), rather than adopting it as the backup key carrier.

**Filed tasks (AC#3).** Each delivers one testable outcome:
- TASK-40133: key reference (schema v9). This is the foundation the others build on.
- TASK-40134: capture the repository database.
- TASK-40135: a replace restore makes the restored records readable.
- TASK-40136: a restore without a usable key keeps the data and says what to do.
- TASK-40137: a restored profile does not resume its old link. This includes the `personal_context_sync_profile_binding_conflict` rebind.
- TASK-40138: the detached-layout treatment, by name.

Follow-ups:
- TASK-40139 (security): investigate whether a restored `runtime.sync_state` lets Notes and Chat sync resume as the old device.
- TASK-40140: re-encrypt the profile after a restore (decision 4).

**ADR notes.** Added dated notes, "2026-11-10 note (TASK-39148, owner-approved)":
- ADR-102: decision 8's reading of "never leave Chatbook".
- ADR-126: restore never overwrites a keyring entry, and the stored key reference keeps that rule workable.

**Files:**
- the spec;
- `backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md`;
- `backlog/decisions/126-complete-local-backup-and-recovery.md`;
- the eight new task files, TASK-40133 to TASK-40140.
- This task needed no tests, because it is design-only. Preflight runs through the scratch wrapper.
<!-- SECTION:NOTES:END -->
