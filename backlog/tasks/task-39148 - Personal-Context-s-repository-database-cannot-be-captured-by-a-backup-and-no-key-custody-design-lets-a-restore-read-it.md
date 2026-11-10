---
id: TASK-39148
title: "Personal Context's repository database cannot be captured by a backup, and no key custody design lets a restore read it"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 16:28
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
- [ ] #1 A design (spec or ADR) answers each open question above, and the owner approves it before any capture code lands
- [ ] #2 Every owner decision the design needs is recorded verbatim with its date
- [ ] #3 The implementation is filed as tasks that each deliver one testable outcome: capture of the repository; a restore that reads its records; a restore where the key is unavailable, with no data loss and copy that says what to do; and the detached-layout treatment of the drafts file
<!-- AC:END -->

## Implementation Plan

1. Read the custody code (`Personal_Context/key_protector.py`, `repository.py`, `link_key_custody.py`, `link_service.py`), the backup and restore machinery (`Backup_Recovery/capture.py`, `credentials.py`, `replacement.py`, `isolated_restore.py`, `activation.py`, `detached_layout.py`), ADR-102, ADR-126, the owner inventory, and TASK-39110, TASK-39116 and TASK-40132.
2. Write `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. For each open question, give the code facts with file:line, two or three options with trade-offs, and a recommendation. End with "Decisions for the owner" and "Filed implementation tasks (proposed)".
3. The coordinator brings the decisions to the owner. Record each ruling verbatim with its date (AC#2).
4. After approval, file the implementation tasks listed in the spec (AC#3). Then fold the approved decisions into ADR-102 and ADR-126 notes. No capture code lands before approval (AC#1).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Draft spec (2026-11-10):** `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md` (round 13, branch fu/r13-pcdesign, base dev ae9f9e8d21). It answers every open question in the description with facts from code, options and a recommendation, and ends with eleven decisions for the owner and six proposed implementation tasks.

**Owner rulings recorded so far, verbatim:**
- 2026-11-06 (TASK-39110 AC#2): "Capture it: back up the database and design key custody so a restore can read it — a larger feature that needs its own design task."
- 2026-11-10: "Draft it now."

**Status of the criteria:**
- All three criteria stay unticked: the owner has not yet ruled on the spec's decisions.
- AC#2 needs those rulings recorded here, verbatim and dated.
- AC#3 needs the spec's proposed tasks filed after approval.
- This task writes no product code.
<!-- SECTION:NOTES:END -->
