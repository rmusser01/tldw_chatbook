---
id: TASK-41123
title: "credential_skill_trust_manual_unlock_required is not a manual credential issue, so an include-mode archive with skill trust files is treated as partial"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- skills
priority: low
references:
- tldw_chatbook/Backup_Recovery/credential_policies.py
- tldw_chatbook/Backup_Recovery/credentials.py
- tldw_chatbook/Backup_Recovery/capture.py
- tldw_chatbook/Backup_Recovery/archive_writer.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** TASK-40135 found that a disclosure-only credential issue made every include-mode archive with Personal Context `partial`, so `plan_restore` refused it (`partial_replacement`); it fixed that for Personal Context. The skill-trust disclosure has the same shape and was left out.

**Evidence.** `credential_policies.py:90-93`: `MANUAL_CREDENTIAL_ISSUES` holds `credential_manual_recovery_required:` and `credential_personal_context_link_keys_excluded:`. `credentials.py:1134-1147` issues `credential_skill_trust_manual_unlock_required:<id>` for every included `skills` trust file. `capture.py:565-571` treats issues outside the manual list as consistency issues, and `archive_writer.py:185-198` writes a non-manual issue as an incomplete archive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test captures an include-mode archive with a skill trust file and records whether a replace restore of it is refused as partial
- [ ] #2 If it is, the skill-trust disclosure is classified as manual (still disclosed and acknowledged) and the test passes, red before the change
<!-- AC:END -->
