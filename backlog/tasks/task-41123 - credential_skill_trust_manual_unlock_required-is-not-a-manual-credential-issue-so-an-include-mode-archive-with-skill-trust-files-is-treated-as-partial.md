---
id: TASK-41123
title: "credential_skill_trust_manual_unlock_required is not a manual credential issue, so an include-mode archive with skill trust files is treated as partial"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:05
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
- [x] #1 A test captures an include-mode archive with a skill trust file and records whether a replace restore of it is refused as partial
- [x] #2 If it is, the skill-trust disclosure is classified as manual (still disclosed and acknowledged) and the test passes, red before the change
<!-- AC:END -->

## Implementation Plan

1. Reproduce: a scratch-profile child creates a skill with a trust store, captures an include-mode encrypted backup of the Tools group, inspects it and asks for a replace restore; record the consistency and whether the plan is refused with `partial_replacement`.
2. If refused, add `credential_skill_trust_manual_unlock_required:` to `MANUAL_CREDENTIAL_ISSUES` (capture and archive_writer read it); the disclosure is still acknowledged at capture and listed in the report.

## Implementation Notes

**Reproduced: yes, it was refused.** A scratch-profile child created a skill with a trust store (`SkillTrustService.unlock_with_passphrase` + `bootstrap_trust`), made an encrypted include-mode backup of the Tools group through `RecoveryService` (acknowledging the capture's credential review), inspected it and asked for a replace restore of the same group. Before the change the summary read `consistency: partial` and `preview_restore` raised `partial_replacement`, exactly TASK-40135's Personal Context shape.

**Fix.** `credential_policies.MANUAL_CREDENTIAL_ISSUES` now lists `credential_skill_trust_manual_unlock_required:` beside `credential_manual_recovery_required:` and `credential_personal_context_link_keys_excluded:`. `capture.py` and `archive_writer.py` both read that tuple, so the disclosure goes to the archive's report lines instead of its consistency issues, and an archive whose only issues are these stays `coherent`. The capture review still requires the disclosure to be acknowledged (`process_credentials` is unchanged).

**Test and red/green evidence** (load 53-56 on 18 CPUs):
- `Tests/Backup_Recovery/test_skills_recovery_controls.py::test_include_mode_archive_with_skill_trust_files_is_not_partial` (lane: pr-fast-lane, whole file). It asserts the trust disclosure was acknowledged at capture, `(consistency, refusal) == ('coherent', None)`, a replace plan comes back, and the disclosure is in the archive's report.
- Red before the fix: `AssertionError: ('partial', 'partial_replacement')`. Green after.
- Serial time about 48 s at load 55 (the child prepares the required stores, backs up, inspects and plans; there is no restore). It is the longest node in that file; the coordinator may prefer to move it.
- First attempts failed on setup, not the product: without the required stores the backup review never settled (`missing_identity`, `shared_identity_unavailable`), so the child now runs TASK-40135's store preparation in a subprocess.

**Files.** `tldw_chatbook/Backup_Recovery/credential_policies.py`, `Tests/Backup_Recovery/test_skills_recovery_controls.py`.

**Review r15br-rev1 (fixed).** The node's fixed `_run(..., timeout=150)` failed twice under `-n 4` (114.7 s serial at load 75-88). It now runs `_run`'s scratch layout itself and waits through `Tests/Backup_Recovery/test_config_sibling_capture._wait_while_busy`: it fails once the child's process tree is idle for 60 s, with an 840 s wall backstop and `@pytest.mark.timeout(900)` over the lane's 180 s. Under `-n 4` at load about 89 it passed in 150 s. In that run, five older nodes in the file failed their fixed 45 s `_run` budgets; they are not this task's.
