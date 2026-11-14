---
id: TASK-40138
title: "A detached source holding Personal Context files is treated by name"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
priority: medium
references:
- tldw_chatbook/Backup_Recovery/detached_layout.py
- tldw_chatbook/Backup_Recovery/config_adapter.py
- backlog/docs/backup-recovery-owner-inventory.md
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-39148
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** The detached-layout preview (ADR-256 portable imports) refuses any file it does not know as `detached_unknown_content` (`Backup_Recovery/detached_layout.py:1146-1153`). Neither Personal Context database is known (`:1037-1058`).
- So every portable import from a profile that ever started an interview is refused, for a reason the user cannot see.
- The fixed owner list (`:929-956`) already gives `notes.sync_state` an excluded treatment, with its paths in `known` (`:1046-1049`).

**Interview drafts.** They are local-only by policy, and an excluded `_LocalSQLite` yields `intentionally_excluded` rows (`config_adapter.py:693-706`, `:1163-1170`). Their key is keyring-only (`Personal_Context/interview_launch.py:97-113`), so the bytes would be unreadable anywhere else.

**The repository.** Its key cannot come along: detached import is credential-excluded (`detached_layout.py:1`, `:492-493`), and the relocation host runs with a null keyring (`portable_relocation.py:250`). Decision 10 refuses it by name instead.
- If TASK-40134 has landed, its captured adapter is picked up by the detached layout automatically. The refusal must hold either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A detached source holding the interview drafts database and its sidecars is imported with the drafts as an excluded row and a review line saying they stay on the original device, instead of `detached_unknown_content` (red first)
- [ ] #2 A detached source holding the Personal Context repository is refused with `detached_owner_unavailable:personal_context.repository`, and the user sees copy saying its key stays on the original device and to make an encrypted backup with credentials there and restore it with Replace
- [ ] #3 The owner inventory doc's detached-layout notes name both treatments
<!-- AC:END -->
