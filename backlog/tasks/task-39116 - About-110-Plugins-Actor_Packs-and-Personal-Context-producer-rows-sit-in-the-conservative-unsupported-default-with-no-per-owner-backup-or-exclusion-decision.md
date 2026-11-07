---
id: TASK-39116
title: "About 110 Plugins, Actor_Packs and Personal Context producer rows sit in the conservative unsupported default with no per-owner backup or exclusion decision"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- backup
- needs-owner-decision
priority: low
references:
- backlog/docs/backup-recovery-owner-inventory.md
- tldw_chatbook/Plugins/
- tldw_chatbook/Actor_Packs/
- tldw_chatbook/Personal_Context/key_protector.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and on round-11 branch fu/r11-backupinv, whose TASK-39053 notes list these cohorts as "observations for later cohort work". Counts are from the branch's inventory doc; line numbers are dev's.

| Package | Producer rows | In `unsupported` / miscellaneous | Other rows |
|---|---|---|---|
| Plugins | 66 | 61 | 2 `plugins.registry`, 2 memory, 1 process artifact |
| Actor_Packs | 33 | 30 | 3 memory |
| Personal_Context | 19 | 15 | 3 `personal_context.repository`, 1 `personal_context.interview_drafts` |

- `unsupported` is the review's conservative default, not a decision about each owner.
- For Personal Context the decision has a constraint: its database is encrypted with a keyring-held key (`KeyringProfileKeyProtector`, `Personal_Context/key_protector.py:28`, `:127-200`). A backup without the key cannot be read.
- A `PassphraseProfileKeyProtector` that writes a key-bundle file also exists (`:209-300`), so not every custody mode is keyring-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides a cohort for each of the three packages (capture, exclude, or stays `unsupported`), including whether Personal Context key material travels with its database; the decisions are recorded and the doc rows follow them
- [ ] #2 Any owner moved to "exclude" has an installed exclusion that a backup preview shows (precedents: `_DeviceStateExclusion`, `ProjectIndexCacheDeclaration`), with a test red before
<!-- AC:END -->
