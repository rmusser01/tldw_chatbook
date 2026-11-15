---
id: TASK-41106
title: "The recovery CLI cannot remove newer plugin downloads, so a CLI restore of Tools and skills is refused until the user opens the Backup screen"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- backup
- cli
priority: low
references:
- tldw_chatbook/Backup_Recovery/launcher.py
- tldw_chatbook/Backup_Recovery/recovery_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** `tldw-chatbook recovery restore` is the path for a user who cannot start the app. TASK-40309 added the removal to the Backup screen's review only, and the CLI's refusal sends the user to that screen.

**Evidence.** `recovery_service.py:168-175`: `plugin_fetch_removal_required` says "Review the restore on the Backup screen, which lists them and removes them when you confirm; or restore without the Tools and skills group." The `restore` sub-command (`launcher.py:644-668`) has no flag for the newer download folders.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The recovery CLI can list the newer plugin download folders a restore would remove and remove them on an explicit flag, with the same no-safety-copy warning the screen gives
- [ ] #2 Without the flag the refusal names the flag as well as the Backup screen
- [ ] #3 A CLI test covers refusal, listing and removal
<!-- AC:END -->
