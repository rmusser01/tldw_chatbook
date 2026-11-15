---
id: TASK-41111
title: "57 of 85 installed backup owners read 'Other saved data' in restore, recovery-copy and rollback review"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- ux
priority: low
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- Tests/Architecture/test_backup_owner_inventory.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** A user reviewing what a restore or rollback will replace sees "Other saved data" for most categories, so the review cannot tell them that their themes, prompts, skills or voices are involved. TASK-40305 named the Actor Pack and reaction trees only.

**Evidence.** `recovery_service.py:109-111`: `owner_label` returns `_OWNER_LABELS.get(owner, "Other saved data")`. Through run.sh, `owner_registry.install_adapters()` gives 85 owners and 57 have no label, among them `chat.prompts`, `skills`, `ui.themes`, `tts.voices`, `tts.profile_store`, `personas`, `notes.templates`, `mcp.local`, `kanban.local`, `writing.local`, `research.local` and `personal_context.repository`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every installed owner a user can see in a review has its own label; owners that never appear in a review are listed in the notes with the reason
- [ ] #2 An Architecture pin fails when a new owner is installed without a label, shown red with one label removed
<!-- AC:END -->
