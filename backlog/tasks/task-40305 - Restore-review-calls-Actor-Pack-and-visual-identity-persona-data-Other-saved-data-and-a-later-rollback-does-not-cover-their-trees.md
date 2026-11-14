---
id: TASK-40305
title: "Restore review calls Actor Pack and visual-identity persona data 'Other saved data', and a later rollback does not cover their trees"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- backup
- recovery
- persona
priority: low
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- tldw_chatbook/Backup_Recovery/later_rollback.py
dependencies:
- TASK-39116
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (census stream, TASK-39116). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Restore review shows Actor Pack persona images as "Other saved data", so the user cannot tell what a restore replaces. A later rollback of a restore that changed these trees does not get the persona-tree preservation checks persona artwork gets.

**Evidence.**
- `Backup_Recovery/recovery_service.py:77-104`: `_OWNER_LABELS` names `persona.assets` ("Persona artwork") and `persona.visual_identity_builtin`, but not `persona.actor_packs` or `persona.visual_identity`; `owner_label` (`:107-109`) falls back to "Other saved data".
- `later_rollback.py` handles only `persona.visual_identity_builtin` and `persona.assets` (`:247`, `:293`, `:371`, `:635`, `:667`, `:785`, `:1350`, `:1420`); nothing there, in `publication.py` or in `replacement.py` names `actor_packs`.
- The owner inventory states both limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restore and rollback review name persona.actor_packs and persona.visual_identity with their own labels
- [ ] #2 A later rollback of a restore that changed actor_packs/ or the visual-identity tree restores the prior files byte-equal or refuses with a named reason; a test covers it
<!-- AC:END -->
