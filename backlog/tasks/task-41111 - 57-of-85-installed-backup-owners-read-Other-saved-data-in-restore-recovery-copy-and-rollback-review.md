---
id: TASK-41111
title: "57 of 85 installed backup owners read 'Other saved data' in restore, recovery-copy and rollback review"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-15 19:05
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
- [x] #1 Every installed owner a user can see in a review has its own label; owners that never appear in a review are listed in the notes with the reason
- [x] #2 An Architecture pin fails when a new owner is installed without a label, shown red with one label removed
<!-- AC:END -->

## Implementation Plan

1. Complete `recovery_service._OWNER_LABELS` (the one label table `owner_label` reads; the registry carries no names) with a user-facing name for every owner `owner_registry.install_adapters()` installs, plus `recovery.credentials`, which capture adds to every include-mode archive.
2. Pin in `Tests/Architecture/test_backup_owner_inventory.py`: every installed owner (taken from the registry, not a list) has its own distinct label; show it red with one label removed.

## Implementation Notes

**Summary.** `recovery_service._OWNER_LABELS` now names every owner `owner_registry.install_adapters()` installs (85), plus `recovery.credentials`, the credential material include-mode capture adds to the archive. `owner_label` still falls back to "Other saved data", now only for an owner this build does not install (an archive from a newer version).

**Decisions.**
- **Where the labels live.** The registry holds adapters with no display name, and no other owner-name table exists (`data_groups.BACKUP_GROUPS` names groups, not owners). So the existing `_OWNER_LABELS` table stays the one source, and the pin takes its owner set from the registry. Putting a label on every adapter class would touch about 25 modules (Personal_Context among them) for the same result.
- **Which owners reach a review.** All of them. The archive summary's **Contents** line lists every owner in `doc.owners`, and capture writes an owner row for every inventory item, including excluded and unused ones (runtime locks, caches, scratch folders). Dependency groups and portable relocation name the owners of captured files. So no installed owner is left unlabelled, and none needs listing as "never seen".
- Labels are user-facing names, distinct from each other (the pin checks both), for example `tts.voices` "Saved voices", `tts.profile_store` "Voice profiles", `ui.themes` "Themes", `chat.prompts` "Prompt templates", `personal_context.repository` "Personal Context profile", `runtime.config_lock` "Settings lock file".

**Tests and red/green evidence** (load 53-55 on 18 CPUs):
- `Tests/Architecture/test_backup_owner_inventory.py::test_every_installed_backup_owner_has_its_own_review_label` (lane: architecture-guards, whole file). Red before the labels: "Left contains 58 more items, first extra item: 'actor_packs.import_staging'" (57 owners plus `recovery.credentials`). Green after. Negative control with the `ui.themes` label removed: `assert ['ui.themes'] == []`; restored.
- TASK-40305's `test_actor_pack_and_reaction_trees_have_their_own_review_labels` still passes.

**Files.** `tldw_chatbook/Backup_Recovery/recovery_service.py`, `Tests/Architecture/test_backup_owner_inventory.py`.

**Not changed.** Restored profiles' **Needs setup** list renders pending owners from their ids (`tts / profile store`), not through `owner_label`; that list is not one of the reviews this task names (proposed follow-up).
