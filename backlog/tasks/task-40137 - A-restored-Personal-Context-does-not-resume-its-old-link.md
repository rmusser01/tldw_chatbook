---
id: TASK-40137
title: "A restored Personal Context does not resume its old link"
status: To Do
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
- security
priority: medium
references:
- tldw_chatbook/app.py
- tldw_chatbook/Personal_Context/link_service.py
- tldw_chatbook/Personal_Context/link_key_custody.py
- tldw_chatbook/Sync_Interop/recovery.py
- tldw_chatbook/Backup_Recovery/activation.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies:
- TASK-40134
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** A restored device must not act as the old device towards the server (spec constraint 1; decision 5).
- **Link state travels.** It lives in `tldw_chatbook_sync_state.db`, which backup already captures as `runtime.sync_state` (`Sync_Interop/recovery.py:33`, `:79-90`). That includes the `personal_context_link_state` device id, key ids and cursors.
- **Link keys do not.** The device wrapping key, staged integrity keys and dataset staging keys live only in the OS keyring (`Personal_Context/link_key_custody.py:19-23`, `:205-248`, `:401-547`).
- **What restart does.** It resumes sync from that state plus the staging key (`app.py:2625-2662`).
  - On the same machine, a restored profile could resume with stale cursors and outbox.
  - On another machine, it fails with a raw `personal_context_staging_key_unavailable`.

**Approved design.**
- **Which restores are covered.** Every replace restore, not only one that publishes the repository.
  - An automation-only restore publishes `runtime.sync_state` (the "automation" group, `Backup_Recovery/data_groups.py:172-177`) and can rewind `personal_context_link_state` under a live repository.
  - Every restore binds all activation-required owners (`replacement.py:2585-2587`), so the sync gate from TASK-40134 fires for both kinds. The gate is `execution_allowed` on `personal_context.repository` and `runtime.sync_state` (`activation.py:224-227`), like the precedents at `app.py:4717` and `app_service_wiring.py:3348`. This task pins it for both.
- **Force a fresh link.** The review clears that server profile's Personal Context link state, so the next **Link to home server** is a fresh reviewed first link.
- **Fix the cross-machine relink.** A fresh link on a new machine fails today with `personal_context_sync_profile_binding_conflict`.
  - The restored `sync_profile_state.device_id` (`Sync_Interop/recovery.py:39`) still names the old device, and `_seed_sync_profile` refuses a different one (`link_service.py:1014-1028`).
  - That row is shared with Notes and Chat sync (`workspace_scope=None`), and `_seed_sync_profile` carries its `dataset_cursors` forward (`link_service.py:1029`).
  - The rebind must leave Notes and Chat sync state for that profile unchanged. If the shared row cannot be rebound that way, stop and bring it to the owner.
  - What Notes and Chat sync themselves do after a cross-machine restore is investigated separately.
- **Old device registration.** It stays on the server, and Chatbook cannot unregister it (User Guide `personal-context-profile.md:335`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personal Context sync stays off (no outbox dispatcher, no dataset staging key) until review, after both kinds of replace restore: one that publishes the repository, and an automation-only one that publishes only `runtime.sync_state` under a live repository; a test pins each
- [ ] #2 Reviewing either kind of restored generation clears that server profile's Personal Context link state, so the next Link to home server runs a fresh reviewed first link
- [ ] #3 A fresh first link after a cross-machine restore, where the restored sync profile still names the old device, ends bound to the device the server issues for Personal Context instead of failing with `personal_context_sync_profile_binding_conflict`, and the old device id is not kept for Personal Context
- [ ] #4 After that rebind, Notes and Chat sync state for that profile (device id, dataset cursors, outbox) is unchanged; a test pins it
- [ ] #5 No link key (device wrapping key, staged integrity key, dataset staging key) is in any archive, in any credential mode; a test inspects an archive that includes credentials
- [ ] #6 The Settings panel says Personal Context sync is off because the profile was restored, to link again, and to remove the old device on the server if it is lost; the User Guide page says the same
<!-- AC:END -->
