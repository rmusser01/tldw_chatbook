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
- **Which restores force a fresh link.** A restore that publishes `personal_context.repository` or `runtime.sync_state`.
  - An automation-only restore publishes `runtime.sync_state` (the "automation" group, `Backup_Recovery/data_groups.py:172-177`) and can rewind `personal_context_link_state` under a live repository.
  - Every restore binds all activation-required owners in `finalize_candidate` (`publication.py:592-604`, called from `replacement.py:1682` and `isolated_restore.py:260`).
  - So TASK-40134's interim guard already keeps sync off after every restore until review. This task pins the two kinds above, using `execution_allowed` on `personal_context.repository` and `runtime.sync_state` (`activation.py:224-227`), like the precedents at `app.py:4717` and `app_service_wiring.py:3348`.
- **Force a fresh link.** For those restores, the review clears that server profile's Personal Context link state, so the next **Link to home server** is a fresh reviewed first link. A restore that publishes neither owner leaves the link state as it was.
- **Fix the cross-machine relink.** A fresh link on a new machine fails today with `personal_context_sync_profile_binding_conflict`.
  - The restored `sync_profile_state.device_id` (`Sync_Interop/recovery.py:39`) still names the old device, and `_seed_sync_profile` refuses a different one (`link_service.py:1014-1028`).
  - That row is shared with Notes and Chat sync (`workspace_scope=None`), and `_seed_sync_profile` carries its `dataset_cursors` forward (`link_service.py:1029`).
  - The rebind must leave Notes and Chat sync state for that profile unchanged. If the shared row cannot be rebound that way, stop and bring it to the owner.
  - What Notes and Chat sync themselves do after a cross-machine restore is investigated separately.
- **Old device registration.** It stays on the server, and Chatbook cannot unregister it (User Guide `personal-context-profile.md:335`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a restore that publishes `personal_context.repository` or `runtime.sync_state` (including an automation-only restore under a live repository), Personal Context sync stays off (no outbox dispatcher, no dataset staging key) until review; a test pins each kind
- [ ] #2 Reviewing such a restore clears that server profile's Personal Context link state, so the next Link to home server runs a fresh reviewed first link; a restore that publishes neither owner leaves the link state as it was
- [ ] #3 A fresh first link after a cross-machine restore, where the restored sync profile still names the old device, ends bound to the device the server issues for Personal Context instead of failing with `personal_context_sync_profile_binding_conflict`, and the old device id is not kept for Personal Context
- [ ] #4 After that rebind, Notes and Chat sync state for that profile (device id, dataset cursors, outbox) is unchanged; a test pins it
- [ ] #5 No link key (device wrapping key, staged integrity key, dataset staging key) is in any archive, in any credential mode; a test inspects an archive that includes credentials
- [ ] #6 The Settings panel says Personal Context sync is off because the profile was restored, to link again, and to remove the old device on the server if it is lost; the User Guide page says the same
<!-- AC:END -->
