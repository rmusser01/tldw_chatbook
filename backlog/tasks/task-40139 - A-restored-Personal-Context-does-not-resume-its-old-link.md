---
id: TASK-40139
title: "A restored Personal Context does not resume its old link"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 17:13
labels:
- follow-up
- backup
- recovery
- personal-context
- security
updated_date: '2026-11-13 19:50'
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
- TASK-40135
- TASK-40137
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
  - An automation-only restore publishes `runtime.sync_state` (the "automation" group, `Backup_Recovery/data_groups.py:172-177`) and can rewind `personal_context_link_state` under a live repository. This already applies under TASK-40134 alone.
  - Restores that publish the repository exist only once TASK-40135 lands, hence the dependency.
  - Every restore binds all activation-required owners in `finalize_candidate` (`publication.py:592-604`, called from `replacement.py:1682` and `isolated_restore.py:260`).
  - TASK-40134's interim guard keeps sync off after every restore until review. Its review approves only restores that published neither owner. After one that published either, sync stays off with a named reason.
  - This task supplies the missing review for those restores. It pins the two kinds above, using `execution_allowed` on `personal_context.repository` and `runtime.sync_state` (`activation.py:224-227`), like the precedents at `app.py:4717` and `app_service_wiring.py:3348`.
- **Force a fresh link.** For those restores, the review clears that server profile's Personal Context link state and only then approves, so the next **Link to home server** is a fresh reviewed first link. A restore that publishes neither owner leaves the link state as it was.
- **What the review approves.** It approves only Personal Context's own owner, `personal_context.repository`, and never `config` or `runtime.sync_state`.
  - Approval is per owner and generation (`Backup_Recovery/activation.py:351-384`).
  - Every Sync V2 call needs those two shared owners approved (`_sync_call`, `Sync_Interop/server_sync_service.py:85-88`). That includes `LocalFirstSyncService.sync_once`, which syncs Notes, Chat and Personal Context in one call (`local_first_sync_service.py:70-71`).
  - Approving them here would turn Notes and Chat sync back on with the restored device id.
- **How the fresh link gets past `_sync_call`.** The link calls (`link_service.py:170`, `:730` → `server_sync_service.py:710-712`, `:811-813`) also need `config` and `runtime.sync_state` approved.
  - This task follows TASK-40137's ruling on who may approve those owners after a restore, and through what review.
  - If that ruling needs an owner decision not yet recorded, or requires a guard or review that has not shipped, stop and bring it to the owner.
- **Fix the cross-machine relink.** A fresh link on a new machine fails today with `personal_context_sync_profile_binding_conflict`.
  - The restored `sync_profile_state.device_id` (`Sync_Interop/recovery.py:39`) still names the old device, and `_seed_sync_profile` refuses a different one (`link_service.py:1014-1028`).
  - That row is shared with Notes and Chat sync (`workspace_scope=None`), and `_seed_sync_profile` carries its `dataset_cursors` forward (`link_service.py:1029`).
  - The rebind must leave Notes and Chat sync state for that profile unchanged. If the shared row cannot be rebound that way, stop and bring it to the owner.
  - What Notes and Chat sync themselves do after a cross-machine restore is TASK-40137's investigation.
- **Old device registration.** It stays on the server, and Chatbook cannot unregister it (User Guide `personal-context-profile.md:335`).

**Amended 2026-11-12 after TASK-40137's owner ruling ("Rebind review + separate config review").** The spec's plan to rebind Personal Context "without changing Notes and Chat sync state" cannot hold: with no active workspace both share one `sync_profile_state` row. So AC#5 now requires Notes and Chat to move to the new device too, and AC#9 adds the sync-identity review that approves `config` and `runtime.sync_state` and clears the restored identity. The separate `config`-only review is TASK-40711.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a restore that publishes `personal_context.repository` or `runtime.sync_state` (including an automation-only restore under a live repository), Personal Context sync stays off (no outbox dispatcher, no dataset staging key) until review; a test pins each kind
- [x] #2 Reviewing such a restore clears that server profile's Personal Context link state and only then approves `personal_context.repository` (never `config` or `runtime.sync_state`), replacing the named reason left by TASK-40134's guard, so the next Link to home server runs a fresh reviewed first link; a restore that publishes neither owner leaves the link state as it was
- [x] #3 After that review following a cross-machine restore, Notes and Chat sync (`LocalFirstSyncService.sync_once`) is still refused with `sync_activation_required`; a test pins it
- [x] #4 Under TASK-40137's ruling on approving `config` and `runtime.sync_state`, a fresh first link after a cross-machine restore, where the restored sync profile still names the old device, ends bound to the device the server issues for Personal Context instead of failing with `personal_context_sync_profile_binding_conflict`, and the old device id is not kept for Personal Context
- [x] #5 After that rebind, Notes and Chat sync for that profile never use the old device id, cursors or outbox: they run on the newly issued device (Personal Context and Notes/Chat share one `sync_profile_state` row when no workspace is active, TASK-40137's finding); a test pins it
- [x] #6 After the fresh first link, Personal Context sync runs, and Notes and Chat sync never sends the old device id (refused, or rebound per TASK-40137's ruling); a test pins both
- [x] #7 No link key (device wrapping key, staged integrity key, dataset staging key) is in any archive, in any credential mode; a test inspects an archive that includes credentials
- [x] #8 The Settings panel says Personal Context sync is off because the profile was restored, to link again, and to remove the old device on the server if it is lost; the User Guide page says the same
- [x] #9 Per TASK-40137's ruling (owner, 2026-11-12), one reviewed sync-identity step approves `config` and `runtime.sync_state` together and, in the same step, clears the restored server identity for every server profile: device id, dataset and pull cursors, remote heads, Personal Context link state and the pending outbox, with the counts shown before the user confirms; the next sync registers a new device; nothing else approves `runtime.sync_state`; tests pin each part
<!-- AC:END -->

## Implementation Plan

1. Personal Context review (`Personal_Context.recovery`): for a generation whose restore published the repository or `runtime.sync_state` (or whose plan cannot be read), delete the restored `personal_context_link_state` rows first, then approve only `personal_context.repository`; Settings offers the review for that case too, with AC#8's copy.
2. Sync-identity review (`Sync_Interop.recovery`, owner ruling on TASK-40137): `sync_identity_review` returns the counts of what it will clear while a restored generation waits for `config`/`runtime.sync_state`; `approve_sync_identity(shown=...)` recounts, refuses if the counts changed, clears device ids, dataset bindings and cursors, pull cursors, remote heads, Personal Context link state and the unsent outbox (with their receipts) for every server profile (`SyncStateRepository.clear_restored_identity`), then approves both owners. The approver pin names it as the only approver of `runtime.sync_state`.
3. Tests: a scratch child per restore kind (repository; automation-only) drives both reviews, `LocalFirstSyncService.sync_once` and a fresh first link through the existing fakes; a real include-mode archive is scanned for link-key canaries.
4. The review's UI: placement needs the owner (see notes). Owner decision 2026-11-13: Backup & Restore ▸ Restored profiles, beside TASK-40711's Approve settings.

## Implementation Notes

**2026-11-13 (round 14, stream pcc).** Both reviews and their guarantees are in; the sync-identity review has no screen yet, because where it lives is not settled (NEEDS_CONTEXT below). AC#9 stays open.

- **Personal Context review (AC#1, #2, #8).** `recovery._restore_reviews` no longer refuses a restore that published the repository or `runtime.sync_state` (or whose plan cannot be read): when confirmed, `_clear_link_state` deletes every `personal_context_link_state` row of the profile's `tldw_chatbook_sync_state.db` (same directory as the repository), then approves only `personal_context.repository`. Settings shows **Review restored profile** for that case with "Personal Context sync is off because this profile was restored. Choose Review restored profile, then Link to home server again. If the old device is lost, remove it on your server." Sync still needs `config` and `runtime.sync_state`, so it stays off (AC#3 pinned through the real `LocalFirstSyncService.sync_once`: `sync_activation_required`, no request).
- **Sync-identity review (AC#4-#6, #9 mechanism).** `Sync_Interop.recovery.sync_identity_review` / `approve_sync_identity` and `SyncStateRepository.restored_identity_counts` / `clear_restored_identity`. Counts: `device_ids`, `datasets`, `pull_cursors`, `remote_heads`, `personal_context_links`, `pending_outbox`. It clears `device_id`, `dataset_id` and `dataset_cursors` on every `sync_profile_state` row (Notes, Chat and Personal Context share the row without a workspace), every pull cursor, remote head and Personal Context link, and the pending/dispatched outbox with its projection receipts (so those local changes can queue again), then approves both owners for every restored generation that binds them. Changed counts refuse (`sync_identity_counts_changed`) before anything changes. Dataset ids are cleared too, so the next Prepare (`run_v2_dry_run`, `device_id=None`) registers a new device and enrolls again, and a fresh first link seeds the device the server issues without `personal_context_sync_profile_binding_conflict`. `test_no_reviewer_approves_the_shared_sync_owners` now names it as the only `runtime.sync_state` approver.
- **Link keys (AC#7).** The end-to-end restore child (TASK-40135's) seeds this device's keyring with wrapping, staged-integrity and dataset-staging canaries under `tldw_chatbook.personal_context.link`, then reads every member of the real, decrypted include-mode and exclude-mode archives: no canary and no link service name appear, while the captured profile key does appear in the include archive (positive control).

**NEEDS_CONTEXT (owner): where the sync-identity review lives.** AC#9 asks the user to see the counts and confirm. TASK-40137's option A placed it "in Backup & Restore or Settings › Sync"; the ruling did not choose. Options: (a) Settings › Sync, beside Prepare and Manual run (`settings_screen.py` is at its exact line budget, so this needs a waiver); (b) Backup & Restore, after a restore; (c) Settings › Personal Context, beside **Review restored profile**, with copy that names Notes and Chat as well. Until a surface exists, a restored profile keeps all server sync off (fail closed), and the Personal Context link cannot pass `_sync_call`. The User Guide says sync stays off after a restore and never resumes as the old device; it names no review step yet.

**Evidence.** Red on the TASK-40140 commit's source (9493849d40): both new child cases and TASK-40134's three relink cases fail on `ValueError: personal_context_restored_link_state` (the old refusal), and the panel case on the old copy. Negative control on the new code: approving without `clear_restored_identity` fails the child's identity assertions; reverted.

**Tests.** `Tests/Backup_Recovery/test_activation_network_sync.py::test_restored_link_and_sync_identity_are_cleared_before_review_approves[repository|sync_state]` (bound generation; Personal Context review; `sync_once` refused; counts, stale-count refusal, clear and approval; a fresh first link through `PersonalContextLinkService` with the first-link fakes binds `device-1`; then `sync_once` pulls as `device-1` with no cursor and no outbox; ~10 s each); TASK-40134's `test_restored_personal_context_sync_waits_for_its_own_review` and the panel's `test_restored_profile_review_never_claims_sync_resumes` updated to the new contract; the archive scan in `Tests/Backup_Recovery/test_restore_data_groups.py`. AC#6's "Personal Context sync runs" is pinned as the gate opening (`sync_allowed`) plus the fresh link's state; the Personal Context dispatcher itself is unchanged.

Files: `tldw_chatbook/Personal_Context/recovery.py`, `tldw_chatbook/Sync_Interop/{recovery.py,sync_state_repository.py}`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, `Tests/Backup_Recovery/{test_activation_network_sync.py,test_restore_data_groups.py}`, `Tests/UI/test_settings_personal_context.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Docs/User_Guide/settings/personal-context-profile.md`, `backlog/docs/backup-recovery-owner-inventory.md`.

**Owner decision, 2026-11-13 (AskUserQuestion, relayed by the coordinator), on the NEEDS_CONTEXT above:** "Backup & Restore ▸ Restored profiles (Recommended)". The option as put to the owner: "Next to the new 'Approve settings' review (TASK-40711), so every post-restore review sits in one place. No size waiver needed."

**2026-11-13 (round 14, pcc round 2): the review's screen; AC#9 met.**
- **Where.** Backup & Restore > Restored profiles > Current profile, beside TASK-40711's **Approve settings**, following its conventions. The list worker adds `RecoveryService.sync_identity_review()` to the current-profile summary. A `markup=False` **Sync identity review** note says why sync is off and what confirming does, then lists the six counts. **Clear sync identity and approve sync** runs `RecoveryService.approve_sync_identity(shown)` on a worker (generation-checked delivery through `_config_approval_done`) with the counts that were shown. A refusal shows "Sync identity review unavailable: <issue message> (<code>)"; `sync_identity_counts_changed` has its own copy.
- **Path.** `RecoveryService._sync_state_path` reads the selected profile's config under a storage lease (as `_recovered_profile_source` does) and returns `<user data>/tldw_chatbook_sync_state.db`.
- **Tests.**
  - `Tests/UI/test_backup_restore_setup_entry.py::test_restored_profiles_sync_identity_review_shows_counts_then_approves[approved|counts_changed]`: a laned file. Red with the screen at the previous head: "AssertionError: no sync review".
  - The relink child now drives the review through `RecoveryService` with a real profile config, and asserts `_sync_state_path()`.
- **Docs.** `Docs/Backup-and-Recovery.md` (Restored profiles) and the Personal Context User Guide describe it.

