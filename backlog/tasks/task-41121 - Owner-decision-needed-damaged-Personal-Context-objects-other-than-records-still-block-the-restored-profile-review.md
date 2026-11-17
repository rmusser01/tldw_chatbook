---
id: TASK-41121
title: "Owner decision needed: damaged Personal Context objects other than records still block the restored-profile review"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-17 02:38'
created_date: 2026-11-15 02:20
labels:
- follow-up
- personal-context
- backup
- owner-decision
priority: medium
references:
- tldw_chatbook/Personal_Context/repository_keys.py
- tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The owner ruled on 2026-11-14 that damaged records are counted, shown and removed during re-encryption (TASK-40140 AC#5). A damaged object of any other kind (manifest, scope, proposal, local body, outbox body, undo entry, binding) still makes re-encryption refuse, so the restored-profile review can never be confirmed and the profile stays unusable. What to do with those objects is a product decision this task asks for; it does not decide it.

**Evidence.**
- `repository_keys.py:243-250`: `_reads` returns None only for a damaged record ("a record only; any other damaged object still refuses"); `damaged_record_versions` (`:225-241`) counts only `object_type = 'record'`.
- `reencrypt` (`:290-318`) re-encrypts every row of `encrypted_objects`; a non-record that fails authentication raises.
- The panel (`personal_context_panel.py:1003-1024`) shows the mutation's generic failure copy and reloads; it does not name the damaged object or offer a way forward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on damaged non-record objects (for example: drop them with a count, keep the profile read-only, or another option) is recorded verbatim in the notes
- [x] #2 The restored-profile review follows that decision, and when re-encryption is refused for a damaged object, the user is told which kind of object and what they can do instead of the generic failure copy
- [x] #3 Tests cover each object kind the decision names, red before the change
<!-- AC:END -->

## Implementation Plan

1. Record the owner decision verbatim.
2. Classify every `encrypted_objects` kind: drop it during re-encryption when the profile stays consistent and usable (with its pointer rows: heads, outbox entries, quarantine notes, and the local table row that names the version); keep refusing a kind whose removal breaks the profile, and report it as NEEDS_CONTEXT.
3. `damaged_objects()` counts damaged versions by kind; re-encryption takes the reviewed counts and refuses if they changed.
4. The review shows the count by kind; a refusal for a kind that cannot be dropped names the kind and what to do instead of the generic failure copy.
5. Tests per droppable kind and for each refusing kind, red first; User Guide update.

## Implementation Notes

Owner decision 2026-11-15: "no"

(Question: "damaged Personal Context objects other than records still block the restored-profile review". Meaning, as relayed by the coordinator: damaged non-record objects must not block the review; give them the treatment TASK-40140 AC#5 gives damaged records, count, show the count by kind, drop during re-encryption, for every kind where dropping leaves the profile consistent and usable; stop with NEEDS_CONTEXT for a kind where it would not.)

**2026-11-15 (round 15, stream pc).**
- **Classification** (`repository_keys._DROPPABLE`, every `encrypted_objects` kind the repository writes):
  - Removed with what names the version: `record` (as before), `proposal` (head or history: a pending suggestion or a content-free receipt; nothing points at a proposal), manifest history and scope history (no lineage pointer reads them; only queued envelopes name them), `outbox` (the `encrypted_outbox` row: a queued sync change is lost; the restored link is cleared and the next link is a fresh first link), `undo` (the `local_undo` row: that edit's undo is gone; undo expires within a day anyway), `record_link` (the `local_record_links` row; `get_record_derivation` has no production caller), `scope_binding` (the `local_scope_bindings` row: the workspace reads as unmapped, which is how a damaged mapping already reads, and Map workspace sets it again), `runtime_policy` (the `local_runtime_policy` row: the global setting goes back to off; a workspace's agent access goes back to the default, Propose; superseded below: it becomes Off).
  - Not removable, so re-encryption still refuses, by name: a **current scope**. Removing it leaves that scope's records pointing at no scope (they can no longer be listed or edited), and removing the global scope breaks context building for the whole profile (`profile_scope_invalid`, one global scope required). **NEEDS_CONTEXT** (below). A **current manifest** never reaches the review: a damaged manifest head fails key authentication, so the profile is locked as "Locked: key belongs to a different copy", which the User Guide's troubleshooting row already says ("A damaged profile header reads the same way").
- **Count.** `damaged_versions()` (was `damaged_record_versions()`) returns counts by kind; `PersonalContextService.damaged_object_counts()` (was `damaged_record_count()`).
- **Re-encryption.** `reencrypt_profile(expected_damaged=<dict by kind>)` skips every removable damaged row, refuses if the counts by kind differ from the review's, and `_remove_damaged` also deletes the local row naming the version. A damaged current scope raises `ProfileLockedError` with `reason_code = DAMAGED_SCOPE`.
- **Review.** Settings > Personal Context shows, e.g., "2 damaged records, 1 damaged older profile version, 2 damaged undo entries and 1 damaged workspace mapping can't be read and will be removed" (records only reads exactly as before). On a `DAMAGED_SCOPE` refusal the panel says "A damaged scope can't be read, and removing it would leave its records without a scope, so the restored profile was not re-encrypted or confirmed. Restore Personal Context again from another backup, or roll the restore back in Backup & Restore > Recovery copies." instead of the generic failure copy; the review stays offered (AC#2).
- **Tests.** `Tests/Personal_Context/test_repository.py::test_reencryption_removes_a_damaged_object_of_each_removable_kind[proposal|manifest|scope|outbox|undo|record_link|scope_binding|runtime_policy]` on a service-made profile holding every kind: count by kind, a wrong count refuses byte-for-byte, then the version, every row naming it and its queued bodies go, everything else stays, and the profile opens and lists its records. `test_a_damaged_current_scope_refuses_reencryption_by_name`. UI: `Tests/UI/test_settings_personal_context.py::test_restored_profile_review_shows_and_removes_the_damaged_count[…]` (new by-kind case), `test_a_damaged_current_scope_refusal_names_the_kind`. The TASK-40140 tests moved to the by-kind API.
- **Red/green.** With the records-only rule restored in `_reads` (temporary edit), all 9 repository cases failed (`ProfileIntegrityError: Canonical object integrity failed.`, re-encryption refused). Skipping the new local-row deletion turned the 5 local kinds red on the `_pointers` assertion. In the panel, dropping the `DAMAGED_SCOPE` branch failed the refusal test (`notices` ended with the generic copy), and a records-only `_damaged_copy` failed the by-kind case on the copy assertion. All reverted; green: 22 + 10 targeted, then the covering run below.
- **Covering runs (2026-11-15/16, `-n 4` unless noted).** All 12 `test_replace_restore_makes_personal_context_readable_and_rolls_back` scenarios: 12 passed under the default 300 s child budget at load 7-45 (46-173 s each). `Tests/Personal_Context` + `Tests/UI/test_settings_personal_context.py` + `Tests/UI/test_personal_context_proposal_review.py`: 803 passed, 6 failed, 1 error at load 100-117; on rerun all pass except `test_production_settings_inspects_private_and_deleted_metadata[size0]`, a 5 s provenance wait that also fails on base with the same command at load 85. `test_recovery_copies.py`, `test_later_rollback.py`, `test_recovery_service.py`, `test_credentials.py`, `test_later_rollback_credential_ui.py`, `Tests/DB/test_private_sqlite_inventory.py`, the rest of `test_restore_data_groups.py`: 235 passed, 1 setup error (`test_f9_later_rollback_requires_explicit_credential_review`: a pip-install timeout under load; serially it fails "real exec expected", identically on base). `test_restore_plan.py`: 119 passed. `Tests/Architecture/test_module_size_ratchet.py` 207 passed; ruff gate, markup census and preflight pass.
- **Lane.** The new tests sit in the files their TASK-40140 neighbours use (`test_repository.py`, `test_restore_data_groups.py`, `test_settings_personal_context.py`), none of which a PR lane runs whole; no census entry added (round-15 rule: the shard count is the coordinator's decision).
- **Files.** `tldw_chatbook/Personal_Context/{repository_keys.py,service.py,recovery.py}`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, `Tests/Personal_Context/test_repository.py`, `Tests/UI/test_settings_personal_context.py`, `Docs/User_Guide/settings/personal-context-profile.md`.

**Superseded by the owner's 2026-11-16 decisions below.** The first round shipped a refusal for a damaged current scope, with copy naming it (NEEDS_CONTEXT), and reverted a dropped workspace agent access setting to the default, Propose. Fix round 1 replaces both.

**Owner decisions 2026-11-16 (relayed by the coordinator):**
- Damaged current scope: "Rebuild the scope (Recommended)." Replace a damaged current scope with a fresh one: global stays global, a named scope becomes "Recovered scope"; its records stay and are re-attached; the scope's own settings reset to defaults; the review states it before Confirm ("1 damaged scope will be rebuilt; its N records are kept").
- Dropped workspace agent-access setting: "Revert to Off (Recommended)." The workspace's agent access becomes Off, with no access until the user sets it again, not Propose; the review says so.

**2026-11-16, fix round 1 (review r15pc-rev1 and the owner decisions).**
- **Rebuild.** A damaged current scope now counts as `scope` (plus `rebuilt_scope_records`, the current records in it that are not themselves damaged; display only, not compared) and no longer raises, so `damaged_object_counts()` reaches the panel. In the re-encryption transaction, after the new key is adopted, `_rebuild_scopes` writes a fresh version of the scope under its own id (`_insert_scope`), so its records stay attached with no rewrite. The kind is sealed in the damaged version, so it comes from a readable older version of the scope, else from a workspace mapping or unlinked row (only workspaces have one), else global when no readable current scope is global. The scope object's own fields reset (fresh times; id and kind carry over), and a readable mapping is rewritten with the same workspace id and the label "Recovered scope". Its agent access row is left alone (fix round 2, below). The global scope's label is "Global" anyway. Ceiling (`ponytail:` in `_rebuilt_scopes`): two damaged unmapped scopes and no readable global one cannot be told apart, and the first by id becomes global.
- **Off.** `AgentAuthority.OFF` (rank 0: allows nothing; `profile_tool_provider` offers no tools; the Settings authority select now lists "Off" first). A dropped current agent access setting of a scope is replaced, in the same transaction, by `{"version": 1, "authority": "off"}`. The global on/off setting (`GLOBAL_POLICY_ID`) is still dropped, which means off. If a rebuilt scope's own agent access row was itself damaged, it is Off.
- **History counts (review finding 5).** An older version of a manifest, scope, workspace mapping or agent access setting counts as `<kind>_history` ("older … version"); the current one stays.
- **Review copy.** For example: "1 damaged scope will be rebuilt; its 3 records are kept. 1 damaged agent access setting can't be read and will be removed. Agent access there is Off until you set it again". The DAMAGED_SCOPE refusal, its panel branch and copy are gone.
- **Tests.**
  - `Tests/Personal_Context/test_repository.py::test_a_damaged_current_scope_is_rebuilt_with_its_records[global|workspace]`: counts, rebuild under the same id and kind, the same record heads attached, `list_records` for the scope unchanged, agent access unchanged (fix round 2), the workspace mapping relabelled; for global, `authorized_context_view()` returns its records.
  - `test_a_dropped_agent_access_setting_is_off[workspace|global]`: Off, `require_agent_authority(…, READ_ONLY)` refuses, and the user can set it again.
  - `test_reencryption_removes_a_damaged_object_of_each_removable_kind` now also has `scope_binding_history` and `runtime_policy_history`; its manifest and scope cases are `*_history`.
  - UI: two new copy cases in `test_restored_profile_review_shows_and_removes_the_damaged_count`; `test_a_damaged_current_scope_refusal_names_the_kind` is removed with the refusal.
- **Red.** With the round-0 source (`git diff > patch`, checkout, run, re-apply), all 8 new repository cases failed. The two rebuilds raised `ProfileLockedError: A current scope is damaged.` The four history cases failed on the count assertion (e.g. `{'scope_binding': 1} == {'scope_binding_history': 1}`). The Off cases failed because `AgentAuthority.OFF` did not exist. Negative controls on the new code, each reverted: skipping the Off write gives `PROPOSE is OFF`, false for both Off cases; skipping the relabel fails the workspace mapping assertion (`'W' … == 'Recovered scope'`). Green: 26 passed (`-k` set of re-encryption and damage tests).
- **Files.** `tldw_chatbook/Personal_Context/{repository_keys.py,runtime_policy.py}`, `tldw_chatbook/Agents/profile_tool_provider.py`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, `Tests/Personal_Context/test_repository.py`, `Tests/UI/test_settings_personal_context.py`, `Docs/User_Guide/settings/personal-context-profile.md` (the damage paragraph and an **Off** line under Agent authority).
- **Fix-round covering runs (load 60-108, `-n 4` unless noted).**
  - `Tests/Personal_Context`: 714 passed. The 20 setup errors were all `test_memory_baseline.py` fixtures hitting the 300 s pytest timeout; that file, rerun with `Tests/Agents/test_profile_tool_provider.py` and `test_profile_tool_scope.py` at `--timeout=1500`, gave 99 passed.
  - `test_settings_personal_context.py` + `test_personal_context_proposal_review.py`: 108 passed. The one failure is the base-failing `test_production_settings_inspects_private_and_deleted_metadata[size0]` (TASK-41120 notes).
  - Restore scenarios `reviewed`, `lost_keyring`, `remove` and `test_recovery_copies.py`: 9 passed (`-n 2`). `link` hit the 300 s child budget at load 95, then passed serially in 232 s at load 60.
  - The owner-inventory node passed, as did the ruff gate, the size ratchet (207 passed), the markup census and preflight.

**2026-11-16, fix round 2 (coordinator ruling on the round-1 concern).** Ruling: a rebuilt scope must not lose a readable agent access row. That row is a separate object and is not damaged, so it stays exactly as it was (Off stays Off); removing it, which reverts the scope to Propose, would quietly widen agent access. Only a damaged agent access row reverts, to Off. The scope object's own settings still reset. The user-selectable Off level is accepted, and the User Guide describes Off for any scope.
- `_rebuild_scopes` no longer touches `local_runtime_policy`; it only relabels a readable mapping.
- Test: `test_a_damaged_current_scope_is_rebuilt_with_its_records[workspace_off]` (the workspace set to Off before the damage) and `[workspace]` (Read only) now assert that the authority and the policy version are unchanged after the rebuild. Red against d13794c127: `[workspace_off]` failed `PROPOSE is OFF`, `[workspace]` failed `PROPOSE is READ_ONLY`. Green: `Tests/Personal_Context/test_repository.py` 98 passed (load 80).
- User Guide: the **Off** line says it can be chosen for any scope and changed back; the restore paragraph says a rebuilt scope keeps its agent authority (Off stays Off) unless that setting was damaged too. The review copy already mentioned Off only for a damaged setting.
- Covering: `test_settings_personal_context.py` + `test_personal_context_proposal_review.py` + `Tests/Agents/test_profile_tool_provider.py` + `test_profile_tool_scope.py`: 138 passed, 2 failed at load 70-86 (`test_plaintext_export_captures_scope_when_export_is_confirmed`, `test_production_settings_inspects_private_and_deleted_metadata[size1]`), both passing serially on rerun. Ruff gate and preflight pass.

**2026-11-16, fix round 3 (review r15pc-rev2 and coordinator rulings).**
- **Scope kind is never guessed by order (important).** The review's probe was a damaged global scope plus a damaged, unmapped, single-version workspace `aaa-server-workspace` that sorts first. Round 2 rebuilt the workspace as global, so global context served its records in every chat.
  - Coordinator ruling: use the `scope-global-` id prefix as evidence of the global scope before any inference. If more than one scope is still unsettled after all evidence, fail closed with a named reason. Never pick the global scope by order.
  - `RepositoryKeyCustody._scope_kinds` settles a scope's kind from, in order: a readable older version, a mapping or unlinked row (workspace), then a `scope-global-` id (global). A profile has one global scope, so once one is known the rest are workspaces; with none known, a single unsettled scope is it.
  - More than one unsettled scope with no known global is refused as `AMBIGUOUS_SCOPES` (`personal_context_damaged_scopes_ambiguous`). Two known globals are refused the same way.
  - `damaged_versions()` counts the ambiguous scopes, so the review shows "Some damaged scopes can't be told apart, so this restored profile can't be confirmed. Restore Personal Context again from another backup, or roll the restore back in Backup & Restore > Recovery copies". Re-encryption refuses with that reason, and the panel shows the same copy instead of the generic failure.
- **Labels (coordinator's reading of the owner decision).** The owner's option text said "nothing readable is lost", so a workspace mapping whose label is still readable keeps it. "Recovered scope" is used only for a mapping with no readable label. This is the coordinator's reading, recorded as asked.
- **Copy.**
  - A rebuilt scope with no records reads "it has no records".
  - A damaged global on/off setting counts as `runtime_switch` ("Personal Context on/off setting") with "Agent use of Personal Context stays off until you turn it on".
  - Scope agent access reads "Agent access for that scope / those scopes is Off until you set it again".
- **Docs.** `personal-context-profile.md`: Off on a workspace withholds global context in that workspace, and Off on the global scope withholds profile context from every chat; the restore paragraph covers labels and the ambiguous refusal. `console/chat-basics.md`: Off in the authority list and in the "no profile block is sent" sentence. The `forget_kept_keys` docstring and the owner-inventory prose now say a later copy "usually names the new account; if a later rollback revived the kept account, the TASK-41122 reinstall covers it".
- **Tests.**
  - `test_damaged_single_version_scopes_rebuild_with_the_right_kinds`, three cases:
    - the probe: both scopes keep their kinds, and global context serves only `record-g`
    - the no-evidence single global with id `server-global-1`
    - an unmapped workspace beside a readable global.
  - `test_damaged_scopes_that_cannot_be_told_apart_refuse`: counts, refusal reason, database unchanged.
  - New `[workspace_unlabelled]` rebuild case; the other workspace cases now expect the kept label "W".
  - UI: three new copy cases; `test_an_ambiguous_scope_refusal_says_what_to_do`; the authority round-trip test now also drives the select to Off.
- **Red against 337d06f767** (patch, checkout, run, re-apply):
  - The probe failed on the kind map, with the workspace rebuilt as GLOBAL. The ambiguity test failed because `AMBIGUOUS_SCOPES` did not exist; the old code rebuilt by order.
  - `[workspace]` and `[workspace_off]` failed `'Recovered scope' == 'W'`.
  - UI: four copy cases failed on the copy assertion; the refusal test failed on import.
  - Already green there, as coverage additions: the no-evidence global case, the unmapped workspace case, and the Off round-trip.
- **Covering (load 54-63, `-n 4`).** `Tests/Personal_Context/test_repository.py` + `test_settings_personal_context.py` + `test_personal_context_proposal_review.py` + `Tests/Agents/test_profile_tool_provider.py` + `test_profile_tool_scope.py`: 247 passed. The owner-inventory node passed serially (162 s). Ruff gate, markup census and preflight pass.

**2026-11-16, fix round 4 (review r15pc-rev3).**
- **Minor fixed.** Older versions of the global on/off switch (`GLOBAL_POLICY_ID`) counted as `runtime_policy_history` ("older agent access setting version"). `_damage_kind` now checks the switch first and returns `runtime_switch_history`, which reads "older Personal Context on/off setting version(s)". `_every_kind` now holds two switch versions. Test: `test_reencryption_removes_a_damaged_object_of_each_removable_kind[runtime_switch_history]`. Red on 50afbeb9c0: `{'runtime_policy_history': 1} == {'runtime_switch_history': 1}`. Green: `Tests/Personal_Context/test_repository.py` 104 passed.
- **Accepted residual (coordinator, no code change).** The reviewer noted that a foreign writer could name a pulled workspace `scope-global-*` and defeat the prefix evidence. The coordinator checked tldw_server: `Personalization/personal_context_service.py` mints every scope id, global or workspace, as `f"scope-{uuid4}"` (`_id_factory("scope")` at lines 149, 545 and 806, and `_sync_plan_id(profile_id, "scope")` at 367). The server therefore never mints `scope-global-*`, and after a link the global id carries no prefix. Two damaged unsettled scopes in that state refuse as ambiguous, which is safe. Only a writer outside Chatbook and the server could plant the prefix.
