---
id: TASK-41121
title: "Owner decision needed: damaged Personal Context objects other than records still block the restored-profile review"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-16 21:23'
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
- [ ] #2 The restored-profile review follows that decision, and when re-encryption is refused for a damaged object, the user is told which kind of object and what they can do instead of the generic failure copy
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
  - Removed with what names the version: `record` (as before), `proposal` (head or history: a pending suggestion or a content-free receipt; nothing points at a proposal), manifest history and scope history (no lineage pointer reads them; only queued envelopes name them), `outbox` (the `encrypted_outbox` row: a queued sync change is lost; the restored link is cleared and the next link is a fresh first link), `undo` (the `local_undo` row: that edit's undo is gone; undo expires within a day anyway), `record_link` (the `local_record_links` row; `get_record_derivation` has no production caller), `scope_binding` (the `local_scope_bindings` row: the workspace reads as unmapped, which is how a damaged mapping already reads, and Map workspace sets it again), `runtime_policy` (the `local_runtime_policy` row: the global setting goes back to off; a workspace's agent access goes back to the default, Propose).
  - Not removable, so re-encryption still refuses, by name: a **current scope**. Removing it leaves that scope's records pointing at no scope (they can no longer be listed or edited), and removing the global scope breaks context building for the whole profile (`profile_scope_invalid`, one global scope required). **NEEDS_CONTEXT** (below). A **current manifest** never reaches the review: a damaged manifest head fails key authentication, so the profile is locked as "Locked: key belongs to a different copy", which the User Guide's troubleshooting row already says ("A damaged profile header reads the same way").
- **Count.** `damaged_versions()` (was `damaged_record_versions()`) returns counts by kind; `PersonalContextService.damaged_object_counts()` (was `damaged_record_count()`).
- **Re-encryption.** `reencrypt_profile(expected_damaged=<dict by kind>)` skips every removable damaged row, refuses if the counts by kind differ from the review's, and `_remove_damaged` also deletes the local row naming the version. A damaged current scope raises `ProfileLockedError` with `reason_code = DAMAGED_SCOPE`.
- **Review.** Settings > Personal Context shows, e.g., "2 damaged records, 1 damaged older profile version, 2 damaged undo entries and 1 damaged workspace mapping can't be read and will be removed" (records only reads exactly as before). On a `DAMAGED_SCOPE` refusal the panel says "A damaged scope can't be read, and removing it would leave its records without a scope, so the restored profile was not re-encrypted or confirmed. Restore Personal Context again from another backup, or roll the restore back in Backup & Restore > Recovery copies." instead of the generic failure copy; the review stays offered (AC#2).
- **Tests.** `Tests/Personal_Context/test_repository.py::test_reencryption_removes_a_damaged_object_of_each_removable_kind[proposal|manifest|scope|outbox|undo|record_link|scope_binding|runtime_policy]` on a service-made profile holding every kind: count by kind, a wrong count refuses byte-for-byte, then the version, every row naming it and its queued bodies go, everything else stays, and the profile opens and lists its records. `test_a_damaged_current_scope_refuses_reencryption_by_name`. UI: `Tests/UI/test_settings_personal_context.py::test_restored_profile_review_shows_and_removes_the_damaged_count[…]` (new by-kind case), `test_a_damaged_current_scope_refusal_names_the_kind`. The TASK-40140 tests moved to the by-kind API.
- **Red/green.** With the records-only rule restored in `_reads` (temporary edit), all 9 repository cases failed (`ProfileIntegrityError: Canonical object integrity failed.`, re-encryption refused). Skipping the new local-row deletion turned the 5 local kinds red on the `_pointers` assertion. In the panel, dropping the `DAMAGED_SCOPE` branch failed the refusal test (`notices` ended with the generic copy), and a records-only `_damaged_copy` failed the by-kind case on the copy assertion. All reverted; green: 22 + 10 targeted, then the covering run below.
- **Covering runs (2026-11-15/16, `-n 4` unless noted).** All 12 `test_replace_restore_makes_personal_context_readable_and_rolls_back` scenarios: 12 passed under the default 300 s child budget at load 7-45 (46-173 s each). `Tests/Personal_Context` + `Tests/UI/test_settings_personal_context.py` + `Tests/UI/test_personal_context_proposal_review.py`: 803 passed, 6 failed, 1 error at load 100-117; on rerun all pass except `test_production_settings_inspects_private_and_deleted_metadata[size0]`, a 5 s provenance wait that also fails on base with the same command at load 85. `test_recovery_copies.py`, `test_later_rollback.py`, `test_recovery_service.py`, `test_credentials.py`, `test_later_rollback_credential_ui.py`, `Tests/DB/test_private_sqlite_inventory.py`, the rest of `test_restore_data_groups.py`: 235 passed, 1 setup error (`test_f9_later_rollback_requires_explicit_credential_review`: a pip-install timeout under load; serially it fails "real exec expected", identically on base). `test_restore_plan.py`: 119 passed. `Tests/Architecture/test_module_size_ratchet.py` 207 passed; ruff gate, markup census and preflight pass.
- **Lane.** The new tests sit in the files their TASK-40140 neighbours use (`test_repository.py`, `test_restore_data_groups.py`, `test_settings_personal_context.py`), none of which a PR lane runs whole; no census entry added (round-15 rule: the shard count is the coordinator's decision).
- **Files.** `tldw_chatbook/Personal_Context/{repository_keys.py,service.py,recovery.py}`, `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`, `Tests/Personal_Context/test_repository.py`, `Tests/UI/test_settings_personal_context.py`, `Docs/User_Guide/settings/personal-context-profile.md`.

**AC#2 stays open, and the task In Progress:** every removable kind follows the decision, and the scope refusal names the kind and what to do, but a damaged current scope still blocks the review until the owner rules.

**NEEDS_CONTEXT (owner): a damaged current scope.** Options and consequences:
1. Keep refusing (what ships): the review cannot be confirmed; the user restores Personal Context from another backup or rolls the restore back. Nothing is lost silently.
2. Remove the scope and every record, proposal, mapping and agent setting in it: the review completes, and that scope's content is lost (it is unreachable today anyway). For the global scope this still leaves a profile with no global scope, which breaks context building, unless a new empty global scope is created (a further product choice).
3. Remove only the scope version: its records stay stored but can never be listed or edited (inconsistent), and the global case breaks context building.

**Concern for the owner (not a block).** Removing a damaged workspace agent access setting reverts that workspace to the default, Propose; a damaged one was read as "denied" before. The review says the setting will be removed before confirming.
