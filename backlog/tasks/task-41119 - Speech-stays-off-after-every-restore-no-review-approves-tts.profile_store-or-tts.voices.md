---
id: TASK-41119
title: "Speech stays off after every restore: no review approves tts.profile_store or tts.voices"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 00:05
labels:
- follow-up
- backup
- speech
priority: medium
references:
- tldw_chatbook/app_speech.py
- tldw_chatbook/Backup_Recovery/activation.py
- tldw_chatbook/TTS/recovery.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** After any restore, speech start-up never runs again, and the user has no review that turns it back on. TASK-40711's notes name this consumer.

**Evidence.** `app_speech.py:986-994` admits speech start-up through `execution_scope(("config", "models.artifacts", "tts.profile_store", "tts.voices"))`. The approvers that exist call `ActivationStore.approve` for `skills` (`Skills_Interop/recovery_activation.py:459`), plugins (`Plugins/backup.py:502`), sync identity (`Sync_Interop/recovery.py:170`), MCP, Personal Context, Notes, `config` (`recovery_service.py:1199`), `models.artifacts` (`RAG_Search/model_recovery.py:330`) and RAG; none names a `tts.*` owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A restored generation's speech owners get a review (in Backup & Restore > Restored profiles or Speech settings) that approves them
- [x] #2 After that review and the existing config and model reviews, speech start-up runs again; a test pins pending then approved
- [x] #3 An approver-census pin allows exactly the new approver for these owners
<!-- AC:END -->

## Implementation Plan

1. `RecoveryService.approve_speech(generation)`: same current-generation witness check as `approve_config` (shared helper), then approve `tts.profile_store` and `tts.voices` only.
2. Backup & Restore > Restored profiles > Current profile: while either owner is pending, a Speech review note says speech start-up stays off until approval and also needs the settings and local model reviews, then **Approve voices** runs `approve_speech` on a worker (TASK-40711's pattern).
3. Tests: a real-binding child in `test_backup_owner_inventory.py` pins speech start-up (`_run_speech_initialization`) off while the speech owners are pending (settings and model approved) and on after `approve_speech`; a mounted screen test pins the copy and wiring; the approver census gains exactly the two speech rows.
4. `Docs/Backup-and-Recovery.md` says what the review turns back on.

## Implementation Notes

**Summary.** Backup & Restore ▸ **Restored profiles** ▸ **Current profile** now carries a **Speech review** while the current generation's `tts.profile_store` or `tts.voices` is pending: a note saying speech start-up stays off until approval, that approving covers every profile this recovery restored, and that speech also needs **Approve settings** and **Review local model** (Settings > Library/RAG), then **Approve voices**. The button runs the new `RecoveryService.approve_speech(generation)` on a worker. It approves those two owners and nothing else.

**Decisions.**
- **No new product decision was needed.** The AC allows Restored profiles, and the owner placed restored-profile reviews there; this follows TASK-40711's settings review exactly (placement under the summary, per-generation approval, worker, stale-generation refusal, messages).
- `approve_config` and `approve_speech` share one helper, `_reviewed_generation(generation, owners, changed)`, which holds the selected profile's lease and refuses with `changed` (`config_review_changed` / `speech_review_changed`) unless the witness is still that generation, requires the owners and is a live paired witness. Each caller keeps its own literal `store.approve(generation, "<owner>")` calls, so the approver census still names the exact owners.
- Only the two owners speech start-up admits through (`app_speech.py`'s `execution_scope`) are approved. `tts.references` is not: nothing gates on it.
- Speech starts from the next start: initialization is scheduled at start-up, so the note and the success message say so.

**Tests and red/green evidence** (load 53-120 on 18 CPUs):
- **Real binding (AC#1, #2).** `Tests/Architecture/test_backup_owner_inventory.py::test_the_speech_review_turns_speech_start_up_back_on[pending|approved]` (lane: architecture-guards). A child binds exactly speech's four owners, approves `config` through `approve_config` and `models.artifacts` through the store (standing in for the local model review), then: `pending` pins the current profile's pending owners as the two speech owners and `_run_speech_initialization` not starting `tts` or `stts`; `approved` pins a stale generation refused with `speech_review_changed`, then `approve_speech`, no pending owners, and both kinds starting.
  - Red before: `[approved]` failed with `AttributeError: 'RecoveryService' object has no attribute 'approve_speech'`.
  - Red on an assertion with `approve_speech` temporarily approving only `tts.voices`: `pending_owners` was `('tts.profile_store',)`.
  - Red with the two TTS owners temporarily removed from `app_speech`'s gate: `[pending]` failed `AssertionError: ('tts', 'handler')`, so the pending pin really depends on the gate. All reverted.
- **Census (AC#3).** `test_no_reviewer_approves_the_shared_sync_owners` gains exactly two rows, `recovery_service.py` `'tts.profile_store'` and `'tts.voices'`, and asserts no other approver's owner set holds them. Red with a temporary stray `store.approve(witness["generation"], "tts.profile_store")` in `Plugins/backup.py` ("Left contains 1 more item"); reverted. `test_only_the_speech_review_calls_approve_speech` pins the screen as the only caller (red before: empty Counter).
- **Screen.** `Tests/UI/test_backup_restore_setup_entry.py::test_restored_profiles_speech_review_approves_voices_alone[approved|changed]` (whole-file UI census entry, no census edit): copy, button label, the approval call with the shown generation, the review going while the settings review stays, and the stale refusal message.
- `test_the_config_review_turns_config_consumers_back_on_and_never_sync` and `test_only_the_settings_review_calls_approve_config` still pass after the refactor.

**Concern (resolved by TASK-41301, owner decision 2026-11-16): `models.artifacts`.** Speech start-up also needs `models.artifacts`, and its only approver is the local embedding model review (`RAG_Search/model_recovery.py`), offered only when the active RAG profile names an existing local HF embedding model. A user without one cannot approve `models.artifacts`, so speech still stays off after any restore. The AC names "the existing config and model reviews", and the copy and docs say what is needed. Whether speech should keep requiring `models.artifacts`, or which review should approve it for speech, is a product decision (proposed follow-up).

**Docs.** `Docs/Backup-and-Recovery.md` (the backup user page; there is no `Docs/User_Guide/` page for it): what the speech review turns back on and what else speech needs.

**Files.** `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/UI/test_backup_restore_setup_entry.py`, `Docs/Backup-and-Recovery.md`.

**Review r15br-rev1 (fixed).** The `[changed]` case could pass even if the refused review never came back. It now waits for `#backup-speech-review` and an enabled `#backup-approve-speech` after the refusal. Red with the list refresh after a refusal temporarily removed (`TimeoutError`). The speech child and the screen copy later moved to TASK-41301's model-files review; see that task.
