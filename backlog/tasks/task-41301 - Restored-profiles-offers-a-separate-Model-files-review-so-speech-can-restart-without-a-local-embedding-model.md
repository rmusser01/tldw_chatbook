---
id: TASK-41301
title: "Restored profiles offers a separate Model files review, so speech can restart without a local embedding model"
status: Done
assignee:
- '@claude'
created_date: 2026-11-16 23:33
updated_date: 2026-11-17 00:30
labels:
- follow-up
- backup
- speech
priority: medium
references:
- tldw_chatbook/Backup_Recovery/recovery_service.py
- tldw_chatbook/UI/Screens/backup_restore_screen.py
- tldw_chatbook/RAG_Search/model_recovery.py
dependencies:
- TASK-41119
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 15 of the 2026-10-23 burn-down from TASK-41119's concern (review r15br-rev1).

**Why.** Speech start-up admits through `config`, `models.artifacts`, `tts.profile_store` and `tts.voices`. After TASK-41119 every one has a review except `models.artifacts`, whose only approver is RAG's local embedding model review (`RAG_Search/model_recovery.approve_local_embedding`), offered only when the active RAG profile names an existing local HF embedding model. Without one, speech stays off after every restore.

**Owner decision, 2026-11-16:** "Separate model-files review (Recommended)".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Restored profiles offers its own Model files review whenever `models.artifacts` is pending, and it approves only `models.artifacts` for the reviewed current generation (a stale generation is refused)
- [x] #2 With the settings, speech and model-files reviews done, speech start-up runs again without a local embedding model; a real-binding test pins it off until both the speech and model-files reviews are done, red first
- [x] #3 `models.artifacts` keeps exactly one approver unless both are needed, with the decision about RAG's "Review local model" explained; the approver census pins the result and its disjointness
- [x] #4 The Speech review copy and the backup docs name the Model files review instead of RAG's local model review
<!-- AC:END -->

## Implementation Plan

1. `RecoveryService.approve_models(generation)` through the shared `_reviewed_generation` witness check, approving only `models.artifacts`.
2. Restored profiles > Current profile: a Model files review note and **Approve model files** while `models.artifacts` is pending; the Speech review copy names it.
3. RAG's "Review local model" keeps its exact-files receipt but no longer approves `models.artifacts`, so the Model files review is the one approver; RAG's message says the model files also need approval.
4. Tests: real-binding speech child extended to the three reviews (red first), RAG local-embedding child updated, mounted screen test, census row moved and disjointness asserted.
5. Docs: `Docs/Backup-and-Recovery.md` (setup after restore and the RAG local model steps).

## Implementation Notes

**Summary.** Backup & Restore ▸ **Restored profiles** ▸ **Current profile** shows a **Model files review** whenever `models.artifacts` is pending. The note says what stays off: local model downloads, installs and activation, speech start-up, and a restored local embedding model. It says approval covers every profile the recovery restored, and that a local embedding model also keeps its own review in Settings > Library/RAG. **Approve model files** calls the new `RecoveryService.approve_models(generation)`, which goes through the shared `_reviewed_generation` witness check, refuses a stale generation with `models_review_changed`, and approves only `models.artifacts`. Speech can now restart after a restore without any local embedding model. The Speech review copy now names **Approve settings** and **Approve model files**.

**Decision: one approver; RAG's "Review local model" stays but no longer approves the owner.** RAG's review fingerprints one embedding model's exact file closure and settings, and writes a receipt that `_require` checks. Until now it also approved all of `models.artifacts`, which reaches further than what it reviewed: it turned on speech and every local model download, install and activation. Both approvals are not needed. Keeping RAG's would leave two approvers for one owner, and the owner's ruling makes the new review the one "used by speech and RAG alike". So:
- `approve_local_embedding` now writes only the receipt.
- A restored local embedding model needs the receipt and the model-files approval, in either order. `_require` already checked both.
- RAG's success message tells the user to approve model files in Backup & Restore too.
- Cost: a RAG user who restored a local embedding model makes one more approval, in Backup & Restore.

**Tests and red/green evidence** (load 85-91 on 18 CPUs):
- **Real binding (AC#1, #2).** `Tests/Architecture/test_backup_owner_inventory.py::test_the_speech_review_turns_speech_start_up_back_on[pending|speech|models|both]` (architecture-guards lane). The settings review is done first. Speech stays off unless both the speech review and the model-files review have run. Each review refuses a stale generation and approves only its own owners, and `both` starts tts and stts.
  - Red first: `[models]` and `[both]` failed, because `approve_models` did not exist.
  - Red on an assertion with `approve_models` temporarily approving nothing: `pending_owners` still held `models.artifacts`.
- **Census (AC#3).** The `model_recovery.py` `'models.artifacts'` row moves to `recovery_service.py` `'models.artifacts'`, and the disjointness assert now covers `models.artifacts`. Red on the previous head (RAG still approving): "Left contains 1 more item". `test_only_backup_and_restore_calls_its_owner_reviews[approve_speech|approve_models]` replaces the speech-only caller pin.
- **Screen.** `Tests/UI/test_backup_restore_setup_entry.py::test_restored_profiles_model_files_review_approves_models_alone[approved|changed]` (whole-file UI census entry). It checks the copy, the button, the call with the shown generation, that the speech review stays, and that after a refusal the review comes back with an enabled button. Red on the previous screen: "no model files review". The speech review test now asserts its copy names **Approve model files** and no longer names "Review local model".
- **RAG.**
  - `test_activation_local_embedding.py`'s `_REVIEW` now asserts the receipt alone approves nothing and still refuses construction. It then approves through the real `RecoveryService.approve_models`.
  - The isolated child approves the owner as the model-files review would.
  - `test_rag_model_recovery_controls.py` checks for the receipt instead of the owner approval.
  - At load 65-90 the fixed 45/60 s child budgets of these files timed out for almost every node, base and head alike: 41 of 43 failures in a `-n 4` run of the four RAG/model activation files were `TimeoutExpired`. The other two (`test_activation_rag.py`) passed serially.
  - With the child budgets temporarily raised and then reverted, 5 of 5 `test_activation_local_embedding.py` nodes passed serially: `review_and_retirement[approved|changed]`, `finalized_isolated_generation_reopens_local_model_receipt`, `receipt_is_durable_and_generation_bound[generation]` and `projection_identity…[identity_owner]`. These run the routed flow for real: receipt alone refused, then the real `approve_models`, then the model loads.
  - **Not verified:** `test_rag_model_recovery_controls.py`'s three approve paths fail at the child's line 121 (`escape(str(model)) in dialog.message`), which runs before any approval. The `[cancel]` node fails at the same line on base source (cc6006fcf5), so it is red at base and not this task's. My edits there (a receipt check, then the model-files approval) therefore never run.

**Docs.** `Docs/Backup-and-Recovery.md`: a Model files review paragraph, the Speech review sentence, and RAG's step 1, which now says the restored model files also need **Approve model files**.

**Files.** `tldw_chatbook/Backup_Recovery/recovery_service.py`, `tldw_chatbook/UI/Screens/backup_restore_screen.py`, `tldw_chatbook/RAG_Search/model_recovery.py`, `tldw_chatbook/UI/Screens/settings_screen.py` (message only, +1 line, under its 35,020 ceiling), `Tests/Architecture/test_backup_owner_inventory.py`, `Tests/UI/test_backup_restore_setup_entry.py`, `Tests/Backup_Recovery/test_activation_local_embedding.py`, `Tests/Backup_Recovery/test_rag_model_recovery_controls.py`, `Docs/Backup-and-Recovery.md`.
