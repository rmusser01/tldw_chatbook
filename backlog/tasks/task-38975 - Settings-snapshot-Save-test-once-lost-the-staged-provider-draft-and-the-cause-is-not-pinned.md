---
id: TASK-38975
title: "Settings snapshot Save test once lost the staged provider draft and the cause is not pinned"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- settings
- tests
- flaky
priority: medium
references:
- Tests/UI/test_llamacpp_snapshot_settings.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/LLM_Management/snapshot_settings.py
dependencies:
- TASK-36530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36530 branch (447932a5f0), 2026-11-05. A reviewer was investigating it when this was filed.

**The failure.** `Tests/UI/test_llamacpp_snapshot_settings.py::test_f9_save_preserves_new_edits_while_persistence_waits` stages a provider model (`_stage_provider_value("model", "snapshot-review-model")`), edits the snapshot keep-count, runs a snapshot Save that is held mid-write, and edits the field again. Its last assertion, `screen._provider_draft().is_dirty`, then failed once: `_provider_draft()` returned None. The staged provider/model draft was gone after the snapshot Save.

**The evidence so far, from TASK-36530's notes:**
- serial rounds at load 6.8-18.8 on 18 cores: 9 of 10 and 10 of 10 (the one failure);
- a probe that logged every pop of that draft caught no pop in 136 runs;
- unprobed parallel runs: 56 of 56;
- the old `pilot.pause` wait: 48 of 48.

That is about 1 failure in 240 runs of the new wait. Whether TASK-36530's wait change exposed it or caused it is unknown. If the product loses the draft, a user who stages a provider/model edit and then saves the snapshot preferences silently loses that edit.

**Where the draft can go.**
- `_provider_draft` is `self._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)` (`tldw_chatbook/UI/Screens/settings_screen.py:13127-13128`).
- The entry can leave in two ways. Many staging paths pop a category's draft when it becomes clean (for example `:7611`, `:7913`, `:8788`, `:9607`). `restore_state` replaces the whole dict (`:4463`). A probe on pops cannot see a whole-dict replacement or a re-staging that makes the draft equal to its originals.
- The probe also changes timing, which may be why it never caught the failure.

The test runs on no PR lane; the census lists only `test_f9_snapshot_controls_paint_and_keyboard_save[size1]` from this file. It is a plain-profile test, so locally it needs the temporary `@private_profile_test` workaround (TASK-38111).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failure is reproduced with a capture that does not change timing before the failure: nothing is recorded until the final assertion fails, and then the notes record the draft dict's identity against the one present at staging, every thread's stack (faulthandler or equivalent), and the draft's values and originals
- [ ] #2 The cause is recorded: the test (for example a wait that ends before a deferred re-stage) or the product (a snapshot Save or a deferred sync that drops or cleans a staged provider/model draft)
- [ ] #3 If it is the product, a deterministic test forces the ordering and is red before the fix: after a snapshot Save, the staged provider/model draft is still present and dirty
- [ ] #4 test_f9_save_preserves_new_edits_while_persistence_waits passes 240 of 240 runs at a recorded load, or the notes give the measured failure ratio after the fix
<!-- AC:END -->
