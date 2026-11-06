---
id: TASK-38985
title: "Library modal inventory has six nodes red on dev because SkillRecoveryReviewModal has no contract row and the file runs on no PR lane"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- library
- tests
- ci
priority: medium
references:
- Tests/UI/test_library_modal_dismissal.py
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/skills_trust_modals.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

**The red.** `Tests/UI/test_library_modal_dismissal.py` at `-n 4`: 6 failed, 181 passed, in 28.2 s at load 14.8. None of the failures is RecoveryRequired. The six nodes:
- `test_library_modal_inventory_matches_declared_edges_bidirectionally`
- five `test_library_modal_inventory_rejects_wrong_type_at_moved_presenter[...]` cases: the two skill-trust passphrase presenters, `handle_library_export_choose_destination`, `_open_root_picker` and `handle_library_ingest_browse`

All six fail with the same message: "unresolved modal constructor in supported presenter: tldw_chatbook/UI/Screens/library_screen.py:LibraryScreen._review_restored_skill_trust (SkillRecoveryReviewModal(review, paths[1], current))".

**Cause.**
- b5251e9a6e (2026-09-16, TASK-32628) added the presenter `_review_restored_skill_trust` (`library_screen.py:27530-27553`). It imports `SkillRecoveryReviewModal` from `tldw_chatbook/UI/Library_Modules/skills_trust_modals.py` (`:546-548`).
- The modal has no row in `LIBRARY_MODAL_CONTRACTS` (`Tests/UI/test_library_modal_dismissal.py:280`), so `_MODAL_TYPES_BY_NAME` (`:825-828`) cannot resolve it.
- The AST walk asserts on that (`:920-925`), which aborts the whole discovery.

**Ownership and lanes.** TASK-31249 (Done) fixed an earlier red in the same test. TASK-32864 (To Do) uses this file only as its regression net. No branch due to merge before this one changes the file. Neither the census nor any workflow runs it, so the red reached dev unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every node in Tests/UI/test_library_modal_dismissal.py passes at -n 4 (red before: 6 of 6 with the message above)
- [ ] #2 SkillRecoveryReviewModal has a declared dismissal contract and the bidirectional check includes its edge; removing that row brings the "unresolved modal constructor" failure back (negative control)
- [ ] #3 The file runs on a PR lane, with its measured whole-file time recorded (28.2 s at -n 4, load 14.8, on dev)
<!-- AC:END -->
