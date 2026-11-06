---
id: TASK-38985
title: "Library modal inventory: six nodes are red on dev because SkillRecoveryReviewModal has no contract row, and the file runs on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-06 07:19
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
- b5251e9a6e (2026-09-16, TASK-32628) added the presenter `_review_restored_skill_trust` (`library_screen.py:27530-27553`). It imports `SkillRecoveryReviewModal` (`library_screen.py:546-548`) from `tldw_chatbook/UI/Library_Modules/skills_trust_modals.py`.
- The modal has no row in `LIBRARY_MODAL_CONTRACTS` (`Tests/UI/test_library_modal_dismissal.py:280`), so `_MODAL_TYPES_BY_NAME` (`:825-828`) cannot resolve it.
- The AST walk asserts on that (`:920-925`), which aborts the whole discovery.

**Ownership and lanes.** TASK-31249 (Done) fixed an earlier red in the same test. TASK-32864 (To Do) uses this file only as its regression net. No branch due to merge before this one changes the file. Neither the census nor any workflow runs it, so the red reached dev unseen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every node in Tests/UI/test_library_modal_dismissal.py passes at -n 4 (red before: 6 of 6 with the message above)
- [x] #2 SkillRecoveryReviewModal has a declared dismissal contract and the bidirectional check includes its edge; removing that row brings the "unresolved modal constructor" failure back (negative control)
- [x] #3 The file runs on a PR lane, with its measured whole-file time recorded (28.2 s at -n 4, load 14.8, on dev)
<!-- AC:END -->

## Implementation Plan

1. Reproduce the 6 reds at `-n 4` on base e8eeae7db9 (exact-base worktree).
2. Give `SkillRecoveryReviewModal` a `LIBRARY_MODAL_CONTRACTS` row (fixture review, Cancel, `#skills-recovery-review` content, True on Confirm review) and declare its launch edge `LibraryScreen._review_restored_skill_trust`. Bump the table and edge counts, and give the public-positive driver its Confirm review click.
3. Negative control: drop the row and show the "unresolved modal constructor" failure comes back.
4. Lane: read recent UI-lane job times read-only, append the whole file at the UI census tail (not the fast lane), raise `MINIMUM_FILES`, and time the file 3 times with the lane's command.

## Implementation Notes

- **Red on base** (exact-base worktree `fu-base-r10`, `-n 4`): 6 failed and 181 passed, in 44.9 s at load 43. All six fail with the message in the description.
- **Three modals were hidden behind the first abort, not one.** `_discover_library_modal_edges` asserts inside its walk, so each fix uncovered the next gap. A probe that registered each unresolved name and re-ran the walk found the complete set.
  - `SkillRecoveryReviewModal` (`LibraryScreen._review_restored_skill_trust`): a contract row with the fixture review, Cancel, `#skills-recovery-review`, and True from Confirm review; its edge; and a public-positive click on `#skills-recovery-continue`.
  - `NotesRecoveryDialog` (`LibraryFileNotesWorkspace._review_pairing`, from the same commit b5251e9a6e). It was not a safe-dismissal modal. It now adopts `SafeModalDismissMixin`, as its sibling `SkillRecoveryReviewModal` does: `SAFE_MODAL_CONTENT = "#notes-recovery-dialog"`, Escape bound to `request_safe_cancel`, and Close going through `request_safe_cancel(source="visible")`. User-visible effect: a click on the backdrop now closes it, as it closes every other Library modal, and focus returns to its opener. Approval is unchanged. It has a contract row (None on every negative gesture, no positive result) and its edge.
  - `FileNotesHistoryDialog` (`_show_history`, d521686cb8, task-34381). It was already a safe-dismissal modal, so it needed only a row and an edge.
  - The edge `LibraryNotesController.handle_library_notes_lasting_folder_requested` had gone stale while the walk was aborting earlier. 67bfde41d1 made that picker a `SelectDirectory`, not a `FileOpen`. The edge is corrected.
  - The table count goes 21 -> 24 and the edge count 35 -> 38.
- **AC#1.** All 203 nodes pass at `-n 4` (45 s, load 26).
- **AC#2, negative controls.** Dropping the `SkillRecoveryReviewModal` row brings back "unresolved modal constructor in supported presenter: …LibraryScreen._review_restored_skill_trust (SkillRecoveryReviewModal(review, paths[1], current))". Dropping only its edge fails the bidirectional check with "undeclared Library modal edges: …_review_restored_skill_trust…".
- **AC#3, lane.**
  - The file goes at the end of dreams-boundary's "Run Dreams and Guardian boundary contracts" step, as the brief prefers. It is not on `pr-fast-lane`, and not in the UI census, whose entries are kept near 30 s; this file is about 2 min serially, and its gesture test alone is 43 s.
  - Recent job times (read-only `gh run view --json jobs`, PR runs 42060638628, 42018794073 and 41989185012): this job took 6.7, 6.2 and 7.1 min of its 20.
  - The file with the step's exact command (`pytest Tests/UI/test_library_modal_dismissal.py --timeout=180 --tb=short`), 3 runs: 140.2, 114.6 and 101.6 s (203 passed each, load 20-34 on 18 CPUs).
  - The whole step in workflow order (`dreamstep.sh`): 675 passed in 383 s.
  - Pins updated: `DREAMS_BOUNDARY_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` and the step argv in `Tests/CI/test_response_rules_critical_gate.py`. `Tests/CI`: 1211 passed, 1 skipped.
  - For the record, with the task's own command (`-n 4`): 45 s at load 26. The task measured 28.2 s at load 14.8 on dev.
- **Covering.** `test_notes_recovery_controls.py`, `test_modal_wide_tier.py`, `test_css_consolidation_recovery_dialogs.py` and `test_css_parse_cache_modal_probe.py` all pass. `Tests/Architecture/test_timer_path_static_update_inventory.py` has 3 failures, and the same 3 fail on `fu-base-r10` (Plugins, provider detection and project-index roots).
- The File Notes User Guide's **Review…** row says how the pairing review closes. A lesson is added to `lessons-testing-evidence.md`: a walk that asserts on its first gap hides the next ones.
- Files: `Tests/UI/test_library_modal_dismissal.py`, `tldw_chatbook/Widgets/Library/notes_recovery_dialog.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `backlog/docs/lessons-testing-evidence.md`, `Docs/User_Guide/library/file-notes.md`.

