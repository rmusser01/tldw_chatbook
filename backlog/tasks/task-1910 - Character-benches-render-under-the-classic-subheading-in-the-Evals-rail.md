---
id: TASK-1910
title: Character benches render under the classic subheading in the Evals rail
status: Done
assignee: []
created_date: '2026-08-02 04:13'
labels:
  - evals
  - character-probe
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A character-probe bench created via '+ New character bench' appears in the rail under the '- classic -' subgroup, which is defined app-wide as read-only pre-word-bench rows. The diamond marker distinguishes it, but the grouping still misinforms: a bench the user just created is labelled as a legacy read-only row.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character-probe benches render under their own subgroup label, not 'classic'
- [x] #2 Word benches and true classic tasks keep their existing grouping
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

`library_rail.py`'s `_benches_section_body` now splits the
non-word-bench rows: character-probe benches render under their own
`─ character ─` separator (same inert-Static shape as the classic one)
with their own row prefix (`evals-rail-row-benches-character-`), while
the classic subgroup keeps true pre-word-bench tasks AND skill-eval
benches (whose plain-name classic placement was the task-32888 brief's
own ruling) under `─ classic ─` unchanged. The `◆` marker stays on the
character rows, belt to the label's braces (and still marking probe
sets in the Datasets section). The header count is unchanged (all rows
under the Benches header).

Tests: `test_a_character_bench_renders_under_its_own_subgroup_not_
classic` and `test_word_benches_and_true_classic_tasks_keep_their_
existing_grouping` (new), plus the two pre-existing character-bench
rail-click/delete tests updated to the new prefix (they addressed the
character row via the classic prefix by construction).

Modified files: `tldw_chatbook/UI/Evals/library_rail.py`,
`Tests/UI/test_evals_screen.py`.
