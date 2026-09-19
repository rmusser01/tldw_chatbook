---
id: TASK-32625
title: >-
  Library Notes: the sync review spends three rows per file and reports skips at
  a different granularity from Import once
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-15 06:44'
updated_date: '2026-09-15 18:33'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A heuristic 8 and section 11, persona Alex, Obsidian workflow.

What happened. Wave 4 (task-32535, PR #2679) made the sync review honest -- it names files, folders and effects now, instead of 'Safe item N' -- and that is a real improvement both assessors confirm. What it did not get is Import once's density: the sync review spends about three screen rows per file row (A cap 49) where Import once fits 65 sources on one page with groups, counts, a 45-file run collapsed to a single openable row, and a reason on every skipped row (A cap 21).

The two also now disagree on granularity. Import once reports skips at FOLDER level ('vault/.trash', 'vault/Templates'); the sync review reports the same vault at FILE level ('.trash/Old idea.md'). Same content, two mental models, one screen apart -- a side effect of the per-file item_skips the wave added.

Cause PROVEN by capture. The right shape is A's improvement idea 7: one dense review component shared by both paths, which would fix the granularity split for free.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two review surfaces share a row renderer, or state why they differ
- [x] #2 The sync review fits a 54-file vault without three rows per file
- [x] #3 Skips are reported at one granularity across both paths
- [x] #4 Import once and lasting sync render a review row through one shared component (idea 7 of task-32627), or the task records the evidence for why the shared component is out of reach and the two agree on granularity anyway
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure what a safe review row actually costs on screen, and why.
2. Take the rows back to one each without losing the conflict rows' bodies.
3. Make skips agree with Import once's folder granularity.
4. Share what can honestly be shared of the row, and say plainly what cannot.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Where the three rows went.** Measured with a 54-file review composed at 100 columns: each safe row's own line (1) + the conflict-choices panel, which is EMPTY for a safe row but held open by `.library-notes-sync-conflict-choices { min-height: 1 }` (1) + `.library-notes-sync-review-row { margin: 0 0 1 0 }`, which exists to separate that panel from the next row (1). The panel is now shown only when it has something in it, and a row with no body drops its margin. 54 files now cost 54 rows: measured both the outer heights ({1}) and the screen distance between consecutive row tops ({1}), because the margin is exactly what an outer height does not include.

The margin is dropped INLINE rather than via a `-plain` CSS rule: the base rule's tokens are all `library-`, so `build_css` splits it out to `screen_agentic_library.tcss` (a screen's `CSS_PATH`), while a `-plain` modifier keeps its block in the boot bundle -- the two would be arguing across sheets. An inline style is the one tier above both, which is why `run_disclosure` already sets its own margin that way.

**Granularity (AC#3).** [CORRECTED in fix round 1 -- what follows is what the code does at head; the round-0 wording this replaces claimed a folder line 'at any length', which is the change round 1 reverted. See the fix-round-1 note below for why.] Import once reports a skip at folder level ('vault/.trash'); wave 4's per-file `item_skips` made this review report the same vault file by file. The run grouping the review already does (`uniform_runs`, keyed on folder + category + effect + destination) is exactly Import once's unit, and both paths now apply it through the SAME threshold: a skipped run of fewer than `UNIFORM_RUN_MIN` (8) files is listed one row per file, and a run of eight or more collapses into one `run_disclosure` summary naming the folder, the count and the reason. So the same vault reads the same way on both screens at every length. The group heading's count is unchanged, so 'Skipped (5)' still means five files.

**The shared component (AC#1 / AC#4, idea 7).** Honest answer: the two reviews now share FOUR of the five pieces of a row -- the path budget (`bounded_row_name`), the group heading (`group_heading`), the uniform-run collapse (`uniform_runs`/`run_disclosure`, shared by wave 4's task-32535) and, added here, the row LINE itself (`review_row_line`: 'name · what happens · where', each clause stripped of its own full stop). They do not share the row WIDGET, and should not yet: an Import row is a `Horizontal` carrying per-item Skip/Create new/Update controls, a sync row is a `Vertical` carrying conflict choices, a selected-choice line and a diff pane. Merging those is a bigger change than this task's density finding needs, and the granularity split -- the thing the shared component was wanted FOR -- is closed without it. Recorded as the evidence the AC asks for rather than claimed as done.

**Modified:** `Widgets/Library/library_notes_add_from_files_canvas.py`, `Widgets/Library/library_note_import_canvas.py` (`review_row_line`, used by both surfaces), `Tests/Widgets/Library/test_library_notes_w5_review_density.py` (new), `Docs/User_Guide/library/notes.md`.

**Red first:** row heights {2} against {1} (the outer height; the margin makes three on screen), and the folder-level skip rows absent entirely ([] against the two expected lines).
**Fix round 1 — AC#3 was ticked on half the evidence (review F1).** My round-0
change collapsed EVERY skipped run on the sync side to one folder line whatever
its length. Import once keeps `if len(run) < _UNIFORM_RUN_MIN` (8) and names the
files under that, so the two still disagreed -- **inverted** below eight rather
than closed. Worse, `test_both_reviews_report_a_skip_at_folder_granularity`
never composed an Import review at all despite its name: it asserted one side
twice and bought the claim.

Both paths now use the ONE rule they were always meant to share -- the same
`UNIFORM_RUN_MIN`, the same folder-keyed run -- so a skip reads identically on
both screens at every length: named files under eight, one collapsed folder
summary at eight and over. The sync-side special case is deleted, not extended;
this is the reviewer's own cheapest fix and it is a smaller diff than round 0's.

The pin is rewritten to earn its name: it composes BOTH canvases from one
fixture at 1, 4, 8 and 11 skipped files and asserts they name the same files and
collapse together. Reverting to round 0 it fails at **all four** lengths: at 1 and 4 with
`1 skipped files: sync names [], Import once names ['.trash/Old idea 0.md']`
-- the inversion, in the test's own words -- and at 8 and 11 on the collapse
assertion, because round 0 emitted a plain `Static` where Import once emits a
`run_disclosure`.

I first wrote "at 8 and 11 it passed even unfixed". That was wrong, and it was
wrong because I read the revert's FAILED list through `head -8` and inferred the
rest. Re-run without the pipe: `4 failed`. The sentence that IS true is about
BASE DEV, not round 0 -- on dev the two paths already agreed at 8 and 11, which
is why a one-sided test could sit there looking green. **Never read a FAILED
list through `head`**; the truncation is indistinguishable from a pass.

Density (AC#2) is unaffected: it is a claim about SAFE rows, and its pin now
scopes to them rather than counting the skipped ones too.
**Why a green checker did not catch the false guide line (fix round 2).** The
stanza promised `.trash · 4 files · Obsidian system folder` as the shape of a
skip row. `scripts/check_guide_claim_strings.py` passed it anyway, and would
pass it again: that line is **composed at runtime** by `review_row_line` from a
folder name, a count and an effect, so it exists nowhere in the source for the
checker to match. The gate proves a quoted string is EMITTED SOMEWHERE; it
cannot prove the sentence around it is true, and it is blind by construction to
any string the code builds rather than stores. A guide claim about composed copy
still needs a human or a pin. Not a miss by the gate -- a limit of it, worth
knowing before the next person reads green as verified.
<!-- SECTION:NOTES:END -->
