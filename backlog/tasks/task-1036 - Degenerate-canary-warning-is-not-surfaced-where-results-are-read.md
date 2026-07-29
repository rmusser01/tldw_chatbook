---
id: TASK-1036
title: Degenerate-canary warning is not surfaced where results are read
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 16:00'
updated_date: '2026-07-27 23:23'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during UAT of the Evals screen.

The word bench has a real interpretive hazard: raw mode against an instruct-tuned model is out-of-distribution, so the numbers can be well-formed and meaningless. The design added a **distribution sanity canary** precisely for this, and on the **bench** view it is presented well — a recovery callout naming the target and explaining the consequence: *"a large divergence in its column may reflect that, not the prompt."*

On the **run** view, where the user actually reads results, that explanation is absent.

Walked live: after "Create sample bench" the grid showed `The protestors were [neutral] → "mente" 49%`. `"mente"` is a nonsense continuation — the canary was degenerate — but nothing on screen said so. Searching the whole rendered view for "canary", "degenerate", "warn" or "out-of-distribution" matched **zero** times.

The signal does exist, but only two clicks deep and only as raw jargon: focusing a cell populates the Inspector with `canary degenerate` on a metadata line, alongside `K requested 20 · K returned …` and `truncated mass: 10.4%`. There is no sentence saying what that means or how it should change the reading.

So a first-time user runs the sample bench, sees a nonsense token with a confident-looking 49%, and has nothing telling them the setup is out-of-distribution. That is the exact misreading the canary was introduced to prevent, and the fix is placement and wording rather than new machinery — the verdict is already computed and already stamped on every cell.

See also the sibling defect where the `[warned]` column marker disappears entirely in one of the grid's two render paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A degenerate canary is visible on the run view without requiring a cell click
- [x] #2 The wording explains the consequence, not just the state
- [x] #3 The bench view and run view agree on how prominently it is presented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the shared degenerate-canary sentence in inspector.py (_recovery_callout_text) and the run view's compose() in results_grid.py.
2. Factor the sentence into a shared function (degenerate_canary_text) in results_grid.py, extended to name one or several warned targets with correct singular/plural grammar; have inspector.py call it for its single-target case.
3. In ResultsGrid.compose(), compute the warned targets from the run's stored preflight snapshot (in column order) and, only when non-empty, yield a .ds-recovery-callout Static (markup=False, matching the DataTable bracket-escaping trap already documented in this file).
4. Add the matching CSS rule for the new callout's id in css/features/_evals.tcss and rebuild the CSS bundle.
5. Add tests: names + consequence for a single warned target, absence for a clean run, naming multiple warned targets, and survival of a bracket-containing target name -- verify each by actually reverting the implementation, observing the failure, and restoring.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a run-view degenerate-canary callout using the same .ds-recovery-callout
pattern the bench view already used, sharing its exact wording rather than
duplicating it.

- results_grid.py: new degenerate_canary_text(target_labels) builds the
  sentence (singular/plural grammar, e.g. "steered and distilled preflighted
  with a degenerate canary: their plain-text continuations looked
  out-of-distribution... These targets are still runnable -- a large
  divergence in their columns may reflect that, not the prompt."). A
  single-element list reproduces the bench view's original sentence byte for
  byte. _warned_target_labels() reads the run's stored preflight snapshot in
  column order. ResultsGrid.compose() yields the callout (id
  evals-grid-canary-callout, classes="ds-recovery-callout", markup=False)
  right after the meta/state Statics, only when at least one target is
  warned -- no empty container otherwise.
- inspector.py's _recovery_callout_text now calls degenerate_canary_text([
  target_label]) for its is_warned branch instead of inlining the sentence,
  so the two views cannot drift apart again.
- Escaping: target names are free text that can contain "[...]" (the same
  DataTable/_safe_cell trap this file already documents); the callout is a
  Static with markup=False, not a DataTable cell, so no _safe_cell wrapping
  needed.
- Multiple warned targets: named via an Oxford-commaed join
  (_join_target_names), never collapsed to "a target".
- CSS: #evals-grid-canary-callout margin rule added to
  css/features/_evals.tcss (mirrors #evals-inspector-bench's own callout
  margin); bundle rebuilt via build_css.py and verified with
  check_bundle_sync.py.
- The callout is a plain Static yielded unconditionally in compose(),
  independent of the DataTable's own header rendering -- so it should stay
  visible even under TASK-1034's headerless render path, though TASK-1034
  itself was not reproduced or fixed here.

Tests (Tests/UI/test_evals_results_grid.py): 4 new tests -- names the target
and states the consequence; absent for a clean run; names every warned
target when several (asserts exact equality against degenerate_canary_text);
survives a target name containing markup. Each was verified by actually
reverting the corresponding implementation piece, confirming the specific
test failed with the expected error, then restoring (diffed byte-identical
after restore).

One revert caught a real trap: the initial escaping test used
`str(callout.renderable)`, mirroring other tests in this file, but this
project's own Textual compatibility shim (tldw_chatbook/__init__.py)
aliases `Static.renderable` to `.content` -- the raw, unparsed constructor
argument -- so it read back correctly regardless of whether markup=False
was actually applied. Removing markup=False did not fail the test. Fixed by
switching to `callout.visual.plain` (the actual parsed-visual path
Static.render() draws from), matching the pattern already used in
Tests/UI/test_library_ingest_canvas.py::test_error_and_warning_markup_is_escaped.
After the fix, removing markup=False correctly failed with
`AssertionError: assert 'steered [redacted]' in 'steered  preflighted with a
degenerate canary: ...'` (the bracket span silently stripped by Rich's
markup parser).

Full targeted suite: 182 passed (Tests/UI/test_evals_results_grid.py,
Tests/UI/test_evals_screen.py, Tests/Evals/word_bench).

Modified: tldw_chatbook/UI/Evals/results_grid.py,
tldw_chatbook/UI/Evals/inspector.py,
tldw_chatbook/css/features/_evals.tcss,
tldw_chatbook/css/tldw_cli_modular.tcss (rebuilt),
Tests/UI/test_evals_results_grid.py.
<!-- SECTION:NOTES:END -->
