---
id: TASK-15994
title: 'CSS consolidation test hygiene: cross-stream ordering pin and error-message scope fence'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two weaknesses in `Tests/UI/test_widget_css_consolidation.py` found in review. (1) `test_base_class_blocks_precede_their_subclasses` (~:222-256) builds its order from the CONCATENATION of the self and scoped sheets, but the two streams carry different tie-breakers (0 vs -1,000,000) that decide precedence regardless of position — so a cross-stream index comparison pins nothing, and the test also only inspects direct syntactically-named bases (a grandparent inversion passes). (2) The mounted dialog test (~:414) fences its scope with `"clear" not in str(raised)` — a substring match on an error message; the load-bearing StylesheetParseError assertion runs first so the pin holds, but the fence is fragile (dissolves entirely once TASK-15992 fixes the underlying dialog crash). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ordering test compares only within a stream (or asserts the tie-breaker relation directly) and covers transitive bases
- [x] #2 The mounted-dialog test's exception tolerance is either removed (after TASK-15992) or keyed on exception type and site, not message substring
- [x] #3 Both tests still born-red against a seeded violation
<!-- AC:END -->

## Implementation Plan

1. Rebase the worktree onto `origin/dev` -- it started 5 commits behind (missing
   PR #1673/#1674), so TASK-15992's escape-hatch removal wasn't actually present
   yet despite the setup's assumption. Re-verify AC2 post-rebase.
2. Trace the actual load-bearing tie-break relation: read `build_css.py`'s
   `widget_defaults_sources`/`SCOPED_DEFAULTS_TIE_BREAKER` and Textual's own
   `Stylesheet._check_and_refresh`/`Styles.extract_rules`/`DOMNode._get_default_css`
   to confirm precisely how a specificity tie is broken (tie_breaker first, then
   source/text order within a single tie_breaker) -- this determines what the
   fixed test must actually assert.
3. Rewrite `test_base_class_blocks_precede_their_subclasses` (AC1):
   - Build the self-stream and scoped-stream banner-order dicts SEPARATELY
     (never concatenated).
   - Replace the direct-AST-bases-only scan with a real transitive-MRO walk
     (import each consolidated class, walk `__mro__`) so a grandparent
     inversion is caught even when an intermediate class declares no CSS of
     its own.
   - Compare base-vs-subclass ordering only within a shared stream.
   - Factor the checking logic into small helpers so the seeded-violation test
     can drive them directly with synthetic inputs.
4. Verify AC2 is satisfied by TASK-15992 (PR #1673) at the rebased HEAD --
   confirm the substring escape hatch is gone and the dialog test now
   re-raises unconditionally; do not touch that test.
5. Add a permanent regression test for AC3 that seeds one synthetic instance of
   each of the two defects (cross-stream conflation; syntactic-direct-bases-only)
   and proves, with real pytest assertions: the OLD algorithm (reconstructed
   inline for comparison) passes vacuously, and the extracted helpers used by
   the real test catch it.
6. Run the targeted test file, ruff check/format on touched files, record
   evidence, update the task file, commit.

## Implementation Notes

**Setup correction.** The worktree's checked-out HEAD was actually 5 commits
behind `origin/dev` (missing PR #1673 `task/15992-burn` and PR #1674
`task/15768-burn`), despite the task setup assuming it already included
PR #1673. Rebased `task/15994-burn` onto `origin/dev` (clean, no conflicts)
before doing anything else, since AC2's verification depends on that PR's
change actually being present.

**AC2 (verified, not touched).** At the rebased HEAD,
`test_selection_dialog_opens_without_a_stylesheet_error`
(Tests/UI/test_widget_css_consolidation.py:375) no longer has the
`"clear" not in str(raised)` substring fence -- TASK-15992 (PR #1673,
commit `3e70fb4d0`) replaced the dialogs' nonexistent `Vertical.clear()` call
with `remove_children()` and changed the tail of the test to
`if raised is not None: raise raised` (unconditional re-raise), with a
docstring explaining the change. Confirmed by reading the file at HEAD; left
untouched per the assignment's scope.

**AC1 -- the actual load-bearing relation.** Traced Textual's tie-break chain
(`textual/css/dom.py:_get_default_css`, `styles.py:extract_rules`,
`stylesheet.py:_check_and_refresh`): a rule's full comparison key is
`(is_default, is_important, *specificity, tie_breaker)`, compared with `max()`.
Originally each ancestor class got its OWN stylesheet source with a distinct
`tie_breaker` (`0` for the class itself, `-depth` per ancestor), so a subclass
beat its base on ties by that raw number, regardless of text position. The
consolidation (`css/build_css.py`) collapses ALL classes' self-stream rules
onto one shared source (`tie_breaker=0`) and all scoped-stream rules onto
another (`SCOPED_DEFAULTS_TIE_BREAKER=-1_000_000`). Two same-stream rules that
still tie on specificity therefore fall through to source order: `max()`
scans `reversed(self.rules)` and keeps the first-seen max, so on an exact tie
the LAST rule in source/text order wins. That makes the load-bearing
invariant: **within a single stream, a base class's block must precede its
subclass's**; a pair split across streams is already decided outright by the
differing tie-breakers, so comparing their raw text positions (the old test's
approach, via `"".join(self_text, scoped_text)`) pins nothing.

Rewrote `test_base_class_blocks_precede_their_subclasses` around three new
helpers: `_stream_order` (banner index within ONE generated sheet, called
separately for self and scoped), `_transitive_base_pairs` (imports each
consolidated class via `importlib` and walks its real `__mro__`, not just its
syntactic `ast` bases, so a grandparent inversion is caught even when an
intermediate class in the chain declares no `BUNDLED_CSS` of its own), and
`_ordering_problems` (flags a `(class, base)` pair only when both appear in
the SAME stream and the base's index is greater). Verified against the real
package: 46 `BUNDLED_CSS` classes, importing all of them cleanly (0 failures);
the only real ancestor pairs today are `BaseAppScreen` <- `{ChatScreen,
LibraryScreen, MCPScreen, PersonasScreen}` (all direct, depth 1), already
correctly ordered in both streams.

**AC3 -- born-red, both directions, both defects.** Added two permanent
regression tests using synthetic/monkeypatched inputs (no generated sheets
touched):
- `test_ordering_check_catches_a_cross_stream_conflation_the_old_index_missed`
  seeds a case where `BaseWidget`/`SubWidget` are correctly ordered in the
  self stream but `SubWidget`'s scoped block sits BEFORE `BaseWidget`'s (a
  real inversion). Reconstructs the retired merged-index algorithm inline and
  asserts it passes vacuously (`old_problems == []`), then asserts the real
  `_ordering_problems` catches it (`new_problems` non-empty).
- `test_ordering_check_catches_a_transitive_base_inversion_the_old_scan_missed`
  builds a real `Grandparent -> Middle -> Grandchild` chain where `Middle`
  declares no CSS. Asserts the direct-`__bases__`-only reconstruction of the
  retired scan finds nothing (`Grandparent` is not a direct base of
  `Grandchild`), then asserts the transitive `__mro__` walk finds it and that
  `_ordering_problems` flags a seeded same-stream inversion.

Mutation-verified the seeded tests are not vacuously true: temporarily
replaced `_ordering_problems`'s body with the retired cross-stream-merge
algorithm and reran -- `test_ordering_check_catches_a_cross_stream_conflation_
the_old_index_missed` correctly went red (`AssertionError` on
`assert new_problems`), while the other two tests still passed. Reverted via
`Edit` (no generated files or `git checkout` involved) and reran the full file
to confirm 17/17 green again.

**Testing.** `PYTHONPATH=<worktree> .venv/bin/python -m pytest
Tests/UI/test_widget_css_consolidation.py -q` -> 17 passed (15 original + 2
new). `ruff check` and `ruff format` on the touched file only -- both clean
after one formatter pass (two long lines wrapped). `pytest Tests/UI/
--collect-only -q` -> 12,532 tests collected with no errors (confirms no
import-time breakage elsewhere). No production code was changed; the diff is
confined to `Tests/UI/test_widget_css_consolidation.py`.

**Files changed:** `Tests/UI/test_widget_css_consolidation.py` only.
