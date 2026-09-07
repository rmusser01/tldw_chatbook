---
id: TASK-15997
title: 'Extend the invalid-CSS parse guard beyond the package: Tests/ and Helper_Scripts/'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - tests
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15450's `test_every_class_level_css_block_parses_as_a_stylesheet` runs every class-level CSS block in the tldw_chatbook package through `Stylesheet.parse()` (the API that actually raises — `textual.css.parse.parse` only collects errors), and on its FIRST run found two live crashers nobody knew about (`audio_troubleshooting_dialog`, `dictation_performance_widget` — plus the two selection dialogs that motivated it). Nothing sweeps `Tests/` or `Helper_Scripts/` (including the custom splash-card examples) for the same defect class: an invalid property in any sheet poisons the whole stylesheet at parse time. Extend the guard's walk to those trees (reusing its block-extraction helper), fix what it finds, and record what it found. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The parse guard covers class-level CSS blocks under Tests/ and Helper_Scripts/
- [x] #2 Any newly-found invalid blocks are fixed (removal or translation, with the never-parsed rationale applied as in TASK-15450)
- [x] #3 Notes record the found-count so the sweep's value is measurable
<!-- AC:END -->

## Implementation Plan

1. Confirm the existing guard's block-extraction helper: `_class_css_blocks()` in
   `Tests/UI/test_widget_css_consolidation.py` walks `_PACKAGE_ROOT` (the
   `tldw_chatbook` package) for class-level `DEFAULT_CSS`/`CSS`/`BUNDLED_CSS`/
   `BUNDLED_SCREEN_CSS` string literals via AST, applying `widget_css.EXCLUDED_DIRS`.
2. Generalize `_class_css_blocks()` to take an optional `root` (default: the
   package root, preserving every existing call site's behaviour byte-for-byte)
   and `excluded_dirs` so Tests/ and Helper_Scripts/ can reuse the exact same
   scan/AST logic instead of a forked copy.
3. Run the generalized helper + the existing `Stylesheet.parse()` check against
   `Tests/` and `Helper_Scripts/` as a one-off probe to get the found-count and
   see whether anything is actually broken today.
4. Extract the existing test's parse-check body into a shared
   `_assert_class_css_blocks_parse()` helper (used by the original package-scope
   test too, so there is one check, not two), adding an explicit
   per-`(module, class_name)` allowlist mechanism (with a required reason
   string) for deliberately-invalid negative-test fixtures, plus a staleness
   check so an allowlist entry that has been fixed (or renamed away) is flagged.
5. Add a new parametrized test sweeping `Tests/` and `Helper_Scripts/` through
   the shared helper.
6. Add a permanent born-red regression test that seeds an invalid CSS block
   into a synthetic `tmp_path` tree shaped like each newly-covered root and
   asserts the shared check catches it -- proving the guard is not a no-op
   because both real trees currently come back clean.
7. Fix whatever the real sweep finds (expected: nothing, based on the probe;
   record the found-count either way).
8. Run the new/changed tests plus the full `test_widget_css_consolidation.py`
   module; ruff check/format on touched files; record notes.

## Implementation Notes

**Approach.** `_class_css_blocks()` (`Tests/UI/test_widget_css_consolidation.py`)
was the guard's one block-extraction helper -- an AST walk over
`_PACKAGE_ROOT` collecting every class-level `DEFAULT_CSS`/`CSS`/`BUNDLED_CSS`/
`BUNDLED_SCREEN_CSS` string literal. It took a `root: Path | None = None`
(defaulting to the package, so all three existing call sites are byte-for-byte
unchanged) and an `excluded_dirs` override, so `Tests/` and `Helper_Scripts/`
reuse the identical scan instead of a forked copy. The parse-check body of
`test_every_class_level_css_block_parses_as_a_stylesheet` was pulled out into
`_assert_class_css_blocks_parse(blocks, *, allowlist)`, shared by the original
package-scope test and the new sweep, so a fix to the check itself (or to the
`Stylesheet.parse()` call it makes) never has to be applied twice.

**Found-count (AC3).** Swept once while adding the new test:
- `Tests/`: **28** class-level CSS blocks (all plain `CSS` on test-harness
  `App`/`Screen`/`ModalScreen` subclasses across 20 files under `Tests/Chat/`
  and `Tests/UI/`, plus one top-level `Tests/test_enhanced_filepicker.py` and
  `Tests/test_splash_fullscreen.py` each). **0 failed** `Stylesheet.parse()`.
- `Helper_Scripts/`: **0** class-level CSS blocks. The custom-splash-card
  examples (`Helper_Scripts/Examples/custom_splash_cards/*`) are `.toml` data
  files, not Python classes with a CSS attribute; the tree's few `.py` files
  that mention "CSS" (`Helper_Scripts/UI/visualize_textual_*.py`) don't declare
  one. **0 failed** (vacuously).

So AC2 ("fix what's found") had nothing to fix -- no crashers exist in either
tree today. That null result is exactly why the born-red test exists: it seeds
one throwaway module per tree (`tmp_path`, never written into the real trees)
with the identical defect class TASK-15450 found in the package
(`font-size: 10;` -- not a Textual property) and asserts
`_assert_class_css_blocks_parse` raises for both, proving "0 found" means
"nothing is broken" and not "the sweep never ran". Ran red before the fix (no
`root` parameter existed to point the extractor anywhere but the package) and
green after; see `test_css_parse_guard_catches_seeded_invalid_blocks_in_newly_covered_trees`.

**Allowlist mechanism.** No `Tests/`/`Helper_Scripts/` fixture today declares
deliberately-invalid CSS as a negative test, but the sweep needed a real
mechanism ready for when one shows up rather than a silent directory/filename
skip: `_KNOWN_INVALID_CSS_FIXTURES: dict[tuple[str, str], str]` maps
`(module, class_name) -> reason`. An allowlisted block is excluded from the
failure list but is asserted to still actually fail (a "stale" entry -- one
whose fixture now parses cleanly -- fails the test, so a fixed bug can't hide
behind a forgotten allowlist row), and an allowlist entry that matches no
scanned block at all (renamed/removed fixture) also fails the test. Currently
empty; `test_css_parse_guard_catches_seeded_invalid_blocks_in_newly_covered_trees`
exercises the underlying check (not the allowlist itself, since there is
nothing yet to allowlist).

**Files changed.**
- `Tests/UI/test_widget_css_consolidation.py`: generalized `_class_css_blocks`
  (`root`/`excluded_dirs` params); extracted `_assert_class_css_blocks_parse`
  with the allowlist + staleness/unused checks; added
  `_KNOWN_INVALID_CSS_FIXTURES`, `_TESTS_ROOT`, `_HELPER_SCRIPTS_ROOT`;
  added `test_class_level_css_blocks_outside_the_package_parse` (parametrized
  over `Tests`/`Helper_Scripts`) and
  `test_css_parse_guard_catches_seeded_invalid_blocks_in_newly_covered_trees`.
  No production code touched -- there was nothing to fix.

**Verification.** `Tests/UI/test_widget_css_consolidation.py`: 20 passed (was
17; +3 for the two new parametrize cases and the born-red test), no skips.
`pytest Tests/UI/ -q --collect-only`: 12555 tests collected, no collection
errors (confirms no naming collisions or import breakage from the refactor).
`ruff check` and `ruff format --check` clean on the touched file.
