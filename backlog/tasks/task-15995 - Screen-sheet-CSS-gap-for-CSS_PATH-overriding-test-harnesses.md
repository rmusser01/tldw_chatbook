---
id: TASK-15995
title: 'Screen-sheet CSS gap for CSS_PATH-overriding test harnesses'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:10'
labels:
  - tests
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Before consolidation, Textual auto-registered a pushed screen's class-level CSS in `_load_screen_css` regardless of the harness's own `CSS_PATH`. After TASK-15450, the 7 `BUNDLED_SCREEN_CSS` modals get their CSS from the app-level sheets — `Tests/UI/consolidated_css.py` carries them (fixed in the PR's m8 round), but a harness subclass that declares its OWN `CSS_PATH` overrides that list, and its comment (~:10-16) claims this 'matches what those harnesses had before', which is inaccurate: such a harness pushing one of the 7 modals now mounts it with NO CSS where it used to get the class CSS automatically. 33 test modules combine ConsolidatedCSSApp with a CSS_PATH; currently latent (only `test_library_prompts_canvas.py` also pushes a consolidated modal, and all 49 modal-adjacent tests pass) — but it is a vacuous-pass trap for the next geometry-asserting test. Fix direction: make ConsolidatedCSSApp merge the screen sheets into subclass CSS_PATH declarations (e.g. via `__init_subclass__` or a get_css_path override), and correct the comment. Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A ConsolidatedCSSApp subclass with its own CSS_PATH still loads the generated screen sheets when pushing a BUNDLED_SCREEN_CSS modal
- [x] #2 A test pins that behavior (a modal pushed under a CSS_PATH-carrying harness has its styles applied)
- [x] #3 The inaccurate comment is corrected
<!-- AC:END -->

## Implementation Plan

1. Read Textual 8.2.8's `App.__init__` CSS_PATH resolution (`css_path = css_path or
   self.CSS_PATH`, then `_css_path_type_as_list` + `_make_path_object_relative`).
   Confirmed: a subclass shadows `CSS_PATH` via ordinary attribute lookup for the
   class-attribute path, and a `css_path=` constructor kwarg short-circuits the
   `self.CSS_PATH` branch entirely for the kwarg path -- both need to be handled by
   the same mechanism, so it must live in `__init__`, not `__init_subclass__` (which
   only rewrites the class attribute and would miss a subclass instantiated with an
   explicit `css_path=` kwarg).
2. Add `ConsolidatedCSSApp.__init__` that computes the effective incoming CSS path
   (`css_path` kwarg if given, else `self.CSS_PATH`, mirroring Textual's own
   resolution) and rebuilds the final `css_path` passed to `super().__init__()` as
   `[scoped_sheet] + [effective entries, minus the two sheets if already present] +
   [self_sheet]` -- same pair, same order as `build_css.screen_css_paths` (the single
   ordering source per TASK-15450), matching production's
   `[scoped, bundle, self]` bracket with the subclass's own CSS_PATH occupying the
   "bundle" slot.
3. Correct the inaccurate module comment (~:10-16 claims the override "matches what
   those harnesses had before", which is false -- they used to get the class CSS
   auto-registered).
4. Add a new test module (`Tests/UI/test_consolidated_css_harness.py`, additive/new
   file to avoid touching `test_widget_css_consolidation.py`, which a concurrent
   session (task-15994) just changed on origin/dev) with a `ConsolidatedCSSApp`
   subclass that overrides `CSS_PATH` (mirroring the ~27 real combiners) and pushes
   `NoteSelectionDialog` (one of the 7 `BUNDLED_SCREEN_CSS` modals). Assert a
   computed geometry consequence of its screen CSS (`#note-selection-container`'s
   `width: 80` rule) rather than absence-of-exception. Confirm the test fails
   against the pre-fix module (born red), then passes after the fix.
5. Run the new test, `test_widget_css_consolidation.py`, `test_selection_dialogs.py`,
   `test_library_prompts_canvas.py`, and a sample of the heaviest real CSS_PATH
   combiners (test_library_shell.py, test_mcp_inspector.py, test_console_workbench_contract.py,
   test_evals_screen.py, test_home_screen.py, test_mcp_rail.py) to confirm no
   regressions. ruff check + format on touched files only.

## Implementation Notes

**Mechanism.** `ConsolidatedCSSApp.__init__` now intercepts CSS_PATH resolution
before delegating to `App.__init__`: it computes `effective = css_path if
css_path is not None else self.CSS_PATH` (the exact same `css_path or
self.CSS_PATH` logic Textual's own `App.__init__` uses), then calls a new
`_merge_screen_css_paths()` helper that returns `[scoped_sheet, *own_entries,
self_sheet]` (own entries de-duplicated against the two sheets), and passes
that explicit list to `super().__init__(css_path=...)`. This was chosen over
`__init_subclass__` because a class-attribute rewrite only fixes the
`CSS_PATH`-class-attribute path -- it cannot see a `css_path=` constructor
kwarg, which short-circuits `self.CSS_PATH` entirely in Textual's own
resolution (`textual/app.py:731`, `css_path = css_path or self.CSS_PATH`).
Intercepting in `__init__` sees whichever form actually won, for both forms,
uniformly. Verified this composes correctly with the real usage pattern in
this codebase: every one of the ~27 real combiners sets `CSS_PATH` as a class
attribute (none use the `css_path=` kwarg today), and a subclass's own
`__init__` commonly calls `super().__init__()` with no arguments (e.g.
`test_library_prompts_canvas.py`'s `_CanvasHost`) -- both keep working
because `*args`/`**kwargs` pass through untouched and only `css_path` is
special-cased. `_merge_screen_css_paths` was sanity-checked directly (no
override, single-string override, multi-entry list override, `None`) to
confirm the middle slot preserves the caller's own entries in order without
duplicating the bracketing sheets. Ordering matches production exactly by
construction: both `ConsolidatedCSSApp.CSS_PATH` and the merge helper source
the pair from `build_css.screen_css_paths()`, the same single ordering
authority `TldwCli.CSS_PATH` uses to bracket its app bundle
(`[scoped, bundle, self]`) -- a subclass's own CSS_PATH now occupies the same
"bundle" slot.

**Born-red evidence.** `Tests/UI/test_consolidated_css_harness.py` (new file,
kept separate from `test_widget_css_consolidation.py` since a concurrent
session's task-15994 had just changed that file on origin/dev) pins two
paths: a `CSS_PATH`-class-attribute override (mirroring the real ~27
combiners, pointed at the real app bundle) and a `css_path=` constructor
kwarg override (no real harness does this yet, but the mechanism must
compose with it). Both push `NoteSelectionDialog` (one of the seven
`BUNDLED_SCREEN_CSS` modals) and assert a computed geometry consequence of
its screen-sheet CSS (`#note-selection-container`'s `width: 80` rule)
rather than absence-of-exception. Ran both tests against the pre-fix module
first: both failed at exactly the expected point (`region.width == 120`, the
full unstyled `Container` default, instead of `80`); the kwarg test's own
custom-stylesheet assertion (which comes first) passed even pre-fix,
isolating the failure specifically to the screen-sheet drop. After the fix,
both pass.

**Regression coverage.** An AST scan of `Tests/UI/*.py` for classes whose
bases literally include `ConsolidatedCSSApp` AND that declare their own
`CSS_PATH` in the same class body found 27 real combiners across 22 files
(the naive `grep` list of 33 files included several false positives, e.g.
`test_console_command_popup.py`, where the two symbols appear in the same
file but never combine on one class). Ran every one of those 22 files
(~2,000+ individual tests) against the fixed module -- all pass except a
small number of pre-existing failures, each independently confirmed
pre-existing by swapping in the original (unfixed) `consolidated_css.py` and
re-running the identical failing test(s), which reproduced the identical
failure/traceback both times (edit-based restore back to the fix
immediately after each check): 3 in `test_library_prompts_canvas.py`
(unrelated focus-race assertions in save/import flows), 1 in
`test_console_workbench_contract.py` (a `ConsoleHarness`-based test with no
`CSS_PATH` override at all, so untouched by this change either way), 3 in
`test_console_native_transcript.py` (unrelated `pilot.pause` timing waits),
and 3 in `test_library_shell.py` (1 was a full-suite-only ordering flake that
passed in isolation; 2 -- a tab-order keyboard-capability test -- reproduced
identically pre- and post-fix, and independently confirmed unaffected since
neither generated screen sheet contains any selector touching
`LibraryScreen`/`nav-lab`/`library-notes-new`). No new failures were
introduced by the fix anywhere in this coverage.

**Files changed:**
- `Tests/UI/consolidated_css.py` -- the merge mechanism and corrected comment.
- `Tests/UI/test_consolidated_css_harness.py` -- new, the AC2 pinning tests.
