---
id: TASK-1513
title: >-
  Systemic markup-parsing gap: interpolated names in notify, tooltips, and
  Button labels
status: Done
assignee: ['@claude']
created_date: '2026-07-30 14:00'
updated_date: '2026-07-31 03:50'
labels:
  - evals
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the 2026-07-30 Evals UAT fix batch (Task 2 review, reproduced against installed Textual). `App.notify()`, tooltips, and `Button(label=…)` all parse Rich markup by default; unbalanced markup (e.g. a bare `[/]` from a user-controlled name such as an imported filename stem) raises MarkupError and can crash the app at render time. The batch fixed the Evals screen's four toasts (markup=False), escaped the primary-action tooltip/label restores, and escaped the rail's RUN-row labels — but the gap is repo-wide: zero `notify(..., markup=False)` calls existed anywhere before the batch, and the Evals rail's bench/dataset/classic row labels still interpolate names unescaped (confirmed hazard, currently unreachable only because bench/dataset names are constant or hex-suffixed). Cosmetic sub-item: an escaped name renders a literal backslash in markup=False Statics (`Run a\[b]: Blocked`) — pick one consistent convention at the seam.

**Update (task-1482 Task 1, 2026-07-30):** the Evals-package surfaces named above are now hardened, ahead of the bench-authoring program that makes bench/dataset names user-typed. `library_rail.py`'s `_bench_row_label`/`_classic_row_label`/`_dataset_row_label` now `escape_markup(...)` (mirroring `_run_group_row_label`'s existing fix); `bench_editor.py`'s name/description/dataset-line/probes-line Statics and `snippet_editor.py`'s dataset-name heading now pass `markup=False`; `notify_mixin.py`'s shared `_notify` (used by `LibraryRail`, `ResultsGrid`, and `SnippetEditor`) now passes `markup=False` on both its `app_instance.notify` and `self.app.notify` call sites. All four changes are regression-tested (`Tests/UI/test_evals_empty_states.py`, `test_evals_bench_editor.py`, `test_evals_snippet_editor.py`), each test confirmed red against pre-fix code first. Remaining scope for THIS task = every screen outside the Evals package, plus AC1's repo-wide convention/lint — neither addressed by the above.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A repo-wide convention exists (helper or lint) for user-derived text reaching notify/tooltips/labels
- [x] #2 The Evals rail's bench/dataset/classic row labels are safe for markup-metacharacter names
- [x] #3 A regression test pins at least one representative surface per widget kind (toast, tooltip, Button label)
<!-- AC:END -->

## Implementation Plan

Premise re-verified at branch base `cddc89d3e7` (2026-10-06):

- AC2 was ALREADY satisfied by the task-1482 batch named in the 2026-07-30
  update — `library_rail.py`'s `_bench_row_label`/`_classic_row_label`/
  `_dataset_row_label` all `escape_markup(...)` at this base; verified by
  reading, no new work needed.
- AC1 was NOT satisfied: `escape_markup` existed as a helper but nothing
  documented or enforced the rule (no markup rule in CLAUDE.md/AGENTS.md,
  no lint), and 507 unescaped interpolated sites remained across the three
  widget kinds (AST census at base: 399 notify, 44 Button label, 64
  tooltip). A repo-wide fix of all 507 is out of scope per the task's own
  update note; the census ratchet is the sanctioned mechanism.
- AC3 had Evals-package pins only (test_evals_empty_states.py et al.).

1. New census-ratchet lint `scripts/check_markup_interpolation.py` +
   `scripts/markup_interpolation_census.tsv` (mirrors
   check_timestamp_writers.py): AST scan for runtime-text interpolation
   (f-strings, `%`, `.format`, `+` concat) reaching `.notify()` without
   `markup=False`, `Button(label=...)`, and `tooltip=`/`.tooltip=` sites,
   with `escape_markup`/`Text(...)` exemptions. New/grown site fails.
   Wired into `scripts/preflight.sh` (workflow files are out of scope for
   this branch; derived-artifacts.yml step flagged for the maintainer).
2. CLAUDE.md: "Markup in notify / tooltips / Button labels" section — the
   one-convention-per-seam rule (markup=False for notify; escape_markup for
   tooltips/labels) + pointer to the guard. This is where the next author
   sees it.
3. Three representative non-Evals fixes, one per widget kind (each was
   unescaped at base): ChatbookCreationWindow's "Creating chatbook '{name}'"
   toast → markup=False; skills_screen's local-skill "Use" tooltip →
   escape_markup(name); library_skills_canvas' import-review
   `Review "{name}"…` Button label → escape_markup(import_review_name).
   Census shrank 507 → 504.
4. Tests: `Tests/Scripts/test_check_markup_interpolation.py` (scanner
   predicate + exemptions + end-to-end green/fail) and
   `Tests/UI/test_markup_interpolation_widget_pins.py` (one pin per widget
   kind through the REAL parsers, each with a negative control proving the
   parser rejects the unescaped shape).

## Implementation Notes

Everything above implemented as planned; no deviations.

Evidence (venv 3.12.13, `-p no:xdist`):

- AC1: `python scripts/check_markup_interpolation.py` →
  "399 notify_interp, 44 button_label_interp, 64 tooltip_interp
  occurrence(s) (504 pinned). check_markup_interpolation: OK" (census
  regenerated AFTER the three fixes; was 507 at base). A planted new site
  fails (test_guard_fails_on_one_new_site). Premise verification on
  installed Textual 8.2.8: `Content.from_markup("a[/]b")` raises
  MarkupError; `Button('a[/]b')` raises MarkupError AT CONSTRUCTION
  (`_button.py` `Content.from_text`); tooltips render through the Tooltip
  Static (`screen.py` `tooltip.update(content)`).
- AC2: verified at base, no code change — `library_rail.py:196,200,205`
  escape all three label helpers (task-1482's work, still green under
  `Tests/UI/test_evals_empty_states.py`).
- AC3: `Tests/UI/test_markup_interpolation_widget_pins.py` — 9 passed;
  toast (real `app.notify(..., markup=False)` through run_test, literal
  `[/]` preserved), tooltip (real mounted widget + Content.from_markup
  round-trip), Button label (real mounted Button, rendered Content plain
  keeps raw brackets); each with a negative control. Plus
  `Tests/Scripts/test_check_markup_interpolation.py` — 16 passed.
- Touched-file suites, my-change vs HEAD (file-swap A/B, no stash),
  identical both sides: `Tests/UI/test_library_skills_canvas.py` +
  `test_library_skills_browse_controller.py` 32 failed / 136 passed BOTH
  sides; `Tests/UI/test_library_skill_import_trust_journeys.py` +
  `test_chatbook_wizard_open_folder.py` (combined) 5 failed / 10 passed
  BOTH sides (wizard file standalone errors with
  `RecoveryRequired: raw_source_selection_changed` at BOTH sides —
  order-dependent pre-existing guard, not this diff).

ADR required: no — lint + docs convention + three call-site escapes; no
schema, sync, boundary, or contract change (mirrors the ADR-less
check_timestamp_writers pattern's guard mechanics; the convention record
the AC asks for lives in CLAUDE.md and the census header).
Files: `scripts/check_markup_interpolation.py` (new),
`scripts/markup_interpolation_census.tsv` (new, 377 rows),
`scripts/preflight.sh`, `CLAUDE.md`,
`tldw_chatbook/UI/ChatbookCreationWindow.py`,
`tldw_chatbook/UI/Screens/skills_screen.py`,
`tldw_chatbook/Widgets/Library/library_skills_canvas.py`,
`Tests/Scripts/test_check_markup_interpolation.py` (new),
`Tests/UI/test_markup_interpolation_widget_pins.py` (new).

Owner note: `.github/workflows/derived-artifacts.yml` needs a matching
step (workflow files were out of scope for this branch):
`python scripts/check_markup_interpolation.py`.
