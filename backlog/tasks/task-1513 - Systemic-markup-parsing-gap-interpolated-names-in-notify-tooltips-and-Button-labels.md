---
id: TASK-1513
title: >-
  Systemic markup-parsing gap: interpolated names in notify, tooltips, and
  Button labels
status: Done
assignee:
  - '@claude'
created_date: '2026-07-30 14:00'
updated_date: '2026-10-10 21:47'
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
- [x] #4 Required CI rejects new or grown production markup-census sites
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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

Post-merge review follow-up (2026-10-10; base bade7c9a9d)
1. Add red-first guard tests for unescaped from_markup inputs and runtime Button.label assignments, plus safe literal/Text/escaped controls.
2. Exempt only inert Text construction; census label assignments as well as Button construction, keeping inherited unsafe sites explicit.
3. Review every newly detected inherited site before extending the exact census; prove newly introduced unsafe sites still fail the guard. Run targeted script/widget tests and preflight.
ADR required: no
ADR path: N/A
Reason: complete the existing syntactic guard and its tests; no runtime/UI architecture or dependency change.

Independent review correction (2026-10-10)
1. Add red-first direct and nested notify from_markup regressions with markup=False, plus the actual census-gate failure and plain-string exemption controls.
2. Restrict markup=False exemption using the existing prior-parser helper, then run targeted guard tests and preflight.
ADR required: no
ADR path: N/A
Reason: complete the existing guard exemption; no runtime or UI boundary change.

Required CI integration completion (2026-10-10)
1. Reopen this task and add AC4 for the existing production markup-census guard to reject new/grown sites in required CI.
2. Register the guard after timestamp writers in the workflow CHECKERS contract first and observe the unchanged workflow fail that contract.
3. Add the same guard to the existing derived-artifacts job between timestamp writers and the UI census with the existing !cancelled() policy.
4. Run Tests/CI, the actual markup guard, preflight and focused formatter/static checks; preserve the distinct 231-test integration receipt and append new evidence. Keep In Progress pending independent review of this step.
ADR required: no
ADR path: N/A
Reason: routine wiring of the existing guard into its existing required job; no lane, status-context, runtime or dependency choice changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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

Post-merge review completion (2026-10-10)
The guard now inspects from_markup interpolation and label assignments. Plain Text/escape_markup wraps remain inert only when they do not first evaluate a nested markup parser. Direct and nested parser regressions, escaped/literal controls, assignment controls, and actual census-gate exit assertions were red before the fix and pass afterward.
Reviewed baseline growth: exactly 41 inherited occurrences in 36 symbols at bade7c9a9d: 40 label assignments plus WatchlistSourcePickerDialog.compose's from_markup call over pre-escaped locals. The TSV adds only those exact rows, with an explicit provenance header; no production UI source changed and no new unsafe runtime values were admitted. The scanner remains syntactic and existing pins remain migration work items, not safety approvals.
Combined guard/media/widget/fusion/citation run: 218 passed; post-format changed-module plus scoped-media/custom-pipeline/chat-context run: 49 passed. Black 25.1.0 and focused Ruff 0.13.1 error checks passed; preflight passed (543 markup occurrences pinned). No full-suite run. Independent review pending; task remains In Progress until that review.
ADR required: no; correction of the existing syntactic guard and test coverage, with no runtime/UI architecture change.

Independent review correction completed (2026-10-10)
notify(..., markup=False) now remains exempt only when its message expression does not first evaluate from_markup. The existing helper detects both direct and nested parser evaluation; plain-string opt-outs and escaped/literal parser inputs remain exempt. The four new count/gate regression cases failed before the fix (4 failed, 33 passed), then the guard and real widget parser pins passed together (50 passed).
Black 25.1.0 and focused Ruff 0.13.1 error checks passed; preflight passed with the same 543 pinned occurrences and no census edits. The existing evaluation-order lesson now records this second independently reproduced gap. No full suite or production UI changes. Independent re-review remains pending; task stays In Progress.
ADR required: no; correction of an existing syntactic guard exemption, with no runtime/UI boundary change.

Independent immutable review closeout (2026-10-10)
Reviewer review_3029 cleared committed head 4d0130496caf35f76d61ec987f794420414ed5fc after reviewing the post-merge follow-up and its correction; no actionable findings remain in the authorized scope. All acceptance criteria are checked, self-review and independent review are complete, and the task is Done. Prior red/green receipts remain above; targeted checks, formatting, focused lint, whitespace checks and preflight passed. No full-suite run or push.
ADR required: no; routine completion of the existing bug-fix/guard boundaries.
The final independent probe covered eight cases: direct/nested/outer-escape parser hazards and plain markup=False, inert Text, literal/escaped parser exemptions. The correction run passed 50 tests, separately from the earlier 218-test base-fix and 49-test post-format runs. The census remains 543, including the exactly reviewed 41 inherited occurrences across 36 symbols; these remain cleanup work items, not runtime safety approvals.

Local reviewed-stack integration (2026-10-10)
stack_parent: 595618f4592a3495a251ae0e187980b793c12e74
Verification head: 7d9b32474269b0b30a39db72c5e01de011a4036d (rebased C follow-up before this evidence-only commit). The rebase's only conflict was the lessons append; complete upstream lesson text and the reviewed markup lesson are preserved. All follow-up production/test changes matched the previously reviewed tree exactly. Config-fixture ownership and trace-recovery changes from the stack remain intact.
Fresh combined interaction run: 231 passed in 20.47s across the guard, real widget parser pins, real SQLite media dedup, fusion, citation capture, and TestMediaLegScopeEnforcement/TestCustomPipelineInheritance/TestChatEntryPointScopedE2E. This is a separate run from the earlier 218 base-fix, 49 post-format and 50 correction receipts.
Black 25.1.0 checked all four follow-up Python files; focused Ruff 0.13.1 and stack-diff whitespace checks passed. Complete preflight passed (543 markup occurrences pinned; 175 Tests/UI gate entries). No full suite, push or PR creation. Task remains Done; no runtime behavior was changed during integration.
ADR required: no; rebase and verification of previously reviewed routine fixes, with existing boundaries unchanged.

Required CI integration implemented (2026-10-10)
The maintainer owner note above is now fulfilled: the existing derived-artifacts required job invokes python scripts/check_markup_interpolation.py between timestamp writers and the UI census with the same !cancelled() policy. The workflow CHECKERS contract registers it in the matching position; no lane, required status-context, runtime or dependency choice changed.
RED before the workflow edit: 2 failed / 20 passed in the contract file, specifically the absent markup command and 11 checker steps versus 12. GREEN after the edit: all Tests/CI passed (749 passed in 8.44s). This is separate from the earlier 231-test combined stack verification and the retained 218/49/50 receipts.
The actual production guard passed with the unchanged 543 pinned occurrences. Complete preflight passed (175 UI gate entries); Black 25.1.0 checked the modified inventory block, focused Ruff 0.13.1 error checks and whitespace checks passed. AC4 is checked. Independent review of this new workflow step is pending, so task remains In Progress. No full suite, remote Actions run, push or PR creation.
stack_parent: 595618f4592a3495a251ae0e187980b793c12e74
ADR required: no; routine integration of this existing guard into its existing required job. The inherited 41-occurrence census limitation remains unchanged.

Required CI independent review closeout (2026-10-10)
Reviewer review_3029 cleared immutable commit 123ee2a7db312cce8b0a3e5379047bf33e13b1df: verification mode, checker order, failure propagation and !cancelled() are correct; required contexts and lanes remain unchanged. No actionable findings remain in this scope. All four acceptance criteria are checked; task is Done.
The distinct required-CI receipt remains 749 Tests/CI cases passed in 8.44s after the contract RED (2 failed / 20 passed). This is separate from the 231-test stack interaction run and the earlier 218/49/50 receipts. Guard, preflight and focused static checks passed as recorded above. No full suite, remote Actions run or push.
stack_parent: 595618f4592a3495a251ae0e187980b793c12e74 (the reviewed integration boundary; retained for the later carry onto the merged sequence).
ADR required: no; routine integration of the existing guard into its existing required job.

Reviewed sequence carry qualification (2026-10-10)
stack_parent: 3810f8aab9a8c318003281f51f5e47b8e1b72570 (exact drop boundary for later publication onto the actual merged sequence).
Verification source: cf57afa64d373517788c5ee3419495f18e1a6ea3. Carried the six follow-up patches from c08aa049ed2b7d50188e63a105d15e74de60c715 / parent 595618f4592a3495a251ae0e187980b793c12e74 without conflicts or runtime edits. Complete --no-renames correspondence: ten own paths, nine identical nonlesson blob/mode entries, and exact upstream-plus-own lesson union; all six range-diff patches equal. Source proof: /private/tmp/pr3040-3810-correspondence.json and pr3040-3810-range-diff.log.
Fresh functional interaction qualification: 231 passed in 24.71s in the same guard/widget/media/fusion/citation and selected scoped-media/custom-pipeline/chat cases. Separately, all Tests/CI passed: 749 in 9.97s. Both XML receipts have zero failures, errors and skips; logs have no warnings. These are distinct from the historical 218/49/50, 231 in 20.47s and 749 in 8.44s receipts above. Logs/XML: /private/tmp/pr3040-3810-targeted.{log,xml} and pr3040-3810-ci.{log,xml}; complete counts in pr3040-3810-evidence-summary.json.
Preflight passed: 543 markup pins (396 notify, 84 Button labels, 63 tooltips), 175 UI gate entries, all derived guards green. Black 25.1.0 checked four functional Python files and the modified CHECKERS block; syntax and focused Ruff 0.13.1 fatal checks passed for all five owned Python files. Whitespace checks passed and all 5104 task files were readable. Receipts: /private/tmp/pr3040-3810-preflight.log and pr3040-3810-static.log.
Independent reviewer review_3029 cleared immutable cf57afa64d373517788c5ee3419495f18e1a6ea3 onto the exact 3810 parent: source correspondence, connected callers, required failure-propagating guard and final preflight remain correct; no actionable introduced findings. All acceptance criteria remain checked and the task remains Done. This closeout changes metadata only; verified source is unchanged.
The census still includes exactly 41 inherited occurrences across 36 symbols (40 label assignments and one pre-escaped Watchlist parser). Pins are baseline migration debt, not runtime safety approvals; the scanner is syntactic without dataflow. No broad UI cleanup, full suite, hosted Actions, push or PR creation was performed. The preparatory 175-entry local gate is not an exact published-head 178-entry or hosted-CI qualification. Publication remains after original PR3038 merges; later reviewed architecture-copy changes arrive via the actual base.
ADR required: no. ADR path: N/A. Reason: source-equivalent carry and fresh verification of existing routine fixes; no schema, authority, runtime, lane or required-context decision changed.
<!-- SECTION:NOTES:END -->
