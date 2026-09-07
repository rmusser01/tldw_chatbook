---
id: TASK-1513
title: >-
  Systemic markup-parsing gap: interpolated names in notify, tooltips, and
  Button labels
status: To Do
assignee: []
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
- [ ] #1 A repo-wide convention exists (helper or lint) for user-derived text reaching notify/tooltips/labels
- [ ] #2 The Evals rail's bench/dataset/classic row labels are safe for markup-metacharacter names
- [ ] #3 A regression test pins at least one representative surface per widget kind (toast, tooltip, Button label)
<!-- AC:END -->
