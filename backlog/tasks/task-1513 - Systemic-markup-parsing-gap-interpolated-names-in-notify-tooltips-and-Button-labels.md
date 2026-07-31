---
id: TASK-1513
title: >-
  Systemic markup-parsing gap: interpolated names in notify, tooltips, and Button labels
status: To Do
assignee: []
created_date: '2026-07-30 14:00'
labels:
  - evals
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the 2026-07-30 Evals UAT fix batch (Task 2 review, reproduced against installed Textual). `App.notify()`, tooltips, and `Button(label=…)` all parse Rich markup by default; unbalanced markup (e.g. a bare `[/]` from a user-controlled name such as an imported filename stem) raises MarkupError and can crash the app at render time. The batch fixed the Evals screen's four toasts (markup=False), escaped the primary-action tooltip/label restores, and escaped the rail's RUN-row labels — but the gap is repo-wide: zero `notify(..., markup=False)` calls existed anywhere before the batch, and the Evals rail's bench/dataset/classic row labels still interpolate names unescaped (confirmed hazard, currently unreachable only because bench/dataset names are constant or hex-suffixed). Cosmetic sub-item: an escaped name renders a literal backslash in markup=False Statics (`Run a\[b]: Blocked`) — pick one consistent convention at the seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A repo-wide convention exists (helper or lint) for user-derived text reaching notify/tooltips/labels
- [ ] The Evals rail's bench/dataset/classic row labels are safe for markup-metacharacter names
- [ ] A regression test pins at least one representative surface per widget kind (toast, tooltip, Button label)
<!-- AC:END -->
