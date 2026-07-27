---
id: TASK-1034
title: >-
  Results grid intermittently drops its column header and [warned] canary marker
status: To Do
assignee: []
created_date: '2026-07-27 16:00'
labels:
  - evals
  - bug
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during UAT of the Evals screen on `origin/dev` (155574902), driven live against llama.cpp.

The results table renders in **two structurally different ways within the same view**, and which one you get is not under the user's control:

- **Boxed, no header.** Rows render inside a drawn box with no column header at all, e.g. `│The protestors were [neutral]  "mente"  49%`. The target column is unlabelled and the `[warned]` canary marker is absent.
- **Unboxed, with header.** The same grid renders `Snippet | Sample target (llama.cpp) f0fded1f [warned]` above the rows.

Reproduced repeatedly. Immediately after "Create sample bench" completes the grid appears in the boxed/headerless form; at one point during lens interaction it switched to the headed form; a later systematic pass over all five lenses showed `header=0` on every one, including Entropy which had shown a header minutes earlier. Clicking a cell did not restore it. So it is **not lens-dependent** — it is some other state we did not isolate.

Two things are lost in the headerless state, and both matter:

1. **Column identity.** With one target the reader can infer it; with several the numbers become unattributable, which is the whole point of the grid.
2. **The `[warned]` canary marker.** This is the column-level signal that the target preflighted with a degenerate canary. Losing it silently removes the interpretive guardrail the design added on purpose.

Worth investigating whether the boxed form is a different widget (an error/placeholder container) rather than the `DataTable` with `show_header` off — the row prefixes differ (`│ │The…` boxed vs `│  The…` headed), which suggests two render paths rather than one widget in two states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The results grid always renders its column header, in every lens and every entry path
- [ ] The `[warned]` marker is present whenever the run's canary is degenerate
- [ ] The two render paths are reconciled to one, or the second is explained and made deliberate
- [ ] A test fails if the header is absent after a fresh mount
<!-- AC:END -->
