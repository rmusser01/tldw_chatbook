---
id: TASK-1036
title: >-
  Degenerate-canary warning is not surfaced where results are read
status: To Do
assignee: []
created_date: '2026-07-27 16:00'
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
- [ ] A degenerate canary is visible on the run view without requiring a cell click
- [ ] The wording explains the consequence, not just the state
- [ ] The bench view and run view agree on how prominently it is presented
<!-- AC:END -->
