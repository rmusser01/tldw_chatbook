---
id: TASK-1481
title: >-
  Evals copy and polish batch from UAT
status: In Progress
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small fixes found by live UAT (2026-07-30), batched:

1. Detail/inspector copy says "library rail" but the rail is labeled "Catalog" (`evals_screen.py` empty-state and blocked-reason strings).
2. The same strings use ASCII `--` where the rail copy uses real em-dashes.
3. The Δ baseline lens on a single-target run renders every cell as the word "baseline" with an empty Spread column; it should state that the lens needs a second target to compare against.
4. The snippet table header advertises columns (`#  Snippet  Group  Chars  Flags`) that do not align with the actual row layout (meta renders as a right-aligned blob).
5. Stale comment `lab_frame.py:91` claims EvalsScreen binds Escape to a back action; it does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] No user-facing Evals string refers to the "library rail"; copy names the rail as it is labeled
- [ ] User-facing Evals copy uses em-dashes consistently (no ASCII `--` in rendered strings)
- [ ] A single-target run's Δ baseline lens states the lens needs at least two targets
- [ ] The snippet table's header matches what its rows actually render
- [ ] The stale Escape comment in `lab_frame.py` is corrected
<!-- AC:END -->
