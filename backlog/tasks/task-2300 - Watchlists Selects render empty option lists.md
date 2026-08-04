---
id: TASK-2300
title: Watchlists Selects render empty option lists
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT (2026-08-04, live tmux, fresh profile): two Selects on the Watchlists
screen open with no options at all. The Items tab's status filter opens an
empty floating overlay (a bare border, nothing selectable), and the New Rule
form's condition Select displays "No items". Both are dead controls; the
items filter one compounds TASK-2301 into ingested/ignored items being
unreachable. Likely one root cause (option population), possibly interacting
with the recently-added PruneSafeSelect guard — diagnose before fixing.

UAT findings F30 (critical), F36.

## Acceptance Criteria (the what)

- [ ] The Items status filter opens a populated option list covering every
      item status the backend can produce, and picking one filters the list.
- [ ] The New Rule condition Select offers the real condition vocabulary.
- [ ] The root cause is identified and recorded in the task notes (including
      whether PruneSafeSelect was involved), with a regression test that
      fails when option population breaks again.
- [ ] Verified live in a real terminal, not only under pytest.

## Implementation Plan

1. Diagnose empirically before touching anything: mount the production
   Watchlists screen with the production stylesheet, expand
   `#items-status-select`, and read the compositor's painted rows (not the
   widget's `option_count`, which can be right while the screen is wrong).
2. Establish whether `PruneSafeSelect` is involved by measuring `_pruning` /
   `_closing` and the overlay's option count at the moment of expansion, and
   record the answer in the notes either way.
3. Fix the mechanism that actually destroys the options, at the layer it
   lives in, following the TASK-1160 precedent for the same app-wide rule.
4. Regression test that reads the RENDERED rows through the real compositor,
   so it fails again the moment options stop reaching the screen -- an
   `option_count` assertion would have stayed green through this defect.
5. Verify live in a real terminal at 235x52.
