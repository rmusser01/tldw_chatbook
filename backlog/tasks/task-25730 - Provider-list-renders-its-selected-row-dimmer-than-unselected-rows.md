---
id: TASK-25730
title: Provider list renders its selected row dimmer than unselected rows
status: Done
assignee: []
created_date: '2026-08-31 05:10'
updated_date: '2026-08-31 13:28'
labels:
  - console
  - ux-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On the first-run provider step the chosen provider is marked with bold and underline, which is a sound non-colour signal, but its text renders at a lower value than its unselected siblings so the selected row reads as the least prominent one. No leading marker is used, which is the cheapest and clearest selection affordance in a terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The selected row is at least as prominent as unselected rows
- [ ] #2 Selection is carried by a leading marker in addition to text styling
- [ ] #3 Selection state is apparent without comparing rows side by side
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
MEASURED AND REAL, BUT MINOR -- deliberately not fixed; needs a decision I should not take alone.

Root cause: components/_lists.tcss applies one app-wide rule to
.option-list--option-highlighted (shared with SelectionList buttons, DataTable
headers and Tree cursors) setting 'color: $text; text-style: bold underline'.
In the first-run provider list the resting rows paint brighter than $text --
measured fg rgb(233,236,238) resting vs rgb(225,225,225) highlighted -- so
highlighting LOWERS text brightness by ~3%.

Why I left it: (a) the delta is 1.03x, far below any perceptual or WCAG
threshold; (b) selection is already carried by bold+underline, a non-colour cue
the app applies consistently across OptionList, SelectionList, DataTable and
Tree -- so this is not a colour-only signal; (c) the only clean fix is editing
that shared rule, which repaints selection in every list, tree and table in the
app for a 3% gain. Not a trade I should make unilaterally.

My filed suggestion to add a leading marker (a '>' or bullet on the selected
row) should ALSO be declined as written: it would deviate from the app-wide
selection language above and make this one list inconsistent with every other.

If anyone does pick this up, the change belongs in the shared rule (make the
highlighted colour at least as bright as resting), not in a provider-list
one-off.
<!-- SECTION:NOTES:END -->
