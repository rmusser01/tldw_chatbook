---
id: TASK-1091
title: >-
  Watchlist names keep leading whitespace, and the delete copy says "1 source are"
status: Done
assignee: []
created_date: '2026-07-28 04:00'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small copy/validation defects seen in the third Watchlists UAT (`origin/dev` `e82ac1b18`).

**Names are not trimmed.** Renaming a watchlist to `" Daily"` (leading space) stored it verbatim, and the tree renders the space as extra indentation, so the row no longer lines up with its siblings:

```
│ ▸     Daily  0           │
│ ▸  Security Watch  0     │
```

Create has the same gap. A name that is entirely whitespace is presumably also accepted, which would produce an unclickable, unnameable row.

**The delete confirmation mispluralises.** With one source attached it reads:

> Its **1 source are** not deleted. They stay in Watchlists and appear under Unassigned unless they also belong to another watchlist.

Should be "is" for one. The rest of that copy is genuinely good — it explains the consequence clearly, which is why the grammar stands out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leading and trailing whitespace is stripped from a watchlist name on create and on rename
- [ ] #2 A name that is empty or whitespace-only is rejected with a visible reason, not silently accepted
- [ ] #3 The delete confirmation reads correctly for one source and for several
- [ ] #4 Tests cover the whitespace-only name and the single-source wording, proven to fail against current code
<!-- AC:END -->

## Implementation Notes

One of the two reported defects was real. The other was a misdiagnosis, and the underlying visual
had a different cause — which is fixed too.

**Real: the delete copy.** With one source attached it read *"Its 1 source are not deleted. They
stay in..."*. The noun was already pluralised; the verb and pronoun were not. Split into
`watchlist_delete_consequence()` so the wording is testable without driving a modal, with both
branches asserted — including that each keeps the Unassigned explanation, which is the part a user
actually needs and the easiest thing to lose while fixing grammar.

**Not real: name trimming.** `WatchlistBundleService.create` and `rename` both strip before storing,
and a whitespace-only name already raises. Verified directly against the service before changing
anything: `"  Daily  "` stores as `Daily`, `"   "` is rejected. AC#1 and #2 were already satisfied.

Note the near-miss: both methods *validate* on `name.strip()` while passing the unstripped `name`
onward, which reads like the classic "checks the stripped value, stores the raw one" bug. It is not
— `_unique_name()` strips again and returns that. Worth stating because the shape invites a fix that
is not needed.

**What the UAT actually saw.** Every tree node is a `Button`, and Textual centres a Button's label,
so a short name sits further right than a long one:

    ▸     Daily  0
    ▸  Security Watch  0

That is label centring, not stored whitespace. Fixed by left-aligning the tree labels in
`features/_watchlists.tcss`.

This is the second UAT finding in the programme that did not reproduce (after TASK-996's byte-offset
click). The trimming behaviour is now asserted so it is not filed a third time.

Modified: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated bundle).
Added: `Tests/Watchlists/test_watchlist_name_and_copy.py`.

