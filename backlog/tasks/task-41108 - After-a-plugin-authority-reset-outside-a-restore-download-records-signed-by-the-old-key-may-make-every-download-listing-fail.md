---
id: TASK-41108
title: "After a plugin authority reset outside a restore, download records signed by the old key may make every download listing fail"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- investigation
priority: low
references:
- tldw_chatbook/Plugins/coordinator.py
- tldw_chatbook/Plugins/acquisition.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Reported by the round-14 plugin stream as unverified. If true, a user who reset authority and set it up again cannot open Candidates… or update a plugin until they delete files by hand.

**Evidence (read from the code; not reproduced).**
- Setup outside a restore (`coordinator.py:117-136`) refuses only when installations or data roots remain or `data/` exists; it does not move `acquisition-custody/` aside (only the post-restore branch does, `:155-157`). Download records with no installation behind them stay.
- `acquisition._inventory` (`acquisition.py:105-128`) verifies every record's MAC with the current authority (`authority.verify_acquisition_custody`) and raises for the whole inventory on any failure. Six callers list through it (`:200`, `:275`, `:357`, `:389`, `:402`, `:424`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test resets authority with a download record present, sets authority up again, and records whether listing fails
- [ ] #2 If it fails, old-key records are set aside or refused one by one so listing and new downloads keep working, with a test red before the fix
- [ ] #3 If it does not, the notes name what removes or re-signs the records
<!-- AC:END -->
