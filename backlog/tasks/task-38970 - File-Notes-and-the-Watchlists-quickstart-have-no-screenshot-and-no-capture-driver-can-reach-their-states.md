---
id: TASK-38970
title: "File Notes and the Watchlists quickstart have no screenshot and no capture driver can reach their states"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- docs
- screenshots
priority: low
references:
- Docs/User_Guide/library/file-notes.md
- Docs/User_Guide/watchlists-quickstart.md
- Docs/User_Guide/_template.md
- scripts/regen_approval_card_svg.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b and the TASK-36521 branch, 2026-11-05.

TASK-36521's branch merges before this one and deletes two images that no longer matched the app:
- `Docs/User_Guide/images/library/file-notes.svg`, referenced from `library/file-notes.md`;
- `images/watchlists/console-workflow-complete.svg`, referenced with its caption from `watchlists-quickstart.md`.

The branch removes those references too. Only frozen QA evidence JSON under `Docs/superpowers/qa/` still names the files.

Neither picture can be retaken yet because there is no driver for either state:
- File Notes needs a folder-picker flow that chooses a notes folder.
- The Watchlists image needs a completed Console workflow run.

Capture today is the hand recipe in `Docs/User_Guide/_template.md:55-134`. The repo's only capture script is `scripts/regen_approval_card_svg.py`, which captures the approval card through a harness. TASK-36521's capture scripts stayed in a scratchpad. TASK-24632's workflow image was a copied UAT capture, not a driver.

Merge note: the TASK-38757 branch adds 16 lines to `library/file-notes.md` and still has the `![File Notes workspace](../images/library/file-notes.svg)` line (`:53`). Whichever of the two branches merges second must keep the image removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After both branches have merged, no User Guide page references a missing image; a Docs test resolves every image link under Docs/User_Guide and fails when one link is pointed at a missing file (negative control)
- [ ] #2 A checked-in capture driver reaches the File Notes workspace with a folder chosen and the Watchlists Console workflow in its completed state, using a scratch profile and no real provider
- [ ] #3 File Notes and the Watchlists quickstart each show a screenshot again, taken from the current app at 235x52 with that driver and with cdnjs font URLs stripped; if a state cannot be reached, the notes say which and why
<!-- AC:END -->
