---
id: TASK-32357
title: Library Details rail panel ships internal vocabulary and numbers
status: Done
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 08:31'
labels:
  - library
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Server sync WIP · local only', 'Handoff · 0 eligible · 1 blocked · not in this workspace', and DB sizes ('Chats/Notes 8.2MB' on a profile with one note) are engineering status shown to a first-timer (A caps 20/21). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The panel's lines read as user outcomes (e.g. '1 note can't be used in Console yet — add it to this workspace'); 'WIP' does not appear in shipped copy
- [x] #2 DB sizes move behind a diagnostics disclosure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests for both Details strings and for the Diagnostics disclosure.
2. Replace 'Server sync WIP · local only' and drop WIP from the sibling tooltip.
3. Re-render the handoff line as the task the reader can act on (eligibility logic untouched).
4. Move the DB-size rows under a closed-by-default Diagnostics header in the rail; keep #library-details-db-sizes queryable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: two Details strings now name outcomes. 'Server sync WIP · local only' became 'Everything here is stored on this machine · syncing to a server isn't available yet.', and the sibling Create-local-workspace tooltip lost its 'WIP' with it, so the word is gone from this panel's shipped copy. The Handoff row's 'Handoff · 0 eligible · 1 blocked · not in this workspace · <remedy>' is now 'N item(s) can't be used in Console yet · <reason> · <remedy>' -- task-32230's eligibility derivation (which rows are blocked, their shared reason, the matching remedy) is untouched; only the sentence carrying it changed, and the zero-blocked case still renders its bare count.

AC#2: the DB-size rows moved under a closed-by-default 'Diagnostics' `DestinationRailSectionHeader` inside Details. `#library-details-db-sizes` and its -1/-2 continuation ids are unchanged and still queryable while the section is closed, so every existing pin holds; the screen's open-Details refresh patcher now mounts freshly computed rows into the diagnostics body instead of after `#library-details-body`. The open/closed state is rail-local and unpersisted (`LibraryRail.diagnostics_open`) and the rail answers its own toggle press: the screen's generic handler persists the five `LibraryRailPreferences` sections by name and diagnostics deliberately is not one of them -- every visit starts closed, which is what a diagnostics panel should do.

Live-verified at 235x52 and 100x30 (the 100x30 pair re-taken after the storage note moved beside the button it captions, so the captures show the shipped Actions order -- `caps/cap-14`, `caps/cap-15`): closed 'Diagnostics ▸' on arrival, opening it shows 'DB sizes / Prompts … / Chats/Notes 8.2MB / Media …' (the A cap 21 number, now out of the first read), the Actions note reads the new sentence, and the Handoff row reads '1 item can't be used in Console yet · not in this workspace · Copy or link it into this workspace' on a starter profile.

Modified: tldw_chatbook/Widgets/Library/library_rail.py, tldw_chatbook/UI/Screens/library_screen.py (`_workspace_handoff_summary_label`, the Details widgets block, `_refresh_library_details_db_sizes`), Docs/User_Guide/library.md. Tests: Tests/UI/test_library_crit10_notes_details.py; updated pins in test_library_crit9_rail.py (3 handoff strings + the DB-size width pin opens the disclosure) and test_post_release_workspaces_library_depth.py.

Not done here: `#library-notes-template-section`'s TCSS rule is now dead (task-32356 dropped that widget), but `_agentic_terminal.tcss` is owned by other branches this wave, so the rule is left for a follow-up rather than forcing a bundle regeneration into three PRs.
<!-- SECTION:NOTES:END -->
