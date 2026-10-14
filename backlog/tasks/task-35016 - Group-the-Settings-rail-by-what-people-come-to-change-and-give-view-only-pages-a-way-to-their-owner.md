---
id: TASK-35016
title: Group the Settings rail by what people come to change, and give view-only pages a way to their owner
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- information-architecture
dependencies:
- TASK-33007
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings rail lists 33 categories in six groups that misfile them: Agents sits under Troubleshooting and Console Behavior under Interface. Six pages Settings fully edits (RAG, Schedules, Dreams, Guardian, Image Gen, Video Gen) are hidden in a collapsed 'Domain Defaults' group whose tooltip calls everything in it read-only. The seven '(view)' pages owned by other screens say 'change this in <Destination>' but offer no way there, and they show developer roadmap notes both on the page ('Planned: ...') and in the Inspector ('Read-only: ...'). Re-running setup lives only under Troubleshooting. The guard meant to make each new rail category a deliberate decision is red: it asserts 31, but the enum has had 33 since Dreams and Guardian landed, and a sibling domain-contract guard went red in the same commit. So the rail grew without anyone deciding it should. This task applies the owner-approved rail map (OD-1, recommended option A: Core, Look & Feel, Console & Agents, Retrieval & media and Data & Privacy open; Maintenance and Owned elsewhere collapsed) and the view-only page treatment (OD-2a). It closes verified findings D2, D-shell-design-M2 and D-shell-design-M3. It lands after PR #3057 (TASK-33007), which gives settings_screen.py its size-ratchet row. Design, pinned tests and live checks are in backlog/docs/plan-2026-10-14-settings-ia-page-order.md.

Plan: `backlog/docs/plan-2026-10-14-settings-ia-page-order.md`. Findings closed: D2, D-shell-design-M2, D-shell-design-M3 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rail shows the owner-approved groups in the approved order, and each of the 33 Settings categories appears in exactly one of them
- [ ] #2 Only the four Maintenance pages (Diagnostics, About, Internal Prompts, Advanced Config) and the seven view-only pages owned by other screens sit in collapsed groups; every other category sits in an always-open group
- [ ] #3 At 211x44 and 235x52 with the real stylesheet and both collapsed groups closed, every row of the open groups and both group toggles are painted inside the rail's visible window, and the rail does not scroll
- [ ] #4 A collapsed group shows its rows when its toggle is pressed, when one of its categories is the active one (navigation, restored state or deep link), and while a category search matches one of its rows; clearing the search folds it again unless the user opened it
- [ ] #5 No rail group heading or tooltip calls a group read-only unless every page in that group is view-only
- [ ] #6 Each of the seven view-only pages (Artifacts, Roleplay, Skills, Watchlists, Workflows, MCP Defaults, ACP Defaults) offers an Open button naming its owner which, pressed from the keyboard, lands on that owner's screen (Artifacts and Skills on their Library views)
- [ ] #7 None of the seven view-only pages shows developer roadmap copy, either in the detail pane or in the Inspector
- [ ] #8 Overview offers Run setup wizard, which opens the same re-run wizard as the Diagnostics button
- [ ] #9 The category search echo names the match's new group (for example 'Appearance › Reduce motion (Look & Feel)')
- [ ] #10 The Settings category-count guard and the domain-contract guard both pass with all 33 categories and 13 domain contracts counted, and the count guard's history names the Dreams and Guardian step that took it from 31 to 33
- [ ] #11 settings_screen.py does not grow and its size-ratchet row equals its measured size; the boot CSS bytes, UI-ready module census, screen pre-import payload and bare-type CSS subject ratchets do not rise
- [ ] #12 The User Guide's category map lists all 33 categories under the shipped groups, its group-named section headings match the rail, and its view-only page table lists seven pages, each with its Open button
- [ ] #13 The new rail tests are enrolled in the UI Fast Lane census and pass there
<!-- AC:END -->
