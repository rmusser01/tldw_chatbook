---
id: TASK-35017
title: Open Console Behavior on chat defaults and fold agent, capture and replay sections into Advanced
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- console-behavior
- information-architecture
- ux-review-2026-10-14
dependencies:
- TASK-33007
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console Behavior opens on Goal runs, a feature added on 2026-10-11 that is off by default, drawn as five full-width three-row inputs. The section actually titled 'Start here' first appears at row 39 of 44 at 211x44 (row 38 of 52 at 235x52). About twenty sections share one page, and the related Agent run budget sits far down the page, away from Goal runs. DESIGN.md already asks that the common path stay focused and advanced controls stay easy to find, and Providers & Models follows that after PR #3057; Console Behavior does not. This task applies the order the owner chose on 2026-10-14 (OD-3a). The common sections lead: Global fallback defaults, Conversation context and memory, then the display controls. Agent limits (with Goal runs directly after Agent run budget), Permission summaries, Selection side chat, Local reasoning history and Exchange capture fold into closed one-row disclosures under an Advanced heading, each titled with its state. The category stays one draft with one Save/Revert. It closes verified findings D3 (Console part) and B16. The Providers & Models part of D3 and A4 is done by PR #3057 (TASK-33007.6), which this task builds on for the frameless one-row disclosure styling, the open-state memory and the shared fallback rows. Design, pinned tests and live checks are in backlog/docs/plan-2026-10-14-settings-ia-page-order.md.

Plan: `backlog/docs/plan-2026-10-14-settings-ia-page-order.md`. Findings closed: D3, B16, and D6's Goal runs rows (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, opening Console Behavior shows the Global fallback defaults section first, with its Temperature, Max tokens and Streaming rows painted without scrolling
- [ ] #2 The page reads top to bottom in the owner-approved order: Global fallback defaults, Conversation context and memory, Model thinking, Status row placement, Chat images, Rail presentation, Composer paste handling and Background effects, then an Advanced heading with closed disclosures for Agent limits, Permission summaries, Selection side chat, Local reasoning history and Exchange capture
- [ ] #3 Goal runs sits inside Agent limits directly after Agent run budget, and its five limits are one-row, label-left rows in the page's shared control column with every label shown whole
- [ ] #4 Each closed disclosure title is one row and names its on/off or mode state where it has one, and the title follows a staged edit, a save, a revert and an instant-apply write
- [ ] #5 A disclosure the user opened stays open across a save, a revert, and leaving the category and coming back in the same Settings visit
- [ ] #6 A '/' field search for a control inside a closed disclosure opens that disclosure and focuses the control
- [ ] #7 Save and Revert cover every staged Console Behavior value exactly as before, including values edited inside a disclosure that is now closed, and every instant-apply control still writes immediately
- [ ] #8 No section is titled 'Start here', and the 'Conversation context and memory' jump button and the 'Also in this category' line are gone
- [ ] #9 settings_screen.py shrinks and its size-ratchet row equals its measured size; the boot CSS bytes, UI-ready module census, screen pre-import payload and bare-type CSS subject ratchets do not rise
- [ ] #10 The User Guide's Console Behavior section lists the sections in page order, with Goal runs described under Agent limits and reached from F4
- [ ] #11 The new page-order tests run on every pull request and pass
<!-- AC:END -->
