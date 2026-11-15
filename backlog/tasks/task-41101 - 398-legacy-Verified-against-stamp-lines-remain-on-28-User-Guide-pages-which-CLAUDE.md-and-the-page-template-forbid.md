---
id: TASK-41101
title: "398 legacy 'Verified against' stamp lines remain on 28 User Guide pages, which CLAUDE.md and the page template forbid"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- docs
- user-guide
priority: low
references:
- Docs/User_Guide/_template.md
- Docs/User_Guide/library/notes.md
- Docs/User_Guide/library/media-and-conversations.md
- Docs/User_Guide/index.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's. Counts rechecked after the restack onto dev ecad404bd4 (review r14cl-rev1).

**Why.** CLAUDE.md ("UI changes") and `Docs/User_Guide/_template.md:49-51` say a page carries no "Verified against" lines: verification belongs in the task's Implementation Notes, and parallel PRs appending stamps at the same spot caused most User Guide sync conflicts. Old stamps keep inviting new ones, and they tell readers a stale branch and date.

**What is there.** `grep -c "Verified against"` over `Docs/User_Guide` counts 408 lines on 34 files. One is the template's own rule, and nine are on the five pages other tasks own (below), which leaves 398 lines on 28 pages for this task. The largest: `library/notes.md` 73, `library/media-and-conversations.md` 68, `library/import-and-export.md` 49, `library.md` 48, `library/file-notes.md` 26, `settings.md` 25, `library/search-and-rag.md` 18, `console.md` 14; `index.md` has one at `:184-186`.

**Not this task.**
- The four Roleplay pages (`roleplay-chat-dictionaries.md` and its three sub-pages, one stamp each) belong to TASK-33910.18 AC#4.
- `First_Run_Setup.md` (5 lines) belongs to TASK-34100.15 AC#7, under its own evidence rule: its stamps' evidence moves into the named tasks' notes, and TASK-33008 AC#8 drops its stamp requirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No page under Docs/User_Guide other than the four Roleplay pages and First_Run_Setup.md contains a 'Verified against' stamp; the template's rule line stays
- [ ] #2 Any fact a removed stamp carried that the page body does not (for example a retired screen named only in the stamp) is kept in the body or dropped on purpose, and the notes say which
- [ ] #3 A Tests/Docs check fails when a 'Verified against' stamp is added to a User Guide page, shown red with one planted stamp; the five pages other tasks own are exempt by name only until those tasks land
<!-- AC:END -->
