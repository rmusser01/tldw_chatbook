---
id: TASK-35019
title: Settings pages, banners and the User Guide speak plain language
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- copy
- docs
dependencies:
- TASK-35018
- TASK-35017
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings pages still show implementation vocabulary as user copy. The Agents banner reads 'agent_runs.db (SQLite) — immediate CRUD, no draft'. Speech prints the raw status 'not_tested'. Storage names the 'ChaChaNotes DB' and 'active handles'. Guardian and Dreams use raw TOML keys as row labels. Diagnostics and the domain pages carry roadmap and developer notes ('not available yet', 'Source 2: workflow Console handoff payloads', 'future defaults may cover…'). The raw CLI card shouts 'DANGER!!!' against the product's calm, confident voice. The Settings User Guide quotes much of this copy as it renders today. This task rewrites that copy in words and keeps each key in a closed disclosure. The danger warning keeps its full force through its card styling and disclosure lines. The task also brings the guide's quotes of that copy back in line with the app. Other guide drift has its own owner: the category map and view-only count (TASK-35016), the misplaced Goal runs section that names F9 (TASK-35017) and the Tab sentence (TASK-35021). The view-only pages' 'Planned' row and Inspector owner line are TASK-35016's too.

Plan: `backlog/docs/plan-2026-10-14-settings-plain-language-copy.md`. Findings closed: D4, D-Q7-danger-copy, and D-minor-14-user-guide and A-minor-11-user-guide for the copy this task changes (Settings UX review, 2026-10-14). B11 is excluded: it was never corroborated, and its evidence row moved to the Inspector on #3057.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Settings page, State banner or status row shows a database file name, Python class name, raw enum value, TOML section name or roadmap note, including the Agents banner and both Agents scope notes, Storage, Diagnostics and every domain page
- [ ] #2 Speech & TTS shows connection, catalog and sample status in words (Reachable, Unreachable, Not tested, Not supported)
- [ ] #3 Guardian's and Dreams' read-only rows are labelled in words, their headings say to change them in Advanced Config, and their config keys appear only in one closed 'config key' disclosure under each block
- [ ] #4 Domain pages show no 'Source N' rows and none of the roadmap or developer-instruction rows in their contract tables (such as 'What Settings might add later' and 'Settings role: future defaults…'), and keep their owner and status rows; the view-only pages' 'Planned' row and Inspector owner line are left as TASK-35016 ships them
- [ ] #5 The raw CLI and Terminal card's title states the risk in a calm sentence, with no all-capitals or exclamation marks, and the card keeps its error styling and every disclosure line
- [ ] #6 Canvas limits on Privacy & Security sit in a closed read-only disclosure
- [ ] #7 Typing 'chachanotes' in the '/' filter still finds the conversations and notes database field
- [ ] #8 Console Behavior's agent budget and Goal runs help says what each limit does in plain words, keeps each safety fact (including the timeout below which an MCP tool can run twice), and names no config key
- [ ] #9 Overview's guided-edit hint names categories that exist (Providers & Models, Console Behavior)
- [ ] #10 Every UI string this task changed is quoted in the Settings User Guide as it now renders, and its Goal runs text no longer says 'zero disables admission'; the guide's category map and view-only page count (TASK-35016), the Goal runs section's place under Console Behavior (TASK-35017) and its Tab sentence (TASK-35021) are left as those tasks ship them
- [ ] #11 The plain-copy guard also covers banner scope text, page copy, domain contract rows and the Speech, Guardian and Dreams word maps, and fails when any string above is restored
<!-- AC:END -->
