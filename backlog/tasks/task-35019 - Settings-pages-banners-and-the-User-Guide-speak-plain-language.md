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
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings pages still show implementation vocabulary as user copy. The Agents banner reads 'agent_runs.db (SQLite) — immediate CRUD, no draft'. Speech prints the raw status 'not_tested'. Storage names the 'ChaChaNotes DB' and 'active handles'. Guardian and Dreams use raw TOML keys as row labels. Diagnostics and the domain pages carry roadmap and developer notes ('not available yet', 'Planned', 'Source 2: workflow Console handoff payloads', 'future defaults may cover…'). The raw CLI card shouts 'DANGER!!!' against the product's calm, confident voice. Paths are folded onto two lines even in the wide detail pane. The Settings User Guide opens with a misplaced Goal runs section that names a key (F9) Settings does not use, and its category map has the wrong counts and leaves out six categories. This task rewrites that copy in words and keeps each key in a closed disclosure. The danger warning keeps its full force through its card styling and disclosure lines. The task also brings the User Guide back in line with the app.

Plan: `backlog/docs/plan-2026-10-14-settings-plain-language-copy.md`. Findings closed: D4, B11, D-Q7-danger-copy, D-minor-14-user-guide, A-minor-11-user-guide (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Settings page, State banner or status row shows a database file name, Python class name, raw enum value, TOML section name or roadmap note, including the Agents banner and both Agents scope notes, Storage, Diagnostics and every Domain Defaults page
- [ ] #2 Speech & TTS shows connection, catalog and sample status in words (Reachable, Unreachable, Not tested, Not supported)
- [ ] #3 Guardian's and Dreams' read-only rows are labelled in words, their headings say to change them in Advanced Config, and their config keys appear only in one closed 'config key' disclosure under each block
- [ ] #4 Domain Defaults pages show no 'Planned', 'Source N' or future-roadmap rows, and keep their owner and status rows
- [ ] #5 The raw CLI and Terminal card's title states the risk in a calm sentence, with no all-capitals or exclamation marks, and the card keeps its error styling and every disclosure line
- [ ] #6 Canvas limits on Privacy & Security sit in a closed read-only disclosure
- [ ] #7 Typing 'chachanotes' in the '/' filter still finds the conversations and notes database field
- [ ] #8 At 211x44 and 235x52, Storage's and Diagnostics' config path rows show the path on one line when the detail pane is wide enough to hold it
- [ ] #9 Console Behavior's agent budget and Goal runs help says what each limit does in plain words, keeps each safety fact (including the timeout below which an MCP tool can run twice), and names no config key
- [ ] #10 Overview's guided-edit hint names categories that exist (Providers & Models, Console Behavior)
- [ ] #11 The Settings User Guide covers Goal runs inside the Console Behavior section and says it is reached with F4. Its category map lists every rail category with correct group counts, and it lists the seven read-only pages without a roadmap column. Every UI string this task changed is quoted as it now renders
- [ ] #12 The plain-copy guard also covers banner scope text, page copy, domain contract rows and the Speech, Guardian and Dreams word maps, and fails when any string above is restored
<!-- AC:END -->
