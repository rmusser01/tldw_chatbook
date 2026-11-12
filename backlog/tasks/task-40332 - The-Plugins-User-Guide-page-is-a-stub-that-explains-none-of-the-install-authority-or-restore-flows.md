---
id: TASK-40332
title: "The Plugins User Guide page is a stub that explains none of the install, authority or restore flows"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
- plugins
priority: low
references:
- Docs/User_Guide/plugins.md
- Docs/User_Guide/index.md
- tldw_chatbook/UI/Screens/plugins_screen.py
- Docs/Plugins/operations.md
- scripts/capture_user_guide.py
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** F8 Plugins is a top-level destination with install, authority and restore behaviour, but `Docs/User_Guide/plugins.md:1-26` is marked as a stub: orientation, button names and a 4-row keys table. It does not link `Docs/Plugins/operations.md`, which already covers configure, disable/uninstall, recovery and "After restoring a backup". TASK-32695 owns `Docs/Plugins/*` and never names this page.

**What the screen does** (`plugins_screen.py:52-166`, `Plugins_Modules/{browse_state,details,operations,review}.py`): a Scope line; Installed / Browse / Marketplaces views (keys 1/2/3 switch views inside Plugins; they are not destination keys); Repository... (Git URL or absolute local directory, Git ref default HEAD); Import... (Cursor or Codex); Add source... (Detect, Agent Plugins, Cursor, Claude `marketplace.json`); Preview -> Review install...; a paged list; detail actions (Review preview..., Preview entry, Refresh source, README, Manage...); Authority / History / Candidates / Reset authority; and since TASK-40132 a restore puts plugins back and keeps them off until authority recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 plugins.md follows _template.md and covers each view, the three entry forms through Review install..., the detail actions, the authority and history controls, and the off-until-authority-recovery state after a restore
- [ ] #2 It includes one 235x52 capture made with scripts/capture_user_guide.py and links Docs/Plugins/operations.md
- [ ] #3 The stub marker is gone from the page and from the index row, and the 1/2/3 keys are described as view keys within Plugins
<!-- AC:END -->
