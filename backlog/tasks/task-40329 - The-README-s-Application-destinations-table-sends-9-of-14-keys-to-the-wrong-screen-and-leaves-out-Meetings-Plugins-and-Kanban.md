---
id: TASK-40329
title: "The README's Application destinations table sends 9 of 14 keys to the wrong screen and leaves out Meetings, Plugins and Kanban"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- docs
- navigation
priority: medium
references:
- README.md
- tldw_chatbook/UI/Navigation/shell_destinations.py
- tldw_chatbook/UI/Navigation/screen_registry.py
- Tests/Docs/test_user_guide_contract.py
dependencies:
- TASK-38969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (docs stream, TASK-38969). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The README is the first page people read. `README.md:449-462` (from d01daac8b4, 2026-08-30) no longer matches the destination map (`shell_destinations.py:210-226`, compatibility keys at `:231` Ctrl+6 and `:235` Ctrl+9):
- Ctrl+4 is Roleplay (README: Artifacts, retired by ADR-172; Ctrl+6 is kept as a compatibility key that lands in Library, `screen_registry.py:253`); Ctrl+5 is Watchlists (README: Roleplay); Ctrl+6 is the Artifacts compatibility key (README: Watchlists).
- Ctrl+9 is a compatibility deep link to Settings > Tools & MCP (ADR-240, `screen_registry.py:326`).
- Research is F5 (README: F10); Lab, Logs and Settings are F2, F3 and F4 (README: F7, F8, F9). F7 Meetings, F8 Plugins and F9 Kanban are missing.
- `README.md:655` says "Open **Logs** with **F8**"; Logs is F3 and F8 is Plugins.

TASK-38969's F-key drift test scans only the User Guide (`Tests/Docs/test_user_guide_contract.py:90`, `:120`), so it cannot catch the README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The README table lists every destination shortcut with its destination, plus the Ctrl+6 and Ctrl+9 compatibility keys and where they land
- [ ] #2 Every other hotkey mention in the README matches the map, including the Logs line
- [ ] #3 A Docs test compares the README table with the destination map and is red on the current README
<!-- AC:END -->
