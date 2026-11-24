---
id: TASK-42554
title: "Importing a character card in Roleplay leaves event-loop gaps of 41 to 111 ms whose cause is not isolated"
status: To Do
assignee: []
created_date: 2026-11-24 12:00
labels:
- follow-up
- roleplay
- performance
- event-loop
priority: low
references:
- tldw_chatbook/UI/Screens/personas_screen.py
- .worktrees/.b2b-evidence/followups/fu-landing-evidence/adr011/
dependencies: []
---

## Description

The ADR-011 runs for PR #3518 (the Roleplay import-landing PR) measured the loop heartbeat through the card-import phase on both arms and found gaps nobody investigated. The "import" phase max gap was 71.9 ms and 111.4 ms on the base arm and 41.1 ms and 51.4 ms on the head arm (`adr011-base-1/2.json`, `adr011-head-1/2.json`, key `phases` / `import`, 2026-11-12, measured on the #3518 stack and its base, #3441's tip `cec10bbc93`; the JSON does not record the commit). The phase also mounted 8 widgets and removed 6.

This defect exists on dev today (origin/dev `5e73e8a2bc`): the import path `PersonasScreen._import_character_from_path` is unchanged in the respects below, and no open PR changes it (PR #3518 is open and changes only the landing, not these steps). No existing task covers it: TASK-40659 (lore entries-table fill) and TASK-40660 (slow world-book write), both filed in PR #3518, are about lore imports, not cards.

What the code does on the loop, from reading dev `5e73e8a2bc` (not measured per step): `validate_path_simple` plus `source.stat()` run synchronously; the file read, the TTS inspection, `import_character_card_with_outcome` and `fetch_character_by_id` run in `asyncio.to_thread`; `_land_import` then runs on the loop (`refresh_character_list`, `_select_character`, which mount and remove the card view and inspector). Two candidate causes, neither tested: the loop-side list reload and card-view swap (the 8 mounted / 6 removed widgets), and GIL contention from the threaded parse and write (see the lesson filed beside this task in `lessons-live-verification.md`). The gaps are small next to the lore-import stalls, so this is an investigation first.

## Acceptance Criteria

- [ ] The gaps in a card import are attributed to a named step, with a per-step measurement (loop-side callback timing or a thread-time probe), not by time interval alone.
- [ ] If a loop-side step owns a gap over 50 ms for a typical card, it is moved off the loop or made incremental, and a mounted test pins the heartbeat gap across the import phase.
- [ ] If the gaps are GIL contention from the worker threads or a generation-2 collection, the task records that and closes without a code change, noting the measured numbers.
