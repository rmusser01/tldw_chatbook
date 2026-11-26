---
id: TASK-42554
title: 'Importing a character card in Roleplay leaves event-loop gaps of 41 to 111 ms whose cause is not isolated'
status: To Do
assignee: []
created_date: '2026-11-24 12:00'
labels:
  - follow-up
  - roleplay
  - performance
  - event-loop
dependencies: []
references:
  - tldw_chatbook/UI/Screens/personas_screen.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ADR-011 runs for PR #3518 (the Roleplay import-landing PR) measured the loop heartbeat through the card-import phase on both arms and found gaps nobody investigated. The "import" phase max gap was 71.9 ms and 111.4 ms on the base arm and 41.1 ms and 51.4 ms on the head arm (two runs per arm of a heartbeat probe with per-phase max loop gap, 2026-11-12, run on the #3518 branch and on its base, #3441's tip `cec10bbc93`; the probe's output does not record the commit, and the raw JSON is local-only, not in the repo). The phase also mounted 8 widgets and removed 6.

This defect exists on dev today (origin/dev `52351c4e0b`, which includes #3518): the import path `PersonasScreen._import_character_from_path` is unchanged in the respects below, and no open PR changes it (#3518 merged and changed only the landing, not these steps). No existing task covers it: TASK-40659 (lore entries-table fill) and TASK-40660 (slow world-book write), both filed by #3518 and now on dev, are about lore imports, not cards.

What the code does on the loop, from reading dev `52351c4e0b` (not measured per step): `validate_path_simple` plus `source.stat()` run synchronously; when the card carries a portable voice profile (`inspection.portable_profile`), `_import_character_from_path` also awaits `_character_tts_profile_service()` and `observe_portable_profile`, then `_commit_imported_character_tts` (which can ask a reuse confirmation or a collision choice, a dialog), all outside `to_thread`; the file read, the TTS inspection, `import_character_card_with_outcome` and `fetch_character_by_id` run in `asyncio.to_thread`; `_land_import` then runs on the loop (`refresh_character_list`, `_select_character`, which mount and remove the card view and inspector). Two candidate causes, neither tested: the loop-side list reload and card-view swap (the 8 mounted / 6 removed widgets), and GIL contention from the threaded parse and write (see the lesson filed beside this task in `lessons-live-verification.md`). The gaps are small next to the lore-import stalls, so this is an investigation first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The gaps in a card import are attributed to a named step, with a per-step measurement (loop-side callback timing or a thread-time probe), not by time interval alone.
- [ ] #2 If a loop-side step owns a gap over 50 ms for a typical card, it is moved off the loop or made incremental, and a mounted test pins the heartbeat gap across the import phase.
- [ ] #3 If the gaps are GIL contention from the worker threads or a generation-2 collection, the task records that and closes without a code change, noting the measured numbers.
<!-- AC:END -->
