---
id: TASK-34394
title: Debounce and batch legacy UI hot paths
status: To Do
created_date: 2026-10-11 21:44
labels:
- performance
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy surfaces still rebuild everything per keystroke or per event where the repo already has the cure: config settings search builds its DOM index twice per keystroke with no debounce (UI/Widgets/config_search_widget.py + Tools_Settings_Window.on_config_search_changed); CodeRepo tree filter re-walks every node per keystroke; MCP Hub tools/audit filters clear and rebuild their DataTable per keystroke with per-row schema parses; CCP conversation load mounts ~200 message widgets one mount() at a time; chat message compose runs FileExtractor._find_filename_hint with O(k*bytes) prefix slicing; media_details_widget does sync SQLite + full-text reformat on the event loop per selection; STTS audiobook file selection reads whole files synchronously.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings config search debounces and builds its index once per pane activation,CodeRepo tree filter debounces and precomputes lowercased paths,MCP tools/audit filters debounce and do not re-parse tool JSON schemas per keystroke,CCP message display batches mounts in a single call,Filename-hint search scans only lines near each code block instead of the whole prefix,media_details chunking-config fetch and STTS audiobook file read run via asyncio.to_thread
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
