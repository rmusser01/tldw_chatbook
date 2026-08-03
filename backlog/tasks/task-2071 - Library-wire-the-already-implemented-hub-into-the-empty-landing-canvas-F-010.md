---
id: TASK-2071
title: >-
  Library: wire the already-implemented hub into the empty landing canvas
  (F-010)
status: In Progress
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-03 22:02'
labels:
  - ux-review
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Landing canvas is one line of copy in a huge void while _hub_state_summary/_hub_readiness_summary/LIBRARY_EMPTY_NEXT_ACTION_COPY sit uncalled. Evidence: library_screen.py:3237,3255,4223. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty landing shows actionable next-step rows (ingest/search/new note) plus recents/counts from the existing helpers,Dead hub helpers are wired or deleted,Tests cover the empty landing content
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (wires existing helpers into an existing branch; dead-code deletion of never-called helpers). Steps: 1. RED tests (Tests/UI/test_library_shell.py): landing hub shows counts/recents/next-action rows; each of the 3 action rows opens its canvas (parametrized); dead helper names asserted absent from LibraryScreen/module. 2. library_screen.py landing else-branch: counts line via _hub_source_count_value (… while loading), recents line via _source_recent_value (only when any exist), 3 quiet action Buttons (Import media/Search/New note) with row_id/target_kind/target_id attrs + a second @on(.library-hub-action) selector on the existing rail-row handler so they dispatch identically. 3. Delete dead helpers: LIBRARY_EMPTY_NEXT_ACTION_COPY, _hub_state_summary, _hub_readiness_summary, _hub_key_value_row, _hub_readiness_counts, _hub_recent_sources_label, _hub_inventory_readiness_label, _hub_inventory_console_label, _hub_console_status, _hub_inventory_row, _hub_section_rule, _source_recent_label, LIBRARY_HUB_INVENTORY_* constants. Keep live: _hub_table_cell, _hub_source_count_value, _source_recent_value, _source_sample_titles. 4. Run library shell/shell-state/rail tests + parity + ruff.
<!-- SECTION:PLAN:END -->
