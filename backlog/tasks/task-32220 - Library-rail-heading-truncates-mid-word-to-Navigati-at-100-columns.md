---
id: TASK-32220
title: Library rail heading truncates mid-word to 'Navigati' at 100 columns
status: Done
assignee: []
created_date: '2026-09-10 14:54'
updated_date: '2026-09-10 19:04'
labels:
  - library
  - rail
  - copy
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail heading is cut mid-word in the 22-column compact rail. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 17.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The heading is shortened by rule (e.g. 'Nav') or ellipsised, never cut mid-word
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing painted test of the heading at 100 columns.
2. Add a text-overflow rule for the heading label.
3. Rebuild the bundle; live-verify at 100 columns.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
At 100 columns the rail heading label is 8 cells wide and painted "Navigati". `#library-rail-heading-label { text-overflow: ellipsis; }` gives it "Navigat…"; wherever the full word fits (235 and 60 columns) it is unchanged. Pinned by a painted-frame test that reads the heading through the production bundle. Live: "Navigat…" at 100x30, "Navigation" at 235x52 and 60x24.

Files: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit9_rail.py
<!-- SECTION:NOTES:END -->
