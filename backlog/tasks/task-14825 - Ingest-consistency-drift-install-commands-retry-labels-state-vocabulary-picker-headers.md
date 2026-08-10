---
id: TASK-14825
title: >-
  Ingest consistency drift: install commands, retry labels, state vocabulary, picker headers
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
labels:
  - library
  - ingest
  - copy
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique — consistency defects accumulated across the two shipped arcs, each captured live. Individually small; together they are why the Consistency heuristic dropped from 3 to 2.

1. **Two install-command forms for one dependency on one screen**: the preflight warning says `pip install -e ".[ebook]"` while the queue row for the same missing dep says `Install with: pip install tldw_chatbook[ebook]`.
2. **Three retry labels coexist**: the per-row `Retry`, the batch `Retry this batch`, and the footer's `r retry last batch`.
3. **State vocabulary drift in the queue**: `2 running` and `2 parsing` describe the same in-flight items on adjacent lines, and item rows use a third form (`● parsing`).
4. **The supported list is written two ways on one screen**: the intro ends `…plain text files, web pages.` while the unsupported-file error ends `…plain text files, web pages (by URL).`
5. **Picker column headers misaligned with their own columns** (measured by column index): the `Size` header occupies cols 185–188 while values right-align to 186; the `Modified` header starts at col 201 while its dates start at 191 — the header sits 10 columns right of its column's left edge, visually labelling the HH:MM half. The header row also sits outside the bordered list box. This is inside the header row task-3304 added.
6. **Mid-sentence detail fragment**: `✗ failed · empty_file.txt · is empty; there was nothing to ingest.` — the basename-echo strip removes the subject, leaving a sentence starting with "is".
7. **Collapsed titles advertise values of disabled fields**: `Images — Extract text (OCR): on` while the control itself reads `— needs OCR backend installed`; `build_type_group_title` filters on emptiness and default-ness but never consults `field_disabled_state`.
8. Raw Python exceptions reach row lines (`Failed to ingest pdf file: 'NoneType' object has no attribute 'FileDataError'`), and nested tool prefixes stack four deep before the actual cause.
9. The consent copy says "Press **Start** again" even when the user armed it with Enter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 One install-command form is used wherever a missing dependency is reported
- [ ] #2 Retry affordances share one vocabulary across row, batch and footer
- [ ] #3 The queue describes an in-flight job with one word across tally, batch header and row
- [ ] #4 The supported-format list has one wording, from one source
- [ ] #5 Picker column headers align with the values they label
- [ ] #6 A failure detail reads as a complete sentence regardless of basename stripping
- [ ] #7 Collapsed panel titles do not advertise values of fields that are disabled
- [ ] #8 A raw exception repr never reaches a queue row line; the user-facing cause comes first
<!-- AC:END -->
