---
id: TASK-3305
title: >-
  Ingest copy & labels: human select values, one destination name, honest supported list, exception-free errors
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - copy
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-09/11/12/13/14/16/18/19 of the 2026-08-07 Media Ingestion review. (1) Selects render raw internal tokens as user-facing values (`pymupdf4llm`, `filtered`, `parakeet-onnx`, `url_level`, `recursive_scraping`) — canvas builds `[(opt, opt)]`. (2) Three names for one destination on one screen: "Import media" / "Add content…" / "Import / Export". (3) The supported-list copy and the start-gate reason omit web/URLs while the surface accepts them. (4) URL preflight surfaces a raw Python exception repr (`<urlopen error [Errno 8]…>`). (5) Recent ingests shows literal backslashes (escape_markup applied to markup=False Statics). (6) "1 done — in queue" for finished jobs. (7) The audio collapsed title is a ~140-char run-on with a dangling empty value (`Local Parakeet model folder: ,`). (8) Grammar batch: "Applies to all Plain text & HTML in this import.", label==placeholder duplication, failed-empty-row repeating its basename, commit line visible while the option-error gate blocks, breakdown "1 web" noun, URL shown as "1 file · 0 B".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Every option select shows a human label (with the internal value still persisted); no raw token appears in the rendered panel
- [ ] #2 The ingest destination carries one name across header, rail button, and rail section
- [ ] #3 Supported-list copy and start-gate reasons name URLs/web pages wherever the surface accepts them
- [ ] #4 URL preflight failures render a plain-language message with no exception repr
- [ ] #5 Escaped filenames render clean in Recent ingests; queue summary says "done", not "in queue", for finished runs; collapsed titles cap at a few salient pairs with no dangling empty values
- [ ] #6 Grammar batch items fixed (scope sentence, placeholder, empty-file row, gate-vs-commit mixed message, breakdown nouns, URL estimate line)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add display-label support to the option schema (label map per field option) consumed by the canvas select builder; keep persisted values stable.
2. Naming decision: adopt "Import" family everywhere (matches header + picker frame) — header "Import media", rail button "Import media…", section unchanged if it lists both import and export.
3. Copy fixes at their sources (`ingest_capabilities` hints, `library_ingest_state` breakdown nouns/summaries, `ingest_preflight` error mapping, canvas title builder cap, guardrail/queue captions).
4. Tests: rendered-label assertions, preflight error mapping, collapsed-title cap, breakdown noun table.
<!-- SECTION:PLAN:END -->
