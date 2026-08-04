---
id: TASK-2313
title: Watchlists copy terminology and empty-state sweep
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: low
---

## Description (the why)

UAT minor findings, one sweep: three stacked empty-state messages on first
run (header + pane + inspector all shouting variants of "nothing yet");
"Latest run status: unavailable" and "State: ready" vocabulary (unavailable
reads as an error; ready-for-what?); "Console follow" jargon duplicated;
"Mixed | Local/Server" cryptic in the screen header; backend Select
duplicated by a "Backend: local" label; "Import OPML" twice on one screen;
"scraped" vs "checked" terminology drift; "Queued" items column meaning
undiscoverable; bare empty tables on Runs/Rules/Notifications vs Overview's
excellent guidance; Inspector's Console block permanently outranking the
selected object's actions and "Type: source" debug-style line.

UAT findings F3-F8, F10, F21, F25, F35 (+F4 vocabulary).

## Acceptance Criteria (the what)

- [ ] One empty-state voice: each region states its own emptiness once,
      without triple redundancy on first run.
- [ ] Status vocabulary reads as states, not faults ("No runs yet" not
      "unavailable"); one term for check/scrape chosen and applied.
- [ ] Duplicate affordances resolved (one Import OPML, backend shown once).
- [ ] Runs/Rules/Notifications empty states carry one line of guidance.
- [ ] Inspector: selected-object block above Console actions; "Type:" line
      reads as prose or is dropped.
- [ ] Every remaining column/control has a discoverable meaning.
