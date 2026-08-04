---
id: TASK-2307
title: Reader renders feed HTML as readable text
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: the item reader displays raw HTML markup — "<p>Article URL: <a
href=...>" shown literally. It is correctly escaped and inert (keep that
property), but unreadable as content. The reader region is also ~9 rows at
52-row terminals with no advertised way to expand it (z/Z region controls
exist but nothing on screen says so).

UAT findings F26 (high), F27.

## Acceptance Criteria (the what)

- [ ] Feed HTML content renders as readable text (tags stripped or
      converted; links presented legibly), while remote-derived text remains
      inert against markup/injection — the escaping-terminal rule holds at
      the NEW final render step.
- [ ] The reader advertises how to give itself more room (or provides a
      visible expand affordance).
- [ ] Regression tests cover HTML-to-text rendering and the injection-
      inertness of rendered remote content.
