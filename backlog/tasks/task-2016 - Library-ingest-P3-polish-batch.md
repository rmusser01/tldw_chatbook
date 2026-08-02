---
id: TASK-2016
title: >-
  Library ingest P3 polish batch
status: To Do
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: low
dependencies: []
---

## Description (the why)

P3 polish and needs-reproduction findings from the 2026-08-02 Library
ingest UAT (critique snapshot 2026-08-02T21-04-04Z). None block tasks;
grouped so they are not lost. Items marked (repro) were observed once on a
contaminated instance or depend on environment — reproduce before fixing.

## Acceptance Criteria (the what)

- [ ] Done rows state "done" once and show the file basename (full path
      available in details), not the absolute path inline.
- [ ] "Expand all / Collapse all" render only when more than one options
      panel exists.
- [ ] The generic panel's scope line no longer claims "Applies to all
      Plain text / documents / HTML in this import." when zero such files
      are staged (reword for the global-options case).
- [ ] Intro lines disappear once a path is typed (state already says so;
      the DOM-surgery typing path never removes them).
- [ ] The file picker opens at the last-used directory and hints which
      extensions are ingestible.
- [ ] Rail counts no longer flash "(0)" before the lazy count arrives
      (defer rendering the number until known).
- [ ] (repro) The ingest error-details modal renders fully inside the app
      frame with a visible close affordance.
- [ ] (repro) First submit never smears dependency warnings / loguru
      DEBUG over the TUI — route warnings and stderr away from the TTY
      while the TUI owns the terminal.
- [ ] `#library-search-input` gets an `Input.Changed` handler so typed but
      unsubmitted rail-search text persists/clears the way the user left
      it instead of resurrecting from `_library_rag_query` on recompose.
- [ ] `[first_run] setup_started/setup_completed` are only written when
      the user actually starts/completes (or explicitly skips) the wizard,
      not at app open.
