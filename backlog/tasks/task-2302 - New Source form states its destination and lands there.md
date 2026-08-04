---
id: TASK-2302
title: New Source form states its destination and lands there
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

UAT: the New Source form has no watchlist-destination field or indicator.
Created while a watchlist was the active scope, the source silently landed in
Unassigned — directly contradicting the first-run guidance ("press New Source
to add a feed to it"). The user cannot predict where Create will put the
feed, and gets no notice afterwards. Form polish issues found in the same
pass: the Type Select has no visible label (bare "RSS ▼"), the noise-field
help subtitle truncates mid-sentence at 235x52, and the CSS ignore-selectors
block is prefilled and prominent for RSS sources where element selectors do
not apply.

UAT findings F13 (high), F17, F11, F12, F14.

## Acceptance Criteria (the what)

- [ ] The create form shows, before submit, which watchlist (or Unassigned)
      the source will join — honoring the active scope by default and
      letting the user change it.
- [ ] After Create, the source is where the form said it would be, and the
      user gets a visible confirmation naming the destination.
- [ ] The Type Select carries a visible label.
- [ ] The noise-field help text is fully visible at supported sizes, and the
      ignore-selectors block is only presented where it applies (or clearly
      marked as page-scrape-only).
- [ ] A regression test covers "source created under an active watchlist
      scope joins that watchlist".
