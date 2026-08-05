---
id: TASK-2302
title: New Source form states its destination and lands there
status: In Progress
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

## Implementation Plan (the how)

1. Add a `Watchlist` destination `PruneSafeSelect` to the create form
   (`sources_pane.py`), with a visible `Static` label, an explicit
   `Unassigned (no watchlist)` entry, and the watchlist list seeded from
   the screen. Default = the active tree scope's watchlist, re-applied
   every time the form is opened interactively.
2. Carry the choice as draft state so it survives a workbench rebuild, the
   same way name/url/tags already do (`CreateFormDraftChanged` gains the
   destination and the source type), and seed it in `_build_detail_pane`.
3. Put the destination into `CreateSourceRequested.payload`; in
   `_create_source`, write the membership row through
   `WatchlistBundleService.add_source` after the source is created, then
   confirm with a toast that names the destination.
4. Give the Type Select a visible label, using the form's own existing
   label idiom (the `Active` Static beside the Switch).
5. Render the ignore-selectors block only for types it can affect
   (url/url_list/sitemap/site) and add the missing `Web page` type so that
   gate is reachable at all; verify the field's label and help copy fit the
   field's real painted width at both supported sizes.
6. Tests: the destination is shown before submit, the membership row is
   actually written (asserted against the bundle service, not the toast),
   Unassigned really means unassigned, and the noise copy fits.
