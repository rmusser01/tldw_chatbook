---
id: TASK-2302
title: New Source form states its destination and lands there
status: Done
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

- [x] The create form shows, before submit, which watchlist (or Unassigned)
      the source will join — honoring the active scope by default and
      letting the user change it.
- [x] After Create, the source is where the form said it would be, and the
      user gets a visible confirmation naming the destination.
- [x] The Type Select carries a visible label.
- [x] The noise-field help text is fully visible at supported sizes, and the
      ignore-selectors block is only presented where it applies (or clearly
      marked as page-scrape-only).
- [x] A regression test covers "source created under an active watchlist
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

## Implementation Notes

The destination was never dropped -- it was never asked for. `CreateSourceRequested`
carried a source record and nothing else, and membership is a separate table
(`watchlist_sources`), so there was no value anywhere in the create path that
the write could have honoured. The fix is a value, a control, a write and a
confirmation, in that order.

### AC#1 -- the control, and where its default comes from

A compact `PruneSafeSelect` with a visible `Static` label, on its own row
between Type/Active and Tags. `Unassigned (no watchlist)` is the FIRST option
and always present: it is the honest name for what the UAT actually got, and
the only possible answer on a profile with no watchlists. Its value is the
string `"unassigned"`, not `None` -- `Select` reserves a `NoSelection`
sentinel of its own and this control is `allow_blank=False`, so "no
watchlist" has to be a real, selectable option.

The default is `_scope_default_destination()`, resolved on the SCREEN (the
pane has no service, like every other pane here) and re-applied every time
the form is opened *interactively*. That gate matters: `_build_detail_pane`
re-opens an already-open form on every workbench rebuild, so an unguarded
reset would discard a destination the user had already chosen. It is the same
`is_mounted` gate `_pending_create_focus` uses, for the same reason.

The chosen destination is draft state (`CreateFormDraftChanged` gains
`destination` and `source_type`), mirrored on the screen and seeded back on
rebuild -- exactly like name/url/tags/selectors. Without that, any region
collapse re-aims an open form.

### AC#2 -- the write, and a confirmation that cannot lie

`_create_source` pops `watchlist_id` out of the payload before it reaches a
backend that has no column for it, then `_file_created_source` writes the
membership row and RETURNS the destination to name. The toast is built from
what that method actually did, not from what the form asked for: no bundle
service, a server backend, or a created row with no local `source_id` all
report `Unassigned`, which is where the source really is. A toast naming a
watchlist the source is not in would be this task's own defect restated as a
lie instead of a silence.

`created["source_id"]` is read, not `created["id"]` -- the normalizer
publishes a namespaced `local:subscription:5` under `id`, and membership rows
key on the raw local id (the distinction `_resume_source` already documents).

### AC#3 -- the Type label

A `Static` beside the Select, using the form's own idiom (the `Active` label
next to the Switch), not a border title: a compact/bordered Select draws its
border on its child `SelectCurrent` (TASK-2300), so a title set on the Select
itself has no border to sit on, and a label row of its own is a row this form
does not have.

### AC#4 -- and a measurement that was simply wrong

Two findings, one root. The ignore-selectors block is now rendered only for
url-family types, gated in `compose()` (the single owner of what this form
contains) via a `recompose=True` type reactive -- mounting it from a watcher
would give a conditionally-composed control a second owner, a bug class this
codebase has paid for repeatedly.

That gate was unreachable as written: the create Select offered only FEED
types, so the form could not produce a single source `ignore_selectors` can
affect, which is why the field read as decorative prefill. `Web page`
(`url` -- the value `_local_type_for_source_type` accepts verbatim and the
normalizer publishes back) is added so the capability TASK-1362 built is
reachable rather than deleted.

The truncation was a stale number. TASK-1362 recorded the field as 91 columns
at 160x42 and sized its label and help copy to that; measured through the
production stylesheet in the full shell it is **53** columns at 160x42 and
**78** at 235x52, and Textual truncates a border label silently at width - 4.
Both strings now fit the narrowest supported size, the displaced syntax
detail moves to the tooltip (no width budget), and the test reads the mounted
field's own width rather than trusting any number in a comment.

The submit path reads the selectors through the DOM and only when the field
is present. Reading the draft would have been the subtle version of the same
bug: `_clear_create_draft` deliberately keeps that draft prefilled for the
next form, so every RSS source ever created here would have been stored with
page selectors it can never use.

### Verification

* New file `Tests/UI/test_watchlists_create_form_destination.py` (15 tests).
  Where a source LANDED is read from the bundle service's own membership
  query -- never from a toast, a rail count or a table row, all of which are
  downstream of the write and would stay green with it deleted.
* Mutation-verified: **10** mutations (scope default, the membership write,
  the payload destination, the Type label, the ignore gate, the DOM-vs-draft
  selector read, the open-time destination reset, the rebuild seeding, the
  help copy, and an Unassigned-falls-back-to-scope regression), each applied
  and reverted individually -> RED -> restored byte-exact (md5). Zero
  survivors.
* Gates: create-form + destination + vocabulary + sources-pane + shell +
  content-pane + rail-counts + select-overlays + frequency + row-click
  **237 passed** (plus the shell suite re-run at **71 passed** after its
  noise-field test was taught the gate); inspector + collections-screen +
  visual-parity **185 passed**; poisoned-order pass (content pane + the
  create-form e2e, one invocation) **50 passed**; `--collect-only Tests/UI
  Tests/Watchlists` **8782 collected**, no errors.

### Live verification (235x52, fresh profile)

```
scope = Reading, create form open
   Type ▊  RSS                                    ▼  ▎ Active ▊  ▎
   Watchlist Reading                                          ▼
   (no ignore-selectors block)
after Create ("HN Front Page", https://hnrss.org/frontpage)
   header  Local Watchlists snapshot: Reading (1 source)
   table   HN Front Page  rss  active  -  Yes
   toast   Source created in "Reading".
second create, destination changed in the form
   Watchlist Unassigned (no watchlist)
   toast   Source created in Unassigned.
type = Web page
   ▊▔ Ignore elements (CSS selectors, one per line) ▔▔▔ ... ▔▎
   ▊ [class*="cookie-consent"], [class*="cookie-banner"], ...
   ▊▁▁▁ ... ▁ Silence noise; change_threshold limits volume. ▁▎
```

Read back out of the live profile database:

```
1|HN Front Page|rss|Reading|no-selectors
2|Loose Feed   |rss|(unassigned)|no-selectors     <- before the assign below
3|Docs Page    |url|Reading|has-selectors
```

### Files

* `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` -- the control, the
  gate, the draft, the payload.
* `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` --
  `_create_form_watchlist_choices`, `_scope_default_destination`,
  `_file_created_source`, seeding and mirrors.
* `tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated bundle) --
  the destination row and the shared field label.
* Tests: new `Tests/UI/test_watchlists_create_form_destination.py`;
  `Tests/UI/test_watchlists_source_create_form.py` (both form shapes now
  parametrized), `Tests/Watchlists/test_watchlists_sources_pane.py` and
  `Tests/UI/test_watchlists_destination_shell.py` (noise tests choose the
  page type the field lives on).

### Review wave (whole-branch adversarial review: 3 Important, 5 Minor)

Two of the review's five mutations SURVIVED, both in the hand-reconstructed
code and both in one corner -- *the set of watchlists changes while a create
form is alive*. Tests now enter it: deleting the destination watchlist under
an open form (without `_resolved_destination`'s fallback the next recompose
raises `InvalidSelectValueError` out of `compose()`, taking the form AND the
table down), and creating one mid-session (without the `watchlist_choices`
push it never becomes selectable). Both mutation-red now.

**The width measurement in this task's first pass was wrong, and it was
wrong in shipped code.** The geometry tests ran through a harness that loads
no stylesheet: 53/78 columns there, **93/168** under the production
stylesheet. So TASK-1362's 91-column figure was RIGHT, and the comment
declaring it wrong is corrected. The shorter label and help copy are kept on
their own merits -- half the columns, cannot truncate at any supported size,
and the displaced syntax detail sits in a tooltip that has no width budget --
not on the width argument.

**F11/F12 is therefore unresolved, and is recorded as such rather than
claimed.** At 235x52 production CSS gives a 164-column budget for the old
83-character help string; it could not have truncated in that layout. What
the UAT saw is not explained by this fix -- a narrower terminal, a different
region layout, or an older build. The shipped copy is safe either way.

The whole test file moved to the production-CSS harness (not just the
geometry half, so no future test in it can pick the wrong one), and the label
assertions now check what the CSS comment actually claims -- `width: auto`
keeping the label narrower than the control it names, which the old harness
could not express because there the label WAS the row.

Also from the review: the degraded-destination toast is `warning` with "The
watchlist you chose could not be used." appended (M3), and
`test_the_destination_offers_nothing_when_membership_cannot_be_written` now
opens the form under the server backend instead of asserting on two screen
helpers.

Verification: **11** review-wave mutations, all RED, zero survivors, restores
md5-verified; gate wave 1 **178 passed**, wave 2 **259 passed**,
poisoned-order **50 passed**, `--collect-only` **8788 collected**. Live
re-check on a fresh profile: delete-under-open-form degrades to
`Unassigned (no watchlist)` with the draft and the table intact.
