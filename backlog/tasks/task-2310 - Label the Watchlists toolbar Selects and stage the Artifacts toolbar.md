---
id: TASK-2310
title: Label the Watchlists toolbar Selects and stage the Artifacts toolbar
status: Done
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: filter Selects across the screen display bare values with no hint of
what they filter — Sources shows "All ▼ / All statuses ▼ / All ▼" (two
unlabeled), Artifacts shows "Auto + featured ▼ / App default ▼ / Off ▼"
(off for WHAT?), and the New Rule form's severity Select and Threshold field
are unlabeled/unexplained. The Artifacts toolbar also shows 12 controls
including "Stop Serving" before any briefing exists.

UAT findings F9, F37, F38.

## Acceptance Criteria (the what)

- [x] Every Select on the Watchlists screen carries a visible label naming
      what it controls (border-title style is fine).
- [x] The Rule form explains Threshold (unit/meaning) and labels severity.
- [x] The Artifacts empty state foregrounds Generate; serve/export/keep
      controls appear only once they can act on something (or are visibly
      disabled with a reason).

## Implementation Plan (the how)

1. Survey every Select on the screen (Sources/Items toolbar filters,
   Artifacts mode/preset/cadence pickers, Rules condition/severity) and
   the existing labeled-Select idiom (task-2302's `.sources-create-
   field-label`, a sibling `Static` -- a compact/bordered Select's border
   is stripped, TASK-2300, so `border_title` has nowhere to sit).
2. Add a generic, one-row `.watchlists-inline-select-label` CSS class and
   a `Static` before each unlabeled Select, reusing the established idiom
   where it fits the row's height/width budget.
3. Explain Threshold dynamically (its unit depends on the selected
   Condition) and label Condition/Severity in the Rule form.
4. Gate the Artifacts toolbar's "Stop Serving" button on there being
   something for it to act on; verify Export/Keep/Serve already satisfy
   "visibly disabled with a reason" (they do -- no change needed there).
5. Measure against the production stylesheet at every already-tested
   size; correct course where a persistent label does not fit.
6. Tests + mutation verification; live verification in tmux.

## Implementation Notes

**Items, Artifacts (mode/preset/cadence), Rules (condition/severity):**
each unlabeled Select gained a sibling `Static("...", classes=
"watchlists-inline-select-label")` immediately before it, a new generic
CSS class (`_watchlists.tcss`) sized for a one-row `.destination-filter-
strip` -- no new row, no extra height budget. All fit comfortably at
every already-tested size.

**Sources' toolbar (Type/Status/Active) was a genuine width conflict, not
just a copy fix.** The persistent-label approach was tried first and
measured against the production stylesheet: at this toolbar's own tested
floor, 160x42, the row already spends every column it has -- the search
box's placeholder ("Search sources...") only reaches full width today
because the three filter Selects claim zero spare columns beyond their
own fixed widths (sized to their longest option, per TASK-995's own
comment). Adding even one label pushed `#sources-filter-toggle` off the
pane's right edge (measured: x=118..134 against a 93-column pane).
Lowering the search box's `min-width` to compensate broke a DIFFERENT,
pre-existing pinned contract instead: `test_watchlists_sources_toolbar_
controls_are_actually_visible` requires the FULL "Search sources..."
placeholder to reach the screen, not merely a non-overflowing region.
Sources' three filters carry a `tooltip` instead (the one mechanism that
costs no column) -- consistent with how Artifacts' own picker Selects
already documented themselves before this task, and the shipped decision
recorded in `sources_pane.py`'s own comment so the next reader does not
re-attempt the persistent-label version and re-discover the same
conflict.

**Threshold's meaning is condition-dependent, not a single unit.** Traced
`LocalWatchlistsService._evaluate_condition`: `error_rate_above` reads it
as a 0-1 FRACTION, `items_below`/`items_above` read it as an item COUNT,
`no_items`/`run_failed` never read it at all -- so a user typing "50"
for "50% error rate" would silently mean "5000%", i.e. never fires. The
field's placeholder and a help line under it now track the selected
Condition live (`RulesPane.on_select_changed`, in place -- the form may
already have a typed Name, so a recompose was not an option).

**Artifacts "Stop Serving"**: unlike every sibling button in that
toolbar (Export/Keep/Serve, which stay visible-but-disabled specifically
so a first-time user can discover them before they apply), Stop has no
useful disabled-but-visible state -- it can only ever act on a server
THIS pane just started. Rendered only while `feed_server_running`, closing
the UAT's literal complaint ("Stop Serving before any briefing exists",
one of 12 controls). Export/Keep/Export Feed/Serve Feed already satisfied
"visibly disabled with a reason" and needed no change; Generate is
already first, `variant="primary"`.

### Verification

* New/extended tests: `Tests/Watchlists/test_watchlists_sources_pane.py`
  (tooltip coverage), `test_watchlists_items_pane.py` (label), `test_
  watchlists_artifacts_pane.py` (picker labels + Stop Serving absent/
  present), `test_watchlists_rules_pane.py` (Condition/Severity labels +
  the live Threshold-guidance update, including a same-session "value
  survives every condition change" proof that the update is in-place).
* Mutation-verified: 6 mutations (each new label, the Stop Serving gate,
  the Threshold live-update handler), each reverted individually -> RED
  -> restored byte-exact (md5).
* Gates: `Tests/Watchlists/` + `Tests/UI/test_destination_headers.py` +
  `Tests/UI/test_destination_shells.py` + `Tests/UI/test_watchlists_
  select_option_overlays.py` + `Tests/UI/test_watchlists_source_create_
  form.py` **586 passed, 1 skipped**, plus one pre-existing failure
  (`test_stts_screen_composes_destination_header_in_the_lab_frame`,
  unrelated -- zero diff against `origin/dev` in that screen or its test).
  `Tests/Watchlists/test_watchlists_artifacts_pane.py` alone: **130
  passed** (a prior run's single flake in that same run,
  `test_citations_do_not_shrink_the_briefings_table_below_its_pinned_
  minimum`, reproduced RED in the full-file run and GREEN standalone --
  order-dependent, pre-existing, confirmed unrelated).

### Follow-up (UAT batch-5 whole-branch review, finding m1)

AC#1's literal text ("every Select on the Watchlists screen") was broader
than this task's own scoped survey (Implementation Plan step 1: "Sources/
Items toolbar filters, Artifacts mode/preset/cadence pickers, Rules
condition/severity") actually reached. Two pre-existing, tooltip-only
Selects on modals launched FROM the Artifacts pane —
`briefing_preset_modal.py`'s per-speaker Character/Voice `PruneSafeSelect`s
and `kept_briefings_modal.py`'s `#kbm-preset-select` (structurally
identical to `#artifacts-preset-select`, which DID get a label in this
task) — were missed, since neither file is touched by this task's original
diff. Labeled both rather than narrowing the AC, per the review's stated
preference: `briefing_preset_modal.py` gained a column-header row (`Name /
Role prompt / Character / Voice`, reusing the existing per-row width
classes for alignment) above the repeated speaker rows; `kept_briefings_
modal.py`'s preset picker gained the same `watchlists-inline-select-label`
"Preset" `Static` `#artifacts-preset-select` already has. Both
mutation-verified (Edit-tool revert -> RED -> restored byte-exact, md5).

### Files

* `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`,
  `items_pane.py`, `artifacts_pane.py`, `rules_pane.py`.
* `tldw_chatbook/css/features/_watchlists.tcss` (+ regenerated bundle).
* `Tests/Watchlists/test_watchlists_sources_pane.py`, `test_watchlists_
  items_pane.py`, `test_watchlists_artifacts_pane.py`, `test_watchlists_
  rules_pane.py`.
* Follow-up (m1): `tldw_chatbook/UI/Watchlists_Modules/briefing_preset_
  modal.py`, `kept_briefings_modal.py`; `Tests/Watchlists/test_watchlists_
  briefing_presets_ui.py`, `test_kept_briefings_modal.py`.
