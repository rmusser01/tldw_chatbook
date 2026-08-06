---
id: TASK-2313
title: Watchlists copy terminology and empty-state sweep
status: In Progress
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

- [x] One empty-state voice: each region states its own emptiness once,
      without triple redundancy on first run.
- [x] Status vocabulary reads as states, not faults ("No runs yet" not
      "unavailable"); one term for check/scrape chosen and applied.
- [x] Duplicate affordances resolved (one Import OPML, backend shown once).
- [x] Runs/Rules/Notifications empty states carry one line of guidance.
- [x] Inspector: selected-object block above Console actions; "Type:" line
      reads as prose or is dropped.
- [x] Every remaining column/control has a discoverable meaning.

## Implementation Plan (the how)

1. Sequence after task-2312 (layout) so this copy sweep lands on the final
   tab-strip/header layout rather than the one being replaced underneath it.
2. Screen header: drop the "Mixed | Local/Server" segment; keep the two-part
   title that already reads as one sentence.
3. Backend label: `_backend_label_text()` returns `None` off the
   local-only sections instead of an always-on trailing "Backend: local"
   Static, so the Select (with its own new "Backend" prefix label) is the
   single affordance, not a duplicate of it.
4. Overview/Inspector "unavailable" vocabulary: trace `latest_run_status`
   back to its source (`WatchlistsBackendController`) rather than papering
   over it at the two render sites — a literal string "unavailable" as a
   real computed value defeats any falsy/`None` check downstream. Add
   `_watchlists_state_summary_text()` / `_latest_run_status_text()` helpers
   on the screen and a matching `None`-vs-missing-vs-0 fix in
   `OverviewPane._card_value`.
5. Sources "Last scraped" -> "Last checked" column rename, matching the
   "checked" vocabulary already used by Sources' own "Check now" button and
   task-2308/2311's timestamp copy.
6. Empty-state guidance: give Runs/Rules/Notifications one line each (the
   pattern Overview already had), naming the actual control that unblocks
   the user rather than a bare empty table.
7. Rules create form: add Condition/Severity labels and a live threshold
   help line driven by a `_threshold_guidance()` classmethod, keyed off the
   selected condition type, updated via `on_select_changed`.
8. Items: label the Status filter Select and add a one-line legend
   explaining the Queued glyph.
9. Artifacts: label Mode/Preset/Cadence, show the default briefing provider
   in the scope note when generation is available, and only yield "Stop
   Serving" when a feed server is actually running (was always yielded,
   `disabled=` toggled — a disabled control that changes its own label
   depending on state it can't currently have is worse than not showing it).
10. Inspector: replace the bare `f"Type: {kind}"` line with a
    `_entity_kind_sentence()` sentence ("This is a source."); reorder
    `_build_inspector_pane` so the selected-object block comes before
    "Console actions" (was appended last); shorten the first-run hint to
    name one action instead of duplicating Overview's two-step walkthrough;
    rename the disabled Console-follow button from "Console follow
    unavailable" to "Follow in Console".
11. Rename `briefing_service._default_provider()` to the public
    `default_briefing_provider()` so the screen can reuse it for the
    Artifacts scope-note provider display without reaching into a private
    function.
12. `.watchlists-empty-state-hint` renamed to the more general
    `.watchlists-hint-line` (shared by the three new empty-state lines and
    the rules threshold help); rebuild the CSS bundle.
13. Add a discriminating test per behavioural change; run the full
    Watchlists/UI targeted sweep plus a poisoned-order e2e pass; live-verify
    every change in a fresh tmux session, since this task is almost
    entirely about what the user sees on screen.

## Implementation Notes

Landed after task-2312 as planned, so the copy sweep sits on the final
tab-strip/header layout. Touches `overview_pane.py`, `items_pane.py`,
`runs_pane.py`, `rules_pane.py`, `notifications_pane.py`, `inspector_pane.py`,
`sources_pane.py` (from task-2310, column rename only), `watchlists_backend_
controller.py`, `watchlists_collections_screen.py`, `briefing_service.py`,
`briefing_cast.py`, and `_watchlists.tcss` (+ rebuilt bundle).

The most significant catch was live, not test-suite: after fixing
`_latest_run_status_text()` to treat a falsy status as "no runs yet", the
Inspector still showed "Latest run status: unavailable" in a fresh tmux
session. `WatchlistsBackendController.get_overview_data` was setting
`latest_run_status = "unavailable"` as an actual computed **string value**
in two places (the degraded-state dict and the real aggregation default),
not a missing key — a non-empty string is truthy, so the screen-level
falsy check never saw it. Fixed at the data-layer source (`None` instead of
the literal string) plus a matching fix in `OverviewPane._card_value` to
treat a present `None` the same as a missing key (falls back to "-")
without misreading a legitimate `0` the same way. Neither the automated
suite nor the first live pass caught this — only a fresh-process live
re-verification did, because Python module state doesn't hot-reload and an
earlier live check had run against stale code.

Other fixes worth flagging: `sources_pane.py`'s toolbar Selects use
`tooltip=` rather than persistent `Static` labels — persistent labels broke
`test_watchlists_sources_toolbar_controls_are_actually_visible` at 160x42
because the row's search-input slack was already fully consumed by the
placeholder-visibility contract; documented at the call site so a future
attempt doesn't redo the same failed approach. A duplicate-comment mutation
mistake in `sources_pane.py` was caught via `git diff` inspection during
mutation-restore, not just MD5 trust — restores are now always diff-checked
in this batch, not just hash-checked. A case-sensitive grep for the old
Inspector hint text missed a lower-cased literal assertion in
`test_watchlists_overview_loading_state.py`; found via the broader targeted
sweep rather than the narrow grep and fixed.

Live-verified in a fresh tmux session (235x52, scratch profile) across all
seven sections: tab strip position identical on every section click; header
reads as the two-part sentence with no "Mixed | Local/Server" segment;
single "Backend" label; Sources shows "Last checked"; Items shows a "Status"
label and the Queued legend; Runs/Rules/Notifications each show their one-
line empty-state guidance; Rules' New Rule form labels Condition/Severity
and updates the threshold help line live when the condition changes ("Not
used for this condition..." -> "Item count for the run (whole number)." on
switching to "Items above"); Artifacts labels Mode/Preset/Cadence and (once
a watchlist exists) shows "...on request Generate will use OpenAI." in the
scope note, with "Serve Feed" (not "Stop Serving") since no feed server was
running; Inspector shows "This is a watchlist." for a selected watchlist,
above Console actions, with the disabled follow button reading "Follow in
Console"; and, after the backend-controller fix, a populated-but-unrun
profile reads "Watchlists: loaded" / "Latest run status: no runs yet"
instead of "unavailable".

### Follow-up (UAT batch-5 whole-branch review, finding I1)

The `None`-not-"unavailable" fix above (AC#2) itself introduced a NEW,
narrower honesty gap: `WatchlistsBackendController.get_overview_data`'s
`scope_service is None` branch (the feature genuinely not wired up) also
returned `None`, which renders identically to a healthy watchlist that
has simply never run -- exactly the class of dishonesty this whole UAT
programme exists to remove, one level down. Reviewer confirmed unguarded
by mutation (reverted to a sentinel; existing suites stayed green -- no
test anywhere constructs the controller with `scope_service=None` to
exercise this branch).

Fixed with two new sentinels on `WatchlistsBackendController`
(`NOT_CONFIGURED_STATUS`, `LOOKUP_FAILED_STATUS`), kept apart from both a
real DB-sourced run-status string and `None`/"no runs yet": the former for
`scope_service is None`, the latter for `WatchlistsCollectionsScreen.
_refresh_overview_data`'s own except-handler fallback (a REAL exception
calling the controller -- previously the literal "unavailable", now
consistent with the rest of this task's vocabulary instead of a bare
fault-reading string). `_latest_run_status_text` and `OverviewPane.
_card_value` both map the two sentinels to honest prose ("not connected",
"couldn't check") before either ever reaches a widget, so the raw,
machine-readable literal never leaks into UI text. Three new discriminating
tests (controller-level, screen-method-level via a bare `object.__new__`
instance -- no mount needed, since both inputs are plain reads/attributes
-- and pane-level); all mutation-verified (Edit-tool revert -> RED ->
restored byte-exact, md5).
