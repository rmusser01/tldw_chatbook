---
id: TASK-32187
title: >-
  Pay down the boot-parsed CSS byte ratchet breach and re-tighten the limit
status: In Progress
assignee: []
created_date: '2026-09-09 12:30'
labels:
  - performance
  - css
  - infrastructure
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`dev` is red on `Tests/Performance/test_boot_css_byte_budget.py`: the
boot-parsed CSS census reached 805,625 B against the 804,000 B ratchet.
Every one of those bytes is parsed by Textual before first paint. The
breach arrived with the Library ▸ Notes fix wave (first red at
`de6793b9f5`, the merge of PR #2540) and the guard's per-segment diff names
`LibraryNoteImportCanvas` as the largest grower.

ADR-097 makes `MAX_BOOT_PARSED_CSS_BYTES` a ratchet that never rises: the
legitimate responses are to defer the cost off the boot path, shed
equivalent bytes elsewhere in the same PR, or record an owner exception in
the ADR's ledger. Raising the constant is not an option.

The 2,632 B the Import canvas actually costs is not where the fat is. The
useful outcome is structural headroom that survives the next feature, not a
trim of whichever widget happened to cross the line — so this pays the
breach down by taking the largest remaining screen-owned module off the
boot path and by stopping the generated widget-defaults sheets from
carrying text that styles nothing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `Tests/Performance/test_boot_css_byte_budget.py` passes, with the
  measured total materially below the previous limit rather than just
  under it.
- [x] #2 `MAX_BOOT_PARSED_CSS_BYTES` is LOWERED (never raised) to the
  measured total plus the guard's standard 25,000 B slack, per ADR-097's
  tightening convention, with the reason recorded beside the constant.
- [x] #3 Every selector token moved to a lazily-loaded sheet is audited
  against its compose sites using repo-relative paths, and the audit
  filter is shown able to return a negative.
- [x] #4 The Watchlists screen still renders styled: the moved rules arrive
  on first navigation to the route, verified in the running app, not only
  in tests.
- [x] #5 The CSS bundle and every generated sheet still reproduce from
  their sources (`check_bundle_sync.py`).
- [x] #6 The opposing source-count ratchet
  (`Tests/UI/test_widget_css_consolidation.py`) gains no new violation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the breach and read ADR-097 plus the guard's module docstring.
2. Price every candidate paydown with the real classifier rather than a
   prefix histogram, and rank by bytes shed per unit of risk.
3. Run the mandatory owner audit on the chosen module, negative-controlled.
4. Implement, rebuild the bundle, refresh the snapshot, lower the constant.
5. Verify: the guard, the bundle-sync check, the source-count ratchet, the
   rest of `Tests/Performance/`, the Watchlists UI suite, and a live run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two paydowns, 805,625 B -> 743,638 B, and the limit lowered 804,000 ->
768,000.

**1. `features/_watchlists.tcss` became the fourth `ScreenOwnedSplit`
(-51,369 B).** It was the largest un-split bundle module (57,721 B). A
single `watchlists-*` prefix moves almost nothing from it, because the
screen's panes and modals each carry their own vocabulary — `sources-*`,
`items-*`, `artifacts-*`, `bpm-*`, `kbm-*`, `svm-*`, `wl-*`, `inspector-*`,
`overview-*`, `rules-*`. All of them are spelled out in the spec, the way
the scheduling split already spells `("scheduling", "schedules")`; the
classifier only applies them inside this one module, so the blast radius is
the Watchlists feature module and nothing else. The sheet loads through the
existing `TldwCli._SCREEN_OWNED_ROUTE_CSS` seam (TASK-24459) rather than the
screen's `CSS_PATH`, because a `CSS_PATH` also styles the UI-test harnesses
that deliberately model the unstyled tier.

Owner audit, repo-relative paths, all 132 moved tokens: every compose site
is under `tldw_chatbook/UI/Watchlists_Modules/` or
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`. The audit was
negative-controlled first — an owner glob matching nothing flagged 128/132 —
because the previous cross-surface audit filtered ABSOLUTE paths and was
silently vacuous. Two off-owner hits were run down and cleared as substring
false positives: `library-collections-items-toolbar` and
`settings-overview-card` are longer ids that merely contain a moved token.
The Console's `watchlists-operation-*` cards are styled by
`features/_chat.tcss`, not this module, so they never enter the moved set.

**2. `widget_css.render_stylesheets` stopped emitting rule-less segments
(-10,618 B).** `split_scoped_css` preserves line positions, so the stream a
block contributes nothing to still carries that block's comments and the
blank lines standing in for the other stream's rules. The existing
`if not text.strip()` guard let all of that through whenever a comment was
present, and both sheets are boot-parsed: 7,901 B of `widget_defaults_scoped`
and 2,717 B of `widget_defaults_self` were segments containing no rule at
all. `LibraryNoteImportCanvas` — the widget that caused this breach — was
one of them, contributing a 393 B scoped stanza of pure whitespace and
comments. The new `_stream_has_rules` skips a stream when no `{` survives
comment stripping; TCSS has no at-rules and `isolate_local_variables` has
already removed `$var` definitions, so that test is exact.

Deliberately NOT done: moving `LibraryNoteImportCanvas`'s own class CSS to a
Library-owned lazy sheet. The widget-defaults sheets sit in Textual's
default-CSS tier (tie-breakers 0 and -1,000,000); a screen sheet sits in the
app-CSS tier and parses after the bundle, so routing widget defaults there
would flip every tie the app bundle currently wins against them. That is a
cascade change needing its own design, and it would buy 2,632 B — 4% of what
these two paydowns buy.

Modified: `tldw_chatbook/css/build_css.py`, `tldw_chatbook/css/widget_css.py`,
`tldw_chatbook/app.py`, `Tests/Performance/test_boot_css_byte_budget.py`,
`Tests/Performance/boot_budget_snapshots/boot_css_bytes.json`,
`Tests/UI/test_css_build_integrity.py`,
`Tests/UI/test_widget_css_consolidation.py`, plus the regenerated
`tldw_cli_modular.tcss`, `widget_defaults_self.tcss`,
`widget_defaults_scoped.tcss` and the new
`tldw_chatbook/css/screen_feature_watchlists.tcss`.
<!-- SECTION:NOTES:END -->
