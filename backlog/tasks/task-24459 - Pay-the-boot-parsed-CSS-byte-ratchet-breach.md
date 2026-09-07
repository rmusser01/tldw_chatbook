---
id: TASK-24459
title: Pay the boot parsed CSS byte ratchet breach
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - boot
  - css
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`MAX_BOOT_PARSED_CSS_BYTES` is breached: boot-parsed CSS is 862,184 B against a limit of
860,000 B. Every one of those bytes is parsed before first paint.

Growth since the pinned snapshot: `tldw_cli_modular.tcss` +3,424 B (of which
`components/_agentic_terminal.tcss` is +2,156 B and `components/_forms.tcss` +881 B),
`widget_defaults_scoped.tcss` +3,399 B, `widget_defaults_self.tcss` +641 B, including two new
`ConsoleForkChatModal` segments totalling 2,770 B.

Per ADR-097 the constant must not be raised.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_boot_parsed_css_bytes_stay_within_budget` passes on a pristine checkout
- [x] #2 `MAX_BOOT_PARSED_CSS_BYTES` is not raised (TIGHTENED 806,000 -> 804,000)
- [x] #3 The bytes are shed by deferring CSS off the first-paint path or by removing redundant rules, not by moving the measurement
- [x] #4 The CSS bundle regenerates from its sources with no drift
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
NOT IMPLEMENTED in the 2026-08-29 review pass. This is the one ratchet of the four still red:
862,184 B against an 860,000 B limit, so 2,184 B must be shed and the constant must not rise.

Because it is red, task-24461 deliberately EXCLUDED this guard when wiring the boot budgets into
`perf-guard.yml` -- including it would have failed every unrelated PR. It joins that step when
this task lands, and the workflow comment says so.

Growth since the pinned snapshot: `tldw_cli_modular.tcss` +3,424 B (of which
`components/_agentic_terminal.tcss` +2,156 and `components/_forms.tcss` +881),
`widget_defaults_scoped.tcss` +3,399, `widget_defaults_self.tcss` +641, including two new
`ConsoleForkChatModal` segments totalling 2,770 B.

Most tractable route is the `ConsoleForkChatModal` segments plus the `_agentic_terminal.tcss`
growth; it overlaps task-24451. A mechanical dead-selector sweep was attempted and abandoned --
the detection used `\w` in a POSIX ERE and reported every one of 609 ids as dead, which is
exactly the kind of false positive that would have deleted live CSS.

### Re-measured 2026-09-04 at dev tip `b7f8efde73` (schedules programs close-out)

The pinned snapshot (`boot_budget_snapshots/boot_css_bytes.json`, total 780,368 B) was
last written on 2026-08-31 by `b62407e258` and PREDATES both schedules programs, so the
per-source attribution in the Notes above is now stale. Current measurement:

```
boot-parsed CSS grew to 818,874 B (ratchet limit 806,000 B)   # 12,874 B over
```

Largest changed segments vs the snapshot:

| segment | snapshot | now | delta |
|---|---|---|---|
| `tldw_cli_modular.tcss::features/_scheduling.tcss` | 5,994 | 16,165 | **+10,171** |
| `tldw_cli_modular.tcss::components/_agentic_terminal.tcss` | 85,753 | 91,392 | +5,639 |
| `screen_agentic_console.tcss::(whole file)` | 95,707 | 97,577 | +1,870 |
| `tldw_cli_modular.tcss::components/_settings_splash_theme.tcss` | 2,574 | 3,825 | +1,251 |
| `tldw_cli_modular.tcss::core/_variables.tcss` | 5,408 | 6,450 | +1,042 |

Plus new scoped segments, of which the schedules programs contribute `ResultsTab` (531 B)
and `DefinitionAuditView` (181 B), against `TaskDetail` shrinking 746 -> 623 (-123 B).

**Attribution.** The schedules share is ~**+10.8 KB**, essentially all of it the
`features/_scheduling.tcss` growth. Cross-checked against the raw source file across the
redesign program: 6,848 B at `81509271a3^` (before redesign PR-1) -> 16,112 B at
`b7f8efde73` (after redesign PR-4), i.e. **+9,264 B**; handoff PR-6 (`c07a2edbbe`) added
nothing to that file. The remainder of the breach is other programs'.

Two consequences for whoever picks this up:

1. The redesign is the single largest contributor to the current breach, so the cheapest
   route is now a `_scheduling.tcss` pass (the file more than doubled while the screen
   collapsed from four tab surfaces to one — the retired-surface rules are worth checking
   for deletion, and `b7f8efde73` already removed the `TabbedContent` block) rather than
   the `ConsoleForkChatModal`/`_agentic_terminal` route the original Notes recommend.
2. Do not re-pin the snapshot as the fix. AC #2 and #3 still stand: the constant does not
   rise and the measurement does not move. The snapshot is refreshed only by
   `scripts/update_boot_budget_snapshots.py` AFTER the bytes are genuinely shed.
<!-- SECTION:NOTES:END -->

## Implementation Plan (the how)

1. Generalize the TASK-25812 agentic split machinery in `css/build_css.py` to a
   table of screen-owned splits; add `features/_evals.tcss` (owner prefix
   `evals`, sheet `screen_feature_evals.tcss`) and `features/_scheduling.tcss`
   (prefixes `scheduling`/`schedules`, sheet `screen_feature_scheduling.tcss`).
   The proven conservative classifier moves only owner-pure blocks; the
   demotion pass and later-module seeding apply per-module; add a cross-split
   moved-selector disjointness assert.
2. Wire the sheets onto `EvalsScreen.CSS_PATH`, `SchedulesWorkbench.CSS_PATH`
   and `WorkbenchHostScreen.CSS_PATH` (parsed on first visit, not before first
   paint -- the same mechanism as the library/settings agentic sheets).
3. Rebuild, refresh boot-budget snapshots, TIGHTEN `MAX_BOOT_PARSED_CSS_BYTES`
   to measured + 25,000 slack per ADR-097's convention.
4. Add `Tests/Performance/test_boot_css_byte_budget.py` to perf-guard.yml's
   ratchet step (the task-24461 join step) and update its exclusion comment.
5. Extend the css-build integrity tests to the new sheets; unit-test the
   generalized splitter (multi-prefix move, non-owner-token keep, synthetic
   cross-split collision) and mutation-test the new guards.
## Implementation Notes

Generalized the TASK-25812 agentic split into a table of screen-owned
splits (`SCREEN_OWNED_SPLITS` in `css/build_css.py`: a frozen
`ScreenOwnedSplit` spec per module -- owner prefixes, sheet filenames,
pinned tokens). Two modules joined the agentic one:

- `features/_evals.tcss`: 39,695 of 40,518 B is `evals-*`-pure ->
  `screen_feature_evals.tcss`, loaded via `EvalsScreen.CSS_PATH`. Owner
  audit: every selector consumer lives under `UI/Evals/` or
  `UI/Screens/evals_screen.py` (repo-relative-path greps, then re-checked
  with the comment-stripping classifier -- a bare-token grep also "found"
  `lab-rail`/`ds-*` tokens that were comment prose, the #2281 audit trap
  in a new costume).
- `features/_scheduling.tcss`: only the `scheduling-*`/`schedules-*`-pure
  half (7,936 B) moves -> `screen_feature_scheduling.tcss` on
  `SchedulesWorkbench.CSS_PATH` (+ `WorkbenchHostScreen`, belt-and-braces).
  Helper classes (`pane-hidden`, `detail-value-row-*`, bare status
  classes) stay in the bundle by the conservative classifier -- no pin
  list needed. A WHOLESALE move was rejected: `TaskDetail` is a type
  selector and Evals composes a same-named widget class, and
  `.needs-attention` is set by `library_notes_canvas`.

Safety at build time: per-module demotion pass (later-module seeding via
the parameterized `_later_module_selectors`), per-module variables
preamble, and a cross-split moved-selector disjointness guard in
`build_screen_owned_sheets` -- structurally unreachable today because the
demotion pass fires first (verified by writing the natural-path test and
watching it not raise; the guard is a documented backstop, tested at its
own level). Measured zero selector collisions for both modules against
all later bundle modules and all generated post-bundle sheets.

Census: 826,956 (BREACHED vs 806,000) -> 779,365 on the rebased tree.
Ratchet TIGHTENED to 804,000 (measured + standard 25,000 slack, ADR-097;
lowering needs no ledger row). `test_boot_css_byte_budget.py` JOINED
`perf-guard.yml`'s ratchet step -- the exclusion comment now records the
join and forbids removal as the response to a future red.

Verified: css-build integrity 25 passed (exact-rebuild contract
mutation-tested red on a hand-edited sheet); bundle-sync check green with
the new sheets enrolled; full perf-guard suite 26 passed (destination
tour visits Schedules; both new sheets parse under the app's theme
variables); boot budget snapshots refreshed via the script (CSS snapshot
only -- the module-census snapshots were deliberately reverted, their
drift belongs to dev). Evals/scheduling UI selection run as paired arms
against the pristine merge base (see PR body for counts).

**The incident this task minted (also in lessons-testing-evidence.md):
Screen.CSS_PATH loads under EVERY app, including the unstyled-tier
harnesses.** The first wiring put the sheets on the owning screens'
`CSS_PATH` (the TASK-25812 library/settings pattern); the paired arm then
flipped three destination-shell geometry tests, and the probe showed why:
`ConsolidatedCSSApp` harnesses load no app bundle, so harness-mounted
workbenches got ONLY the moved half of the module -- a hybrid of the two
tiers where the automation-detail overlay covered the follow button.
Rewired to an app-owned seam (`TldwCli._SCREEN_OWNED_ROUTE_CSS` +
`_ensure_screen_owned_css`, mirroring Textual's `_load_screen_css`) so
real-app behavior is identical and harnesses keep their tier contract;
two guards pin the seam (map completeness + a CSS_PATH ban on the owning
screens) and a functional test proves boot-absent/visit-present. The
production-CSS harnesses' hard-coded sheet list (already bitten once by
25812) now derives from `APP_STYLESHEETS`. After the rework the four
affected UI files show an IDENTICAL 16-failure set to pristine dev
(paired arms, zero divergence).

Files: `css/build_css.py`, `css/check_bundle_sync.py`, `app.py` (rebuild
staleness list), `UI/Screens/evals_screen.py`,
`UI/Screens/scheduling/schedules_workbench.py`,
`UI/Screens/scheduling/workbench_host_screen.py`, generated sheets +
bundle, `Tests/UI/test_css_build_integrity.py` (4 new contracts),
`Tests/UI/consolidated_css.py`, `Tests/Performance/test_boot_css_byte_
budget.py`, `boot_budget_snapshots/boot_css_bytes.json`,
`.github/workflows/perf-guard.yml`.
