---
id: TASK-24459
title: Pay the boot parsed CSS byte ratchet breach
status: In Progress
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
- [ ] #1 `test_boot_parsed_css_bytes_stay_within_budget` passes on a pristine checkout
- [ ] #2 `MAX_BOOT_PARSED_CSS_BYTES` is not raised
- [ ] #3 The bytes are shed by deferring CSS off the first-paint path or by removing redundant rules, not by moving the measurement
- [ ] #4 The CSS bundle regenerates from its sources with no drift
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
