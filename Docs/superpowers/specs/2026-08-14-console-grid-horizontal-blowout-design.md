# Console grid horizontal blowout design

**Task:** TASK-16220
**Date:** 2026-08-14
**Status:** Approved for implementation planning

## Goal

Keep the Console workspace usable when the Inspector is open at 120 columns,
without weakening the Context rail's 30-column content contract or making rail
visibility depend accidentally on Textual's fractional-width failure mode.

## Measured failure mechanism

The workspace grid is 120 columns wide and has a one-cell border on each side,
leaving 118 usable columns. At 120 columns the production shell can open all
three fractional children:

- Context: `3fr`, minimum 30
- Transcript: `13fr`, minimum 56
- Inspector: `4fr`, minimum 34

Those minimums total 120, which is two columns more than the grid's 118-column
content box. Textual 8.2.7's `resolve_fraction_unit()` successively min-clamps
all three fractional children. When no unresolved fraction remains, it returns
the original remaining space (118) as the fraction unit. The result is therefore
`3 * 118`, `13 * 118`, and `4 * 118`: the observed widths 354, 1534, and 472.

This is not caused by the rails' descendant content or by the app stylesheet.
A minimal Textual reproduction with only a bordered `Horizontal` and the three
width/minimum pairs produces the same 354/1534/472 result. Changing only the
Transcript minimum from 56 to 0 produces the healthy 30/54/34 split. The prior
minimal reproduction did not fail because it already used that zero Transcript
minimum.

The product regression began when TASK-2154 raised the Context minimum from 24
to 30 while preserving the 120-column Inspector auto-open path. The effective
state for that automatic open does not carry the compact-override flag that
waives the Transcript minimum.

## Responsive rail rule

The effective Console layout follows these rules:

1. Below 84 columns, the existing single-pane behavior remains unchanged.
2. From 84 through 99 columns, the existing narrow/explicit-toggle behavior
   remains unchanged.
3. From 100 through 149 columns, Inspector has priority when both rails would
   otherwise be open:
   - Inspector stays open.
   - Context is effectively collapsed.
   - The Transcript minimum is waived through the existing compact-override
     mechanism.
4. At 150 columns and above, both saved open preferences are honored.

This is a rendering decision. Responsive collapse does not rewrite the saved
Context preference. Collapsing Inspector, or growing the terminal to 150
columns, restores Context automatically when its saved preference is open.

Responsive rail replacement must also preserve a usable keyboard position.
When a resize hides the rail that currently owns focus, focus moves to that
rail's now-visible reveal handle after the visibility update. This applies to
the 117-to-118 Context-to-Inspector transition and the 128-to-129
Inspector-to-Context transition. It does not change the established manual
collapse contract, and it does not run in single-pane mode where reveal
handles are intentionally hidden.

The collapsed Context handle remains functional. If the user explicitly asks
to open Context while Inspector owns the compact band, the action switches the
visible rail by closing Inspector through the existing preference update and
then showing Context. It must not be a silent no-op.

At 120 columns with the ordinary horizontal rail handles, the intended shape is
approximately:

```text
| Context handle 13 | Transcript 71 | Inspector 34 |
```

All exact widths continue to be owned by Textual; the product contract is that
each displayed child is contained by the grid and the selected rail priority is
deterministic.

## Implementation boundary

Add one final effective-state resolution step after every automatic Inspector
open has been applied. In the 100-149 band it will:

- collapse effective Context when Inspector is open;
- preserve `preferred_left_open` and the persisted payload;
- mark Inspector as the compact override so the existing Transcript minimum
  waiver applies; and
- leave the established below-100 and 150-plus rules untouched.

The same pure finalizer in `Chat/console_rail_state.py` must be used by
compose-time and mounted/resize-time rail state construction so the first frame
and later resize frames cannot diverge. The priority decision does not add a
new `ChatScreen` method: the screen is already governed by the one-way size and
method-count ratchet, so existing handlers and resize seams call the pure rail
state helper instead.

Every Context-reveal entry point (the collapsed handle and the visible
Workbench action whose event id is exactly `attach-context`) will use the same
pure reveal decision and the existing preference setter. In the compact band
that decision switches away from Inspector rather than requesting an
impossible two-open state. The similarly named `#console-attach-context` and
`#console-staged-context-attach` buttons remain file-picker actions and are not
part of this rail-switch contract.

The resize bands must include every boundary that changes effective rail
geometry. The current `console_rail_width_band()` treats every width at or above
100 as one band and therefore misses 149-to-150 restoration. It will distinguish
the 100-149 Inspector-priority band from 150-plus. The standard-width Inspector
auto-open eligibility edges at 118 and 129 must also trigger recomputation so
compose and resize agree. The resize event's resolved width will be threaded
through auto-open eligibility and finalization; those steps must not reread a
possibly stale `self.size`.

No CSS sizing rewrite, storage migration, dependency change, or Textual patch is
required.

## Evidence

Focused tests will cover:

- the pure 99/100/149/150 boundary and preservation of preferred Context state;
- automatic Inspector open at 120 receiving compact-override authority;
- both explicit Context activation surfaces switching away from Inspector
  rather than doing nothing;
- live resize transitions at 117/118, 128/129, and 149/150 using the resize
  event width as the single authority;
- responsive 117-to-118 and 128-to-129 replacement moving focus from the
  hidden rail to its visible reveal handle, while manual collapse retains its
  existing focus behavior;
- every displayed workspace-grid child staying inside the 120-column viewport
  under the production hierarchy and shipped stylesheet;
- the four existing `test_console_shell_regions.py` `size2` regressions;
- the two parameterized session-row width-budget cases, updating their stale
  12-cell label oracle to the current deliberate 13-cell label-plus-gutter
  contract; and
- mutation evidence that removing the entire effective-state finalizer restores
  the 120-column blowout; separate state assertions pin Context collapse and
  compact-override authority. Existing below-threshold explicit-toggle tests
  continue to pin the main-column waiver where it is geometrically
  load-bearing.

Because `chat_screen.py` already exceeds the repository's historical screen
ratchet on the current development baseline, the focused architecture evidence
will record before/after line and method counts and require that this task adds
no `ChatScreen` method. The ratchet ceiling will not be raised or rewritten by
this bug fix.

Only Console rail/layout tests and static checks for touched files will run.
The unrelated pre-existing recovery-copy failure in
`test_console_rail_width_budget.py` is outside TASK-16220.

## ADR decision

**ADR required:** yes
**ADR path:** `backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md`
**Reason:** the change defines a durable UX conflict rule between two persisted
rail preferences in the compact-width band. ADR-043 already owns compact rail
visibility and explicit-toggle behavior, so it will be refined rather than
duplicated.

The ADR refinement will preserve automatic Inspector-open eligibility and
preference semantics, while replacing its obsolete claim that auto-open leaves
compact-override flags false. The rendered auto-open result now intentionally
grants Inspector priority and compact-override authority.

## Alternatives rejected

### CSS clamps or fixed percentages

This could hide the blowout at one width, but it would leave rail precedence
implicit and fight the rails' existing intrinsic content guarantees.

### Global Textual resolver patch

Changing third-party fraction resolution is much broader than the Console
policy defect and would affect every fractional layout in the application.

### Keep both rails and waive only the Transcript minimum

This yields a mathematically valid 30/54/34 layout at 120 columns, but it does
not satisfy the chosen degradation rule: Inspector must take priority and
Context must yield throughout 100-149 columns.
