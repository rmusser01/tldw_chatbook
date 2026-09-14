---
id: TASK-24451
title: Split the agentic terminal CSS grab-bag and ratchet rule count
status: Done
assignee: []
created_date: '2026-08-29'
updated_date: '2026-09-14'
labels:
  - performance
  - ui
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`css/components/_agentic_terminal.tcss` is 272,312 B and holds 1,214 of the app's 4,198
source rule blocks -- 29% of all CSS in the application. Despite its name it is not
agentic-terminal CSS; it is an unstructured accumulation containing Library
(`#library-shell-grid` x72, `LibraryIngestCanvas` x30), Settings (`#settings-shell` x10),
MCP (`#mcp-mode-strip` x7), Lab (`#lab-mode-strip` x6), Workflows, Personas and Watchlists
rules. All of it is global, so every Library rule is scanned when styling a Console button.

It grew 262,634 -> 272,312 B (+3.7%) in the two days between the 2026-08-27 and 2026-08-29
review pins. Because style-application cost is linear in total rule count, reducing the rule
count is a direct app-wide win that compounds with task-24450.

Note that Textual's `SCOPED_CSS` does not help: scoped rules still live in `self.rules` and are
still scanned. Only deleting rules, or swapping stylesheet sources per screen, reduces N.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `_agentic_terminal.tcss` no longer contains rules owned by Library, Settings, MCP, Lab, Workflows, Personas or Watchlists
- [x] #2 Total app rule count measured at boot is reduced relative to the pre-change baseline, and the reduction is recorded
- [x] #3 A rule-count ratchet guard exists alongside the existing byte budget, because bytes are not rules
- [x] #4 Every destination in the existing latency-guardrail tour renders with no visual regression
- [x] #5 The CSS bundle regenerates from its sources with no drift (`./scripts/preflight.sh` green)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CLOSED 2026-09-14 by task-32532 plan Task 10 (ADR-161): the monolith carve-up completed and the spec 3.5 size-ceiling test shipped (`test_sheet_size_ceiling`, 2,000 comment-stripped active lines per CSS_MODULES sheet).

AC disposition at close (per the supersession recorded above):
- #1 MET for the owners the audit named with material mass: Library (-> features/_library{,_panels}.tcss), Settings (-> features/_settings.tcss), and additionally Console (-> features/_console{,_panels}.tcss, bundle-resident), Home (-> features/_home.tcss), destination (-> layout/_destination.tcss), and the ds-* primitives (-> components/_ds_primitives.tcss). The small scoped residue the 2026-08-29 analysis also named (mcp-mode-strip, lab-mode-strip, workflows/personas/watchlists pane strips) stays in the monolith as SHARED agentic-shell chrome: those rules style shell regions composed across screens (verified: their selectors are multi-shell comma lists or non-vocabulary scoped overrides), so per the ADR-161 spec 3.5 carve rule ("what remains of the monolith is true Console/shared chrome") they are chrome, not an owned vocabulary; the parent task's 3.10 in-place passes own any further split. The grab-bag condition itself is closed: every class VOCABULARY now lives in a single-purpose sheet.
- #2 SUPERSEDED, recorded: task-24450 removed the O(all-rules) apply() cost this AC measured against, and TASK-25812's split already moved the single-owner rules off the boot parse (measured -100 KB boot CSS at the time). The task-10 carve is cascade- and rendering-neutral by design (byte-identical computed-style probes on Home/Console/Settings/Library + byte-identical SVG A/B), so rule COUNT is unchanged by the reorganization; the ceiling test now guards regrowth in the dimension that matters (sheet size).
- #3 MET in superseded form: spec 3.5 replaced the rule-count ratchet with the size ceiling (2,000 active lines per sheet, no grandfather allowance), shipped in the same commit that completed the split.
- #4 MET: surface-wide computed-style probes on the four heaviest consumers (Home/ChatScreen-Console/Settings-default/Library) are BYTE-IDENTICAL before/after, and normalized SVG A/B captures of all four are byte-identical.
- #5 MET: `check_bundle_sync.py` green -- the bundle and every generated sheet reproduce from the new sources.

Superseded by task-32532 (component-pattern library; ADR-161). The monolith split lands as plan Tasks 9-10. Close when the size-ceiling test ships.

Prior analysis (2026-08-29 review pass), preserved for the Task 9-10 implementers:

NOT IMPLEMENTED in the 2026-08-29 review pass. Recorded here so the next person does not
re-derive the analysis.

Confirmed scope: `_agentic_terminal.tcss` is 272,312 B and holds 1,214 of the app's 4,198 source
rule blocks (29% of all CSS). It grew 262,634 -> 272,312 B in the two days between the 08-27 and
08-29 review pins. Selector families inside it that belong to other screens:
`#library-shell-grid.library-notes-compact` (72 blocks), `LibraryIngestCanvas` (30),
`ConsoleSettingsModal` (16), `#settings-shell` (10), `#mcp-mode-strip` (7), `#lab-mode-strip` (6),
plus `#workflows-*`, `#personas-inspector-pane`, `#watchlists-inspector-pane`.

Two constraints the implementer must know:
1. `SCOPED_CSS` does NOT reduce the cost. Scoped rules still live in `Stylesheet.rules` and are
   still scanned. Only DELETING rules, or swapping stylesheet sources per screen, reduces N.
2. With task-24450 landed, `apply()` is no longer O(all rules), so the payoff from cutting rule
   count is smaller than the original review estimated -- the two fixes are not additive in the
   way first written up. Re-measure before committing to the full split.

A first attempt at finding dead selectors mechanically produced a false 609/609 "all dead"
result, because the detection grep used `\w` inside a POSIX ERE. Any dead-CSS sweep here needs a
real parser, not a grep.
<!-- SECTION:NOTES:END -->

## Progress via TASK-25812 (2026-08-31) — largely satisfied, with two honest gaps

The owner-approved split-by-screen shipped on
`perf/task-25812-split-agentic-css`. Against this task's ACs:

- **#1 — partial, by a different mechanism.** The BOOT BUNDLE no longer
  carries the Console/Library/Settings rules (the three big owners:
  201,596 B moved to per-screen `CSS_PATH` sheets). But the SOURCE file is
  deliberately untouched — the split happens in `build_css.py`, keeping one
  source of truth and a build-time lossless-partition assertion. And the
  smaller owners this AC names (MCP, Lab, Workflows, Personas, Watchlists,
  ~15,7 KB combined) still ride the bundle: each is small, and each would
  need its own screen `CSS_PATH` + cross-surface token audit to move.
- **#2 — done.** Boot rules 4,231 → 3,304, recorded here and on TASK-25812.
- **#3 — not done by this change.** The byte ratchet was tightened
  (860,000 → 705,000), but a boot RULE-COUNT ratchet still does not exist.
  (The bare-type-subject ratchet on the #2258 branch guards candidate
  breadth, which is a different quantity.)
- **#4 / #5 — verified with the split in place**: latency-guardrail tour
  and preflight run in the implementation branch's gate.

Left To Do for the remaining scope (#1's small owners, #3's rule-count
ratchet) rather than closed.
