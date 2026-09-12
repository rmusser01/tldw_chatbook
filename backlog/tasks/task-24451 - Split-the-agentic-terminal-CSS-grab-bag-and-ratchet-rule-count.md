---
id: TASK-24451
title: Split the agentic terminal CSS grab-bag and ratchet rule count
status: To Do
assignee: []
created_date: '2026-08-29'
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
- [ ] #1 `_agentic_terminal.tcss` no longer contains rules owned by Library, Settings, MCP, Lab, Workflows, Personas or Watchlists
- [ ] #2 Total app rule count measured at boot is reduced relative to the pre-change baseline, and the reduction is recorded
- [ ] #3 A rule-count ratchet guard exists alongside the existing byte budget, because bytes are not rules
- [ ] #4 Every destination in the existing latency-guardrail tour renders with no visual regression
- [ ] #5 The CSS bundle regenerates from its sources with no drift (`./scripts/preflight.sh` green)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
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
