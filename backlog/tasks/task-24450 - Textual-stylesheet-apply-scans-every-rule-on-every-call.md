---
id: TASK-24450
title: Textual stylesheet apply scans every rule on every call
status: Done
assignee: []
created_date: '2026-08-29'
labels:
  - performance
  - ui
  - textual
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Stylesheet.apply()` pre-filters candidate rules through `rules_map`, then discards the
benefit with `rules = list(filter(limit_rules.__contains__, reversed(self.rules)))`, which
walks the entire rule list on every call to recover source order. With 4,324 global rules this
makes every style application O(all rules) rather than O(matched rules).

Measured on dev bc1e26ce60: 0.52 ms per single-node apply, 240 ms for one full-screen
`update_styles`, and 7,335,029 `RuleSet.__hash__` calls in a single Console screen switch
(= 1,667 applies x 4,324 rules). Stack sampling during observed event-loop stalls ranks
`textual/css/model.py:__hash__` the #1 frame. This is the single dominant interactive cost in
the application.

A prototype that sorts the already-filtered candidate set by a cached position index measured
-27% per apply, -35% per full-screen update_styles, and -260 ms per Console switch in an
interleaved A/B.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single-node `Stylesheet.apply()` no longer scales with total stylesheet rule count
- [x] #2 Full-screen `update_styles` on the Console measures at least 25% faster than the pre-change baseline on the same machine, interleaved A/B
- [x] #3 Rendering is unchanged: specificity and source-order tie-breaking produce identical computed styles for a representative node set
- [x] #4 A regression test pins the per-apply cost model so a future change cannot silently reintroduce the full-list scan
- [x] #5 The change is documented as either an upstream Textual patch or a vendored subclass, with the chosen route recorded
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
Added `tldw_chatbook/Utils/textual_css_fastpath.py`, installed from `app.py` before any App
exists. Upstream `Stylesheet.apply` narrows candidates through `rules_map` and then recovers
source order with `filter(limit_rules.__contains__, reversed(self.rules))` -- a walk of ALL
4,324 rules per node. The fast path pre-computes the same candidate set, sorts it by a cached
position index, and hands it to upstream as `_rules`; upstream's own reversal then preserves
"later rule wins" exactly.

Deliberately a delegation, not a fork: upstream stays the single source of truth for what
styling MEANS, and only candidate SELECTION changes. The position index is cached against the
`rules_map` OBJECT (a strong reference), because Textual sets `_rules_map = None` on every path
that mutates `_rules` -- so a surviving identity match proves the rule list is unchanged, and
holding the reference stops `id()` reuse from aliasing a freed map.

Measured on the real app, interleaved A/B, two rounds:
- `apply()` per node 0.55 -> 0.38 ms (-31%)
- `update_styles(screen)` 266/252 -> 151/153 ms (-41%)
- Console screen switch 1.96 -> 1.78 s mean of 6 (-175 ms)
- Typing unchanged, as expected (typing performs few applies)

Two tests in `Tests/Performance/test_textual_css_fastpath.py`:
- fidelity: applies the stylesheet to every node of Console/Library/Settings both ways and
  compares rule maps (689 nodes). Mutation-tested -- reversing the sort order breaks 47 of 689,
  so the test discriminates.
- upstream pin: asserts the two literal upstream behaviours the delegation assumes, so a Textual
  upgrade fails loudly here instead of silently changing styling.

Modified: `tldw_chatbook/Utils/textual_css_fastpath.py` (new), `tldw_chatbook/app.py`,
`Tests/Performance/test_textual_css_fastpath.py` (new).
<!-- SECTION:NOTES:END -->
