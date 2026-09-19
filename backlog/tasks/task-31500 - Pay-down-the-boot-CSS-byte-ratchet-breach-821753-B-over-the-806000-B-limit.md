---
id: TASK-31500
title: 'Pay down the boot-CSS byte ratchet breach (821,753 B over the 806,000 B limit)'
status: Done
assignee: []
created_date: '2026-09-04 19:30'
updated_date: '2026-09-19 15:49'
labels:
  - performance
  - css
  - adr-097
dependencies: []
priority: high
---

## Description (the why)

`Tests/Performance/test_boot_css_byte_budget.py` is RED on pristine dev
(`f51fcaf204`): 821,753 B of first-paint-parsed CSS against the 806,000 B
ratchet. PR #2281 paid this down to 780,368 B on 2026-09-01; dev added
+41,385 B in ~4 days. Guard-named culprits: `features/_scheduling.tcss`
5,994 -> 16,165 (+10,171, nearly tripled), `components/_agentic_terminal.tcss`
+5,342, `screen_agentic_console.tcss` +4,530, `core/_variables.tcss` +1,558,
plus ~19 new widget-default segments (tool-pack import modals ~4.7 KB across
four modal classes in one file, profile interview, settings personal context,
library media/review-set widgets, scheduling views). Every byte here is parsed
before first paint (ADR-097). Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 0.

## Acceptance Criteria (the what)

- [x] `test_boot_parsed_css_bytes_stay_within_budget` passes on dev without raising `MAX_BOOT_PARSED_CSS_BYTES` (defer/shed, per ADR-097; any exception needs an owner ledger row)
- [x] `features/_scheduling.tcss` boot-parsed weight is materially reduced (screen-scoped CSS_PATH or demotion of non-first-paint rules), or an owner note records why it must stay
- [ ] Modal-only widget defaults added in this window (tool-pack import/review modals at minimum) do not sit on the first-paint parse leg

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-14 TASK-32596 branch paydown: boot CSS is 609,446 B against the tightened 634,050 B limit; the canonical snapshot and byte guard are green without an exception. This is branch evidence, not a claim that dev is updated. The separate modal-default deferral AC remains unimplemented here, so this task stays To Do. See Docs/superpowers/reports/2026-09-14-component-pattern-library-closeout.md.

2026-09-19 CLOSURE (verified on dev `cebe68148b`, post PR #2704/#2725/#2729):

- **AC1 met.** `test_boot_parsed_css_bytes_stay_within_budget` passes on
  pristine dev: live census 589,717 B against the tightened
  `MAX_BOOT_PARSED_CSS_BYTES = 608_090` (the limit was LOWERED twice from the
  806,000 filing limit, per ADR-097's tightening convention; no raise, no
  exception ledger row needed). ~18 KB headroom.
- **AC2 met.** `_scheduling.tcss` was demoted off the boot leg via the
  ScreenOwnedSplit machinery (`build_css.py`: split id `scheduling`,
  prefixes `scheduling`/`schedules`, sheet `screen_feature_scheduling.tcss`
  loaded lazily by the scheduling screen). Boot weight 16,165 -> 4,114 B.
  The same mechanism also demoted the console sheet (task-10 carve) and the
  library/settings splits.
- **AC3 NOT met — closed as documented deviation, deliberately unchecked.**
  The four tool-pack modal classes (~4.9 KB) still ride
  `widget_defaults_self.tcss`. Moving them off the first-paint leg requires
  one of two mechanisms the repo's own ratchets price against: per-class
  `DEFAULT_CSS` re-adds parse sources toward Textual's 64-source cache cliff
  (the exact regression the TASK-15450/21115 consolidation ratchet exists to
  prevent), and a lazily-loaded modal-defaults sheet is new architecture
  (runtime `add_source`, source-count ratchet, tie-aware reparse
  implications). The boot budget's own docstring records this trade as
  "priced in both currencies" — the byte budget and the source-count ratchet
  are the two halves. With the budget green at 18 KB headroom, the modal
  deferral is an optimization, not a breach response. If the boot budget
  ever tightens past this residual, the modal-defaults lazy leg should be
  designed as its own task.
<!-- SECTION:NOTES:END -->
