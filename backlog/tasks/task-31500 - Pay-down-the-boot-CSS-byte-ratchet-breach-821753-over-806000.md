---
id: TASK-31500
title: Pay down the boot-CSS byte ratchet breach (821,753 B over the 806,000 B limit)
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
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

- [ ] `test_boot_parsed_css_bytes_stay_within_budget` passes on dev without raising `MAX_BOOT_PARSED_CSS_BYTES` (defer/shed, per ADR-097; any exception needs an owner ledger row)
- [ ] `features/_scheduling.tcss` boot-parsed weight is materially reduced (screen-scoped CSS_PATH or demotion of non-first-paint rules), or an owner note records why it must stay
- [ ] Modal-only widget defaults added in this window (tool-pack import/review modals at minimum) do not sit on the first-paint parse leg
