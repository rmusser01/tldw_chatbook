---
id: TASK-15999
title: 'Extend CSS consolidation as live-source headroom erodes'
status: To Do
assignee: []
created_date: '2026-08-14 01:10'
labels:
  - perf
  - css
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15450 moved 57 of ~184 DEFAULT_CSS-declaring classes into the generated sheets, bringing the post-tour live-source count to 49 against Textual's 64-source parse-cache cliff, with the mounted-tour regression test asserting < 56 (7 headroom below the threshold; dev's very next merge added one class, 48→49). The remaining ~127 classes were deliberately left: consolidating a never-mounted widget makes its CSS live from boot and taxes every `Stylesheet.apply`. When the tour test starts failing (or headroom drops below ~4), consolidate the next tranche — prioritizing classes that actually mount in common flows — rather than raising the threshold. The BUNDLED_CSS mechanism, prefix-baking, and tie-breaker discipline are all in place in `tldw_chatbook/css/build_css.py` / `widget_css.py`; this is repeat-application, not new design. Owner ruling applies: never raise the test threshold to make room (owner ruling 2026-08-11: stability over quick wins). Found during the TASK-15450 CSS-consolidation review (PR #1616, merged `c3ed2854a`); evidence in the session review record and `Docs/Design/2026-08-11-input-latency-audit.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Post-tour live-source count restored to <= 49 with the tour test's threshold UNCHANGED
- [ ] #2 Newly consolidated classes chosen by mount frequency (state the selection evidence)
- [ ] #3 Cascade parity evidence per the 15450 method (computed-style diff over the tour)
<!-- AC:END -->
