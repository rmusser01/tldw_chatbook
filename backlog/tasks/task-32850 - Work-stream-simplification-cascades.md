---
id: TASK-32850
title: 'Work stream: simplification cascades — one insight, many deletions'
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
references:
  - qa/cascade-review-2026-09-19/report.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-09-17 core-runtime review filed per-site work (helper adoption, dead code, P3 bundles). A follow-up cascade review (2026-09-19, baseline `origin/dev` `d6e2a46384`) hunted the family-level collapses those tasks do not cover: places where one unifying insight makes multiple components unnecessary. It verified nine cascades (~4,700–8,000 combined deletable LOC, conservative→optimistic) and retired four candidates that are either already collapsed or blocked by an ADR.

The largest: ADR-062's `hosted_chat.py` engine serves only moonshot/zai while ~10,100 LOC of per-provider handlers re-roll the same transport — including both entire summarization libraries. Also verified: tool-provider plumbing shared by 9 providers, the Image/Video generation mirror (pre-sanctioned for merge by ADR-044's own Consequences), the dead legacy chat tail, the duplicated console provider-selection builder, and three small riders.

This is a work-stream parent. Its child tasks are the individual units; close this one when they are all closed. Full evidence with file:line citations is in `qa/cascade-review-2026-09-19/report.md`; the blocked/retired candidates are recorded there so they are not re-litigated. Execution risks (red engine contract suite, diagnostic-ledger re-keying, metric-label renames, DeepSeek dual-API sequencing) are in the report's "Execution risks" section and wired into the children as dependencies.
<!-- SECTION:DESCRIPTION:END -->
