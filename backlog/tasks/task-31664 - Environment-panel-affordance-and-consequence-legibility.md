---
id: TASK-31664
title: >-
  Environment panel affordances: mark actionable rows, name consequences, acknowledge Refresh
status: To Do
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P2: Enter has five outcome classes (expand in place, full-screen
navigation, leave the app, append to the composer draft, nothing) on
visually identical rows; "Commit or push" performs navigation and omits
the "…" its own destination uses; Refresh produces zero visible feedback
for 11.7 measured seconds when data is fresh (it works — ≤0.3s when stale
— but is indistinguishable from dead); "stale" is color-only in the exact
hue of error ($ds-status-blocked ≡ $ds-status-error, 2.53:1 on banded
rows). Repo precedent for the fix: the left rail's System line trailing ▸
(chat_screen ~7983) and Change Review's Commit…/Push… ellipses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A consistent trailing-marker convention distinguishes expand rows, rows that open another surface, and rows that modify the composer draft; inert rows carry none
- [ ] #2 "Commit or push · N files" is renamed to name what it does (e.g. "Review & commit… · N files")
- [ ] #3 Refresh shows a transient acknowledgment (e.g. "Refreshing…") even when the data comes back unchanged
- [ ] #4 Stale state carries a text marker alongside color, and stale/error no longer share an identical hue on rows a user must read
- [ ] #5 The UNBOUND copy names the true cause or goes cause-agnostic: workspace_roots == () also occurs when Change Review consent is not ENABLED for a bound folder (the common default), when the consent service is absent/raises, and when all bound roots are skipped — "No folder is bound" is wrong in those cases (31660 re-review obs; the "changes are not tracked here" clause stays true). Distinguish consent-off if the admission data allows; also restore the remediation half of Change Review's copy (bind/enable path)
<!-- AC:END -->
