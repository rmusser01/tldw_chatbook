---
id: TASK-35961
title: >-
  Console Character display check: trim the merged check's residual loop-CPU and
  fallback costs
status: To Do
assignee: []
created_date: '2026-10-21 01:32'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.6 merged the Console Character display check's scope probe and reload into one finite callback, one handle and one ADR-125 helper. A few costs went up and were documented there. An unchanged check, the common one (about once a second during a reply), spends about 0.5-0.9 ms more UI-loop CPU than the original probe, because it qualifies the stock readers (_stock_character_refresh_readers) on the loop for every check. A first scope read that does not settle costs one merged helper plus the whole original probe-then-refresh route. A refresh dispatch customized during the read takes 5 finite callbacks against 4, and a final pair that keeps failing loads the groups 4 times against 3. Only the loop CPU is on the steady-state cadence; the rest are race, error or customization paths. A cheaper early exit for an unchanged check, and a fallback that reuses what the merged read already learned, would remove them without weakening reader provenance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An unchanged stock display check costs no more UI-loop CPU than the original probe route did, shown by a paired in-process measurement
- [ ] #2 A first scope read that does not settle starts no more helper processes than the original probe-then-refresh route
- [ ] #3 A customized, replaced or borrowed reader is still refused (the existing Character source- and entry-control tests stay green)
<!-- AC:END -->
