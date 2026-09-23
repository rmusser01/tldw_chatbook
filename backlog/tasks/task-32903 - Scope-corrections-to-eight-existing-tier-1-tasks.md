---
id: TASK-32903
title: Scope corrections to eight existing tier-1 tasks
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tier 2 measured eight tier-1 tasks whose stated scope is wrong. These are **corrections to existing
tasks, not new work** -- apply them as edits to those task files so the next person to pick one up is not
working from a false premise. Filing them as fresh tasks would duplicate the work streams they belong to.

- **TASK-32808.5 (Done)** -- the helper it drove adoption toward is the **weakest of three** in the repo.
  Re-open as "harden the helper", not "adopt the helper". See TASK-32896.
- **TASK-32808.6** -- "the four helpers" is **six**; "~45 services" is **47**; "33 Interop + 12 core" is
  **31 + 16**; "~2,000 lines" is **1,776**; and "the fix exists in exactly one of the services" is wrong --
  **nine** services thread it, in five mutually incompatible shapes with two opposite fail-directions.
- **TASK-32808.8** -- its premise is that the per-store connection setup should move into the shared base.
  Tier 2 recommends **against** it: the PRAGMA divergences are documented per-store decisions and one
  explicitly warns against copying it. Note also that the often-repeated "all 9 overrides call
  `super()._get_connection()`" is **false** -- 7 do, and `Notifications/event_state_repository.py` and
  `DB/Library_Ingest_Jobs_DB.py` call `connect_private_sqlite` directly, so an auditor grepping for
  `super()._get_connection()` gets a false negative.
- **TASK-32802.1** -- AC#2 is keyed on "sites that currently call `rich.markup.escape`", which
  structurally **cannot** find the sites that call no escaper at all.
- **TASK-32803.5 (Done)** -- 101 writers still emit `+00:00` where ADR-173 mandates `Z`. See TASK-32897.
- **TASK-32805.5 / ADR-175** -- `tldw_api/` is a fourth strict-JSON boundary, in neither the adopter list
  nor the deliberate-exception list.
- **TASK-32806.8** -- scope names two files; three other image ingresses have no pixel cap.
- **TASK-32809.2** -- six unbudgeted modules are larger than budgeted ones, and the ratchet is **5-red**
  at this SHA, so 32809.1's re-pin is outstanding and these rows should land in the same commit.
- **TASK-32863** -- AC#1 is already satisfied on dev; AC#2 as written would **remove** Higgs's 300 s cap.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the listed task files carries the correction in its own body
- [ ] #2 TASK-32808.5 and TASK-32803.5 are re-opened or explicitly superseded
- [ ] #3 TASK-32808.8's recommend-against is recorded with its evidence
- [ ] #4 No new task duplicates work already owned by the corrected tasks
<!-- AC:END -->
