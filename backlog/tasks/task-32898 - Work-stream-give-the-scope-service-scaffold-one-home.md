---
id: TASK-32898
title: "Work stream: give the scope-service scaffold one home"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-helpers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-scoped restatement of TASK-32808.6, whose stated scope is wrong in five measurable ways (see the
scope-corrections task). Six helpers -- `_maybe_await`, `_enforce_policy`, `_require_client`,
`_normalize_mode`, `_identity`, `_dump` -- are re-rolled across **47** services, ~1,776 lines out and
roughly none in.

This needs an ADR before a PR, because a base class cannot be written until four things are settled, and
two of them are currently documented in contradictory ways:
- where the base lives;
- the dispatch predicate **and its fail-direction** (two documented rules disagree today);
- enum-vs-string mode **and the `mode=None` default** -- 25 services default SERVER, 19 default LOCAL,
  with four `Interop`/`server_*` twin pairs inverted, so a base **cannot** hard-code it;
- the three runtime `TLDWAPIClient` importers.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.

## ADR-177 written 2026-09-22 — and it changes the answer

The ADR this task was blocked on is now drafted (`backlog/decisions/177-scope-service-helper-consolidation.md`).
Measuring the tree to write it produced two facts that invalidate the "one base class" framing:

**No service uses the whole scaffold.** Across the 47 files named `*scope_service*.py`:
**0** define all six helpers, **0** define five, 6 define four, **38 define three**, 3 define fewer. There is
no common scaffold to lift — it is a family of overlapping partial duplications, and a base class offering
all six would hand every subclass three methods it never asked for.

**The naming convention hides two thirds of it.** 47 files are named `*scope_service*.py`, but **149 files
carry at least one of the six helpers** — so **102 are outside the convention entirely** (40 Interop, 62
core). A scope-service base class would leave the majority of the duplication in place while looking like it
had addressed it. That is precisely the failure this review documented elsewhere: consolidation tasks
closing without the census moving.

Confirmed exactly as filed: **279 definitions / 1,776 lines**, and the 47 split **31 Interop / 16 core**.
Refined: `_normalize_mode` bodies mention `SERVER` only in **25**, `LOCAL` only in **16**, and neither
literal in **5** (the earlier "25 / 19" was close but not the shape of the split).

**Decision: consolidate per helper into `runtime_policy/`, in four tranches by risk, with no base class.**
`_maybe_await` (65) and `_identity` (31) first — pure, no policy, no fail-direction, 96 of the 279
definitions and near-zero risk. `_normalize_mode` and `_enforce_policy` **last**, and only after a separate
decision settles the mode default and the dispatch fail-direction, because a wrong fail-direction there is a
security regression rather than a style one.

Each tranche carries a census row and a shrink-only guard, because this repo's duplication census has not
moved across two reviews and ten fix-stream PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR settles base location, dispatch predicate and fail-direction, mode type and default, and the runtime-import question
- [ ] #2 The mode default is explicit per service, never inherited from the base
- [ ] #3 Only after the ADR, one PR moves the six helpers to the agreed home
- [ ] #4 A census row and guard exist, or the task records why the cluster does not warrant one
<!-- AC:END -->
