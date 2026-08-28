---
id: TASK-23029
title: >-
  Boot budgets are all within 2-4% of breach two days after being pinned
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
updated_date: '2026-08-28'
labels:
  - process
  - performance
  - startup
priority: high
---

## Description

Four boot budgets were pinned on 2026-08-25 "just above reality". Two days later every one is within
2-4% of breach, and at observed merge rates each breaches within a day or two of normal traffic.

| guard | budget | now | headroom |
|---|---|---|---|
| boot import weight | 660 modules | 657 | **3 (0.5%)** |
| `_ui_ready` census | 970 | ~950 | ~20 - family assertion already RED |
| boot CSS bytes | 860,000 | 842,236 | **17,764 (2.1%)** |
| pre-import payload | 500 / 380k LOC | 481 / 368,814 | **19 (3.8%) / 11,186 (2.9%)** |

This is the finding that outranks the individual costs. Three reviews in six days have each brought
these numbers down and each time they were consumed within days. The guards work - they caught every
regression in this review - but a budget with 0.5% headroom converts the next ordinary feature into a
red build, which trains people to raise the budget.

## Acceptance Criteria

- [x] A decision is recorded on whether these are budgets (raise deliberately, with review) or ratchets (never raise, fix the cause)
- [x] Whichever it is, the guard says so in its failure message, so the next person hitting it knows which move is legitimate
- [x] A breach names the specific edge or module that consumed the headroom, not just the total - tracing it currently takes an import tracer and an hour
- [x] Consider whether headroom itself should be reported per-PR, so consumption is visible before the breach

## Evidence

See the four rows above, all measured on `c6218918d1`. Separately: the guards forbidding
`Chat.trajectory_export` on the first-paint path were written 2026-08-25 and breached within ~24
hours by the current tip (TASK-23020), which touched neither guard file and routed around an explicit
in-code comment forbidding exactly that.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Plan

1. Read the four guards end to end; re-measure all four on the base (`b5eaa9cf64`).
2. Record the owner's decision -- RATCHET, never raise -- as an ADR with the consumption history,
   the exception path (owner-signed ledger rows), and a tightening convention.
3. Give every budget assertion a policy footer stating the three legitimate responses and that
   raising the constant is not one of them.
4. Make breaches name culprits: pin snapshots (module sets for the two censuses, per-source +
   per-segment bytes for CSS, per-route modules/LOC for the pre-importer) and print directional
   diffs against them in the failure message (the TASK-23028 house pattern).
5. Emit one stable headroom line per guard on PASS (print + UserWarning so it reaches the pytest
   warnings summary in default CI invocations).
6. A single deliberate snapshot writer (`scripts/update_boot_budget_snapshots.py`) that refuses to
   bless over-budget states; guards never write.
7. Prove every new failure path with a mutant; unit-test the message plumbing at unit speed.

## Implementation Notes

**Decision recorded as ADR-097** (`backlog/decisions/097-boot-budget-ratchets.md`): the four
budgets are ratchets -- the constants never rise. Responses to a breach: (a) defer, (b) shed
elsewhere in the same PR, (c) explicit owner exception recorded in the ADR's append-only ledger
(a raise without a ledger row should be rejected in review). Tightening convention: a PR that
drops a measured value by more than the guard's standard slack lowers the limit to
measured + slack (slacks tabulated in the ADR). No automatic tightener -- the headroom lines make
the opportunity visible.

**Mechanics.** Shared helper `Tests/Performance/boot_budget_ratchet.py` (policy footer, snapshot
IO, directional diff formatters, headroom emitter), loaded via `Tests/Performance/conftest.py`
fixture. Snapshots in `Tests/Performance/boot_budget_snapshots/` (module lists for
import-weight/ui-ready; per-source + per-segment bytes for CSS -- the generated sheets' `/* =====
MODULE|WIDGET: ... ===== */` markers give 203 attributable segments; per-route module lists + LOC
for the pre-importer). Written ONLY by `scripts/update_boot_budget_snapshots.py`
(`--only import-weight|ui-ready|css|preimport`, `--force`); it refuses to pin an over-budget
measurement. Headroom lines are printed AND warned (`UserWarning`), so they appear in the pytest
warnings summary of a default CI run. The two census snapshots are diagnostic (breach diff +
drift marker), not hard equality pins: the ui-ready census wobbles +/-1 run-to-run and the
import closure can vary across installs, so equality would flake where TASK-23028's static AST
census cannot.

**Every new failure path mutant-proven** (messages captured in the session report): a synthetic
`tldw_chatbook.zz_ratchet_mutant_probe` planted in `app.py` was named `+` by the import guard; a
withheld snapshot line made the ui-ready breach name `+ tldw_chatbook.app`; a skewed CSS segment
row printed `_agentic_terminal.tcss: 260,217 -> 270,217 (+10,000)`; a skewed pre-import route
printed `library: 130,000 -> 137,494 (+7,494)` plus the dropped module by name. All mutants
restored byte-identically (md5-verified). `test_boot_budget_ratchet_messages.py` keeps the
plumbing honest at unit speed (policy wording, stable line format, diff directionality, snapshot
anti-vacuity/consistency, guards-never-write, ADR reference not dangling).

**Found while re-measuring (2026-08-28, dev `b5eaa9cf64`): the import-weight ratchet is ALREADY
BREACHED on pristine dev -- 666/660.** Not raised, per the ratchet. The new breach message names
all 17 added modules (vs the last in-budget set at `c6218918d1`, deliberately pinned as the
snapshot); an import-parent trace attributes them to three edge families
(`chat_persistence_service` module-scope imports ~12, `app.py -> console_raw_cli` 3,
`console_runtime -> thinking_blocks` 1). Repayment filed as **task-23112**; recorded in ADR-097
as "Standing breach at adoption". The other three guards are green but tighter than the review's
numbers: ui-ready 963/970, CSS 854,720/860,000, pre-import 488/500 + 374,697/380,000.

**Files:** `backlog/decisions/097-boot-budget-ratchets.md`,
`Tests/Performance/boot_budget_ratchet.py`, `Tests/Performance/conftest.py`,
`Tests/Performance/boot_budget_snapshots/*`,
`Tests/Performance/test_boot_budget_ratchet_messages.py`,
`scripts/update_boot_budget_snapshots.py`, edits to the four guard files,
`backlog/tasks/task-23112 - ...md`.
