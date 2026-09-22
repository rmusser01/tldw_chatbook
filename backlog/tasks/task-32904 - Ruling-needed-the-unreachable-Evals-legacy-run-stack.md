---
id: TASK-32904
title: "Ruling needed: the unreachable Evals legacy run stack"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-dead
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
About **7,569 lines** (the review's headline said ~9,000; see the correction below) -- `eval_runner`, `specialized_runners`, `dataset_validator`, `dataset_loader`,
`base_runner`, `ui_integration`, `_run_admitted_evaluation` -- with **no reachable entry point**:
`handle_start_evaluation` does not exist and `ABTestOrchestrator` is constructed nowhere. Three test
files keep it green.

This is a ruling, not a cleanup, for one reason: **it carries a `subprocess` code-execution sandbox**
governed by ADR-031. Unreachable code that can execute code is a different risk from unreachable code
that cannot -- it is attack surface that nobody is reviewing precisely because nobody thinks it runs.
Deleting it is probably right; confirming it is genuinely unreachable from every entry point, including
plugins and any server-driven path, is the work.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
## Two corrections to this task as first written

**1. The line count.** The headline "~9,000 lines" double-counts. The six named modules total **7,569**
lines; the review reached ~9,000 by including `eval_templates.py` (1,298) -- which a **sibling finding in the
same slice** states can never execute, because the package of the same name shadows it. That file has since
been deleted on its own merits (TASK-32914, with a repo-wide shadowed-module guard added alongside it), so
the remaining stack is ~7,569 lines plus `_run_admitted_evaluation`.

**2. The reason the review gave is wrong, and a naive delete will not compile.** The review said the stack is
unreachable because `handle_start_evaluation` does not exist and `ABTestOrchestrator` is constructed nowhere.
Both are literally true but neither was ever the live door. Traced on `origin/dev`:

- `app.py:775` imports `EvaluationOrchestrator`, and `app.py:10598` **constructs** it.
- `UI/Screens/evals_screen.py:461-462` consumes it -- reading **only** `.db`.
- `Backup_Recovery/runtime_maintenance.py:252,333` binds it as a lifecycle participant.

So the **class is live**. What is dead is its *execution path*: `run_evaluation` (`:407`) is the only route to
`_run_admitted_evaluation` (`:434`) and thence to `EvalRunner`, and its only callers are
`ab_testing.py:182,192` (via the never-constructed `ABTestOrchestrator`) and `eval_orchestrator.py:1178`
(inside `async def quick_eval` at `:1145`, which has **zero callers** anywhere in `tldw_chatbook/` or
`Tests/`). Plugin and dynamic reachability were checked and are clear: no hits in `MCP/`, `Agents/` or
`Tools/`, and the only dynamic dispatch onto the orchestrator fetches `.db`.

**3. The security framing should be toned down, not up.** I initially wrote that the sandbox therefore loads
into every running app. Measured, it does not -- `tldw_chatbook.Evals.specialized_runners` (which holds the
`subprocess.run` at `:375`) is **not** in `sys.modules` after the `app.py:775` import chain; it is imported
lazily inside a function on the dead path. `eval_runner.py` has no dangerous import-time side effects either.
So this is ~7,569 lines of dead weight containing a sandbox that nothing loads and nothing calls.

**Recommendation:** delete surgically -- `run_evaluation`, `_run_admitted_evaluation`, `quick_eval`,
`EvalRunner`, `specialized_runners`, `base_runner`, `dataset_validator`, `dataset_loader`, `ui_integration`,
`ABTestOrchestrator`, and the three test files that keep them green. **Keep** `EvaluationOrchestrator` and
`.db`. That removes the sandbox from the tree and drops 7 modules off the boot import graph. Treat it as
maintenance, not an urgent security fix -- the real risk is someone wiring a new caller to it by accident.

Full trace in `qa/tier2-code-review-2026-09-21/validation/LEAD-evals-reachability.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reachability is confirmed or refuted from every entry point, plugins included
- [ ] #2 If unreachable, it is deleted with its three test files and ADR-031 is updated to say so
- [ ] #3 If reachable, the path is documented and the sandbox is reviewed as live code
<!-- AC:END -->
