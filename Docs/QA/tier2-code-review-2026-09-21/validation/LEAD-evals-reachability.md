# Lead verification — the `Evals/` legacy run stack (input for the TASK-32904 ruling)

The review listed ~9,000 lines as unreachable and recommended a ruling rather than a deletion PR, because
the stack carries a `subprocess` code-execution sandbox. I traced it end to end on `origin/dev d0face3ebe`.

**The conclusion holds. The stated reason is wrong, and the correction changes what the fix must be.**

## What the review said

> Unreachable — `handle_start_evaluation` does not exist; `ABTestOrchestrator` is constructed nowhere.

Both halves are literally true (`handle_start_evaluation` appears nowhere in the tree; `ABTestOrchestrator`
is defined at `ab_testing.py:489` and constructed nowhere). But neither was ever the live door, so the
argument does not establish what it was used for.

## What is actually true

`EvaluationOrchestrator` **is live and loads at boot**:

| site | what it does |
|---|---|
| `tldw_chatbook/app.py:775` | `from .Evals.eval_orchestrator import EvaluationOrchestrator` — **the main entry point imports it** |
| `tldw_chatbook/app.py:10598` | `self.evaluation_orchestrator = EvaluationOrchestrator(...)` — **constructed at startup** |
| `UI/Screens/evals_screen.py:461-462` | live consumer — but reads **only** `.db` |
| `Backup_Recovery/runtime_maintenance.py:252,333` | live consumer — binds it as a lifecycle participant |

And `eval_orchestrator.py:28` does `from .eval_runner import EvalRunner, ...`, so **`eval_runner` and the
`subprocess` sandbox in `specialized_runners.py:375` are imported into every running app.**

What is genuinely dead is the **execution path**, not the module. `run_evaluation` (`:407`) is the only
route to `_run_admitted_evaluation` (`:434`) and thence to `EvalRunner`, and it has **no live caller**:

- `ab_testing.py:182,192` — reached only through `ABTestOrchestrator`, constructed nowhere.
- `eval_orchestrator.py:1178` — inside `async def quick_eval(...)` at `:1145`, which has **zero callers**
  anywhere in `tldw_chatbook/` or `Tests/`.

## Why the correction matters

1. **Deleting the package would break the app.** `app.py`, `evals_screen` and `runtime_maintenance` all
   depend on the orchestrator class. The review's framing invites a wholesale delete that does not compile.
2. **The security picture is NOT sharper — I checked, and my first reading of it was wrong.**
   I initially wrote that the sandbox loads into every running app. Measured, it does not:
   ```
   $ .venv/bin/python -c "import sys; from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator; \
       print('tldw_chatbook.Evals.specialized_runners' in sys.modules)"
   False
   ```
   The `app.py:775` import pulls in 8 `Evals` modules — `eval_orchestrator`, `eval_runner`,
   `concurrency_manager`, `config_loader`, `configuration_validator`, `eval_errors`, `task_loader` and the
   package — but **not** `specialized_runners`, which is where `subprocess.run` lives (`:375`). That module
   is imported lazily inside a function on the dead path. `eval_runner.py` also has no dangerous
   import-time side effects: module level is a docstring and a guarded `try: from datasets import ...`.

   So the sandbox is dead in the ordinary sense — not loaded, not reachable. The review's security framing
   was right and my "loaded at boot" reading was wrong. Recording it because a ruling made on my wrong
   version would have over-stated the urgency.
3. **The removal is surgical.** Keep `EvaluationOrchestrator` and `.db` and whatever `runtime_maintenance`
   binds. Remove `run_evaluation`, `_run_admitted_evaluation`, `quick_eval`, `EvalRunner`,
   `specialized_runners`, `base_runner`, `dataset_validator`, `dataset_loader`, `ui_integration`, and
   `ABTestOrchestrator` — then the `eval_runner` import at `:28` goes with them and the sandbox stops
   loading at boot. That is the actual prize.

## What does NOT change

The three S11 eval findings demoted on dead-path grounds **stay demoted** — they sit in the run stack,
whose execution path is confirmed dead by this trace too. The S11 validator's independent re-check agreed.
This correction is to the review's *reason* and to the *deletion strategy*, not to those severities.

## Plugin / dynamic reachability — checked, and closed

- `rg 'run_evaluation|EvaluationOrchestrator'` across `tldw_chatbook/MCP/`, `Agents/`, `Tools/` -> **no hits**.
- The only dynamic dispatch onto the orchestrator is `evals_screen.py:461-462`, and it fetches `.db` only.
- No loader imports `Evals` by string. `Evals/__init__.py` uses a PEP 562 lazy `__getattr__`, but `app.py:775`
  imports the submodule directly and so bypasses it.

Nothing left unverified on this question. The execution path is dead; the class is live.

## Recommendation to whoever rules

Delete surgically and the win is real but modest: `run_evaluation`, `_run_admitted_evaluation`, `quick_eval`,
`EvalRunner`, `specialized_runners`, `base_runner`, `dataset_validator`, `dataset_loader`, `ui_integration`,
`ABTestOrchestrator`, plus the three test files that keep them green. Keep `EvaluationOrchestrator` and `.db`.
That removes the sandbox from the tree and drops 7 modules off the boot import graph.

Do **not** frame this as an urgent security fix. It is ~9,000 lines of dead weight with a sandbox in it that
nothing loads and nothing calls — worth removing on maintenance grounds, and worth removing *before* someone
wires a new caller to it by accident, which is the actual risk.
