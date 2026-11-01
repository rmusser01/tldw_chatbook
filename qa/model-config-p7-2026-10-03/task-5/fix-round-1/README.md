# TASK-33007.5 fix round 1: the evidence behind the Task 5 report (2026-10-04)

The Task 5 review found that the committed captures predated the commit (finding 1). It also
listed claims it could not check from the diff (findings 2-8). Each one was re-run on 2026-10-04
at the Task 5 commit `88e08f2f74`, against its base `337dd68692`. These files hold the output.

All test runs used the venv's Python with `PYTHONPATH` set to the tree under test, so the venv's
editable install (which points at the main checkout) was not measured. They ran under `env -i`,
with `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` in a scratch directory and
the null keyring. Base, head and dev `5e0341d1ec` were detached worktrees, removed afterwards.
Scratch paths read `<scratch>` or `<tree>`, and the Task 5 worktree reads `<worktree>`. The
runner, mutation, AST and live-driver scripts lived only in scratch and are not committed.

| File | Finding | What it shows |
|---|---|---|
| `live-rerun.txt` | 1, 8 | the `../README.md` procedure re-run at `88e08f2f74`: 01, 02, 03, 05 and 06 byte-identical; 04 differs only by the Max tokens placeholder and replaces the stale one; the real profile unwritten |
| `rebase-verification.txt` | 2 | the rebase's commit map; an AST check that T1's move is still a pure move of dev's compose, TASK-34201's "Sign in with" included; 34201's 17 tests green on dev and head; the owner-pending 6-press Anthropic+key pin |
| `red-first-and-negative-controls.txt` | 3 | the new tests against the unchanged base code (9 of 10 red, the AC#7 save a regression guard); two negative controls, each red only on its own case and each restored clean |
| `covering-parity.txt` | 4 | the covering files (the Task 5 run's 59 plus an independent grep), both sides at once: counts, every head-only and base-only failure name, each head-only name re-run alone twice per side |
| `architecture-parity.txt` | 4 | `Tests/Architecture` + `test_component_pattern_governance.py`, serial, both sides: identical failure names and R22 offender lists |
| `budgets.txt` | 5 | ui-ready census, pre-import payload and boot CSS on base, head and dev; the limits in each tree; line counts |
| `preflight-88e08f2f74.txt` | 6 | `PYTHON=<venv> ./scripts/preflight.sh` at a clean `88e08f2f74`, unpiped, with its exit code |

Finding 7: Task 5 ran before TASK-33007.6 and TASK-33007.9, against R13's order. Its AC#9 holds at
the commit (`live-rerun.txt`, captures 01 and 05). The later tasks own the re-check: TASK-33007.6
gained AC#13 and TASK-33007.9 gained AC#5. Both say that live 211x44 captures for Anthropic and
llama.cpp must still show Connect through Model defaults without scrolling. Both ACs are in the
task files and in the plan.
