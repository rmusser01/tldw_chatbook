# TASK-33007.4 fix round 1: the evidence behind the Task 4 report (2026-10-04)

The Task 4 review could not check seven claims from the diff, because no logs were
attached. Each claim was re-run on 2026-10-04; these files hold the output.

All test runs used the venv's Python with `PYTHONPATH` set to the tree under test, so the
venv's editable install (which points at the main checkout) was not measured. They ran
under `env -i` with `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` in a
scratch directory and the null keyring. Base `401fc2e1fe` and head `769dff7edc` were
detached worktrees, removed afterwards. Scratch paths read `<scratch>` and the worktree
reads `<worktree>`. The runner, mutation, probe and loopback scripts were scratch-only and
are not committed.

| File | Finding | What it shows |
|---|---|---|
| `covering-parity.txt` | 1 | 134 covering files, both sides at once, `-n 6 --dist loadfile`: counts, every head-only and base-only failure name, and each head-only name re-run alone twice per side |
| `architecture-parity.txt` | 1 | `Tests/Architecture` + `test_component_pattern_governance.py`, serial, both sides: identical failure names and identical R22 offender lists |
| `preflight-769dff7edc.txt` | 2 | `PYTHON=<venv> ./scripts/preflight.sh` at a clean `769dff7edc`, unpiped, with its exit code |
| `budgets.txt` | 3 | ui-ready census, pre-import payload and boot CSS on both sides; the line counts |
| `red-first-and-negative-controls.txt` | 4 | the new tests against the unchanged production code (6 of 6 red), and the four negative controls, each red on its own case only, each restored and checked clean |
| `warnings.txt` | 5 | raw output of the new file, and of every added or rewritten test on both sides: no warnings summary anywhere |
| `live-rerun.txt` | 6 | the README procedure re-run at `769dff7edc`: all five captures byte-identical (`.txt` and `.ansi.txt`); the real profile unwritten |
| `click-trace.txt` | 7 | why a click on "config key" needs the focus early-return, traced event by event |
