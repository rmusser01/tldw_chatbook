---
id: TASK-42520
title: "The keyring-isolation census misses fresh Python-child environments and the .github and Packaging trees"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 17:21
updated_date: 2026-11-24 18:18
labels:
- follow-up
- tests
- hardening
priority: medium
references:
- Tests/test_keyring_isolation.py
- Tests/Performance/test_console_three_turn_profile.py
- Tests/Backup_Recovery/test_recovery_profile_terminal.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by a read-only audit of the suite's keyring isolation (2026-11-24, owner asked for the Keychain exposure to be addressed). The audit found nothing written to the real Keychain; these gaps are about keeping it that way. Checked on dev 9fa2bf1b1d.

**Why.** `Tests/test_keyring_isolation.py` (TASK-42479/42480, PR #3750) holds a census that fails when an environment moves HOME together with a profile key but does not pin `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`. It has three gaps:

- **G1. A fresh environment with no HOME is not seen.** The census only flags mappings that set HOME plus a profile key. `Tests/Performance/test_console_three_turn_profile.py:3999` spawns `sys.executable` with only `PATH` and `PYTHONDONTWRITEBYTECODE`, so the child drops the conftest's pin and, with no HOME, reaches the real home (`Path.home()` falls back to the password database). It only imports modules, so no keyring call happens today; that is luck.
- **G2. `.github/` and `Packaging/` are not scanned.** Both hold Python spawn sites and HOME moves.
- **G5. In-process pins are not read.** A `-c` child that calls `keyring.set_keyring(<null Keyring>())` before any app import is pinned, but the census cannot see it. TASK-42479's notes also say the `test_recovery_profile_terminal.py` child "mounts `TldwCli`" and so reached the Keychain. In fact that child never builds a `TldwCli` (its `Host(App)` borrows `TldwCli.open_recovery_profile`) and has pinned the null backend in-process since it was written.

Out of scope: `pytest --noconftest` children, and the opt-in CI native-credential lane, which uses the real backend only on GitHub-hosted runners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census flags any non-copy mapping passed as `env=` to a call that starts Python and lacks the null pin, whatever keys it sets; parametrized controls show a fresh environment flagged, a pinned one and a copy passing
- [x] #2 The census is red on the `test_console_three_turn_profile.py:3999` site before its pin and green after, and every other site the widened census finds is pinned or exempted with a reason
- [x] #3 The census scans `.github/` and `Packaging/`, every new finding is triaged, and `Docs/` QA harnesses are left out with a comment that names `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/reproduce.sh`
- [x] #4 The census stays above its floors, re-measured on this tree, and an empty spawn scan cannot pass
- [x] #5 An in-process null pin in a `-c` child source is recognised, with controls in both directions, or the reason it is not is recorded
- [x] #6 TASK-42479's notes carry a dated correction of the "child mounts `TldwCli`" claim
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the census, TASK-42479/42480 and the named sites. Survey every call that passes `env=` in the census trees plus `.github/` and `Packaging/` (pure AST) to see which argv shapes start Python and which `env=` shapes occur.
2. G1: extend the census's per-scope mapping tracker so each mapping records whether it is fresh, a copy, unknown (a parameter or another module's result) or a same-module helper's result. Detect calls whose argv starts Python (`sys.executable` and its wrappers, a `python*` string, a name bound to either, or `-m` among the leading options). Judge each `env=` mapping: pinned, copy (passes), fresh (flagged), unknown (not judged; documented). Follow same-module helpers through their `return`s. Add parametrized controls, then run the repo census red: it must name `:3999`.
3. G5: read a `-c` source (constant, module constant, `textwrap.dedent`, f-string) and accept a top-level `keyring.set_keyring(<null Keyring>())` before any `tldw_chatbook` import. Controls in both directions, shown red under temporary mutations. Check what it changes for the terminal child; if the HOME check still needs the env pin, keep it and say why.
4. G2: add `.github` and `Packaging` to the trees, comment the `Docs/` exclusion, triage the new findings (pin, with red-first env-capture tests for the scripts in the laned census file).
5. Pin every flagged site; re-measure the floors; correct TASK-42479's notes with a dated line.
6. Checks: the census file and touched test files 5x at `-n 4`, `Tests/CI` `-n 4`, `Tests/Architecture` serial, preflight, ruff F, ruff gate, the Architecture guards step via `r19-step.sh`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The plan was written after a pure-AST survey of every `env=` call in the five trees and a first prototype of the scan, not before them.

**G1: fresh Python-child environments (`Tests/test_keyring_isolation.py`).**
- The census's per-scope mapping tracker (`_profile_moves`, now `_scan`) also finds calls that pass `env=` and start Python, and judges the mapping they pass. A call starts Python when its argv's first element is `sys.executable` (also through `str`, `os.fspath`, `Path(...).resolve()`, or a name bound to it in the scope or module), a `python*` string, or when `-m` is among its leading options; `asyncio.to_thread(subprocess.run, ...)` and `create_subprocess_exec(sys.executable, ...)` count too.
- Each mapping now records where it came from: fresh (a literal, a `dict(...)` of literals, a comprehension), a copy (`{**os.environ}`, `dict(os.environ)`, `os.environ.copy()`, `os.environ`, `env=None`), unknown (a parameter, another module's helper, a method's result), or a same-module function's result, judged by that function's `return`s. `{**base}` carries the base's provenance and pin, and is read before the name is rebound, so `e = {**e, ...}` keeps what `e` was.
- Verdict: a null pin passes and any other backend value fails; a `pop`/`del` fails a copy; a copy passes; a fresh mapping fails; an unknown one is not judged. Copies pass in every tree, not only in pytest files, because a copy starts the child with the parent's own backend, the same as passing no `env=` at all. A filtered copy built from `os.environ.items()` is fresh, as the HOME check already treats it.
- 28 parametrized controls in `test_the_census_reads_each_way_a_python_child_gets_its_environment`, 20 for G1 and 8 for G5. The G1 ones include the brief's `env={"PATH": p}` case (flagged), its pinned variant (passes), copies, a popped copy, a denylist, the argv shapes, `git` (not counted), and same-module and imported helpers. With the fresh-mapping verdict mutated to pass, 13 of them went red; reverted.
- Limits (in the docstring): the verdict is the mapping's state at the end of its scope, not at the call; parameters, other modules' helpers and closures are not followed. 24 of the 272 spawns are unknown today, nearly all parameters or helpers imported from another module.
- **Red before the pins:** `test_every_fresh_environment_handed_to_a_python_child_pins_the_null_backend` named 15 sites, including `Tests/Performance/test_console_three_turn_profile.py:3999 test_target_imports_resolve_only_below_selected_root`. **Green after** the pins below.
- **Pinned** (`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` added where the environment is built; no exemptions):
  - fresh literals with no HOME, which also reach the real home: `test_console_three_turn_profile.py:3999`, `Tests/Canvas/test_runtime_assets.py` (2, the vendor script's `--verify` and archive runs);
  - allowlists or denylists built from `os.environ.items()`: `Tests/Tools/test_worker_import_closure.py::_probe_environment` (it drops every `PYTHON*` key, so it dropped the pin too), `Tests/CI/test_pytest_root_collection_skips_docs.py`, `Tests/CI/test_pytest_timeout_rearm.py`, `Tests/Chat/test_sensitive_llm_logging.py` (2), `Tests/Image_Generation/test_comfyui_workflow_distribution.py`, `Tests/Library/test_rag_answer_first_query_latency.py::_run`, `Tests/Utils/test_startup_splash_policy.py` (4), `Tests/test_real_profile_guard.py`. The denylists kept the conftest's pin already; they are pinned explicitly rather than taught to the census, which would need an evaluator for each filter.

**G2: `.github/` and `Packaging/`.**
- Both are in `_CENSUS_TREES`. A comment names the `Docs/` exclusion: QA harnesses there are archived, and `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/reproduce.sh` moves HOME without the pin (its `env HOME=...` line launches the app), so re-running it by hand could hit the TASK-32344 authorization hang.
- New findings: two HOME moves, both unpinned: `.github/scripts/task598_external_parakeet_evidence.py::_isolated_environment` (its child imports the transcription service and `TldwCli`) and `.github/scripts/task602_platform_smoke.py::_isolate_process` (moves HOME in-process, then imports app modules). Both now pin; `_isolate_process` restores the key with the others. Red-first env-capture tests in the laned census file, host backend set to `fail`, no child run: `test_the_task598_evidence_child_environment_pins_the_null_backend`, `test_the_task602_smoke_isolation_pins_the_null_backend_and_restores_it`. Red: `assert 'keyring.backends.fail.Keyring' == 'keyring.backends.null.Keyring'` (2 nodes), and the HOME census named both sites; green after. The two trees' Python spawns (one in each) pass or are unknown (`Packaging/windows/portable/wezterm/build.py` passes a captured `vcvars` environment, a copy of the build process's).
- **Floors re-measured** on this tree: 179 profile moves (176 in the first three trees, the same count the base census code gives on this tree; 2 in `.github/`; 1 in `Packaging/`) and 272 Python spawns. The move floor stays 120; the new spawn floor is 200. With `_CENSUS_TREES` emptied by a temporary edit (reverted), both census tests failed: `the census saw only 0 profile moves` and `the census saw only 0 Python spawns`.

**G5: in-process pins.**
- A `-c` child source is read when it is a constant, a module or scope constant, `textwrap.dedent(...)` of one, or an f-string (fields read as `None`), after any leading options (`-I -B -c`). It is pinned when its top level calls `keyring.set_keyring(<null Keyring>())`, the class bound from `keyring.backends.null` and not rebound, before any top-level `tldw_chatbook` import. Eight controls: pinned constant, pin after the app import, macOS backend, rebound name, options before `-c`, f-string, dedent, and a script path (not read). Mutating `_pins_in_process` to always return False turned 4 red; to accept any source turned 3 red; reverted.
- It changes no verdict in today's tree: no child that pins in-process is handed a fresh environment. For the terminal child (an in-memory probe with its env pin removed), the spawn check still passes through the in-process pin, but the HOME check flags `_terminal_child_environment`. The HOME check judges a mapping where it is built, in a helper, and the child source is passed at a different call site; tying them together is cross-scope data flow, not a small change. So the env pin stays. It also covers the code that runs before the in-process pin (`Tests.network_guard`'s import and `install()`).
- TASK-42479's notes now carry a dated correction: the child never builds a `TldwCli` (`Host(App)` borrows `TldwCli.open_recovery_profile`), and its in-process pin dates from its first commit, b5251e9a6e (2026-09-16).

**Checks** (machine shared with other agents; loads are 1-minute averages).
- `Tests/test_keyring_isolation.py` plus the 10 pinned test files and the tests of the two `.github` scripts (`Tests/CI/test_task598_external_parakeet_evidence.py`, `Tests/STT/test_task602_platform_smoke.py`, `Tests/CI/test_task602_platform_evidence.py`), `-n 4`: one run, then 5 more. Every run had 1096 passed. The census file passed 56 of 56 in all 6. Runs 2 and 3 of the 5 (17:26-17:30 UTC, load 17-28) also failed `test_console_three_turn_profile.py::test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write`, the TASK-33372 harness node (`turn-one diagnostics: gateway_calls=0 ... run_status=streaming`, a message its xfail branch does not match). It failed twice more when run alone at 17:37-17:40 (load 17-21). Its code path is identical to base: that file's only change is the `:3999` environment, in another test. It was **not** reproduced at base: 23 runs with the base file content (alone, alternating with this branch, 6 in parallel, and the whole file under a `Tests/CI` load) all xfailed, at load 8-18. The 20 runs of this branch in the same conditions also all xfailed. TASK-42479 recorded the node red at base at load 57-97.
- One whole-file run of `test_console_three_turn_profile.py` under an added `Tests/CI` load (load rising to 34) failed `test_prepare_workspace_runtime_cleans_resources_on_construction_failure[readiness]` once. The output was filtered, so the message was not kept. Its 3 cases passed 6 of 6 alone at load 38-42, and in the other 8 whole-file runs; it builds the workspace runtime in-process and touches nothing this task changed.
- `Tests/CI` `-n 4`: 1948 passed, 2 skipped (69 s, load ~16).
- `Tests/Architecture` serial: 1507 passed, 5 skipped (15 min 9 s, load 33 falling to 10).
- Architecture guards step through `r19-step.sh` (`-p no:xdist --timeout=180`, 53 targets): 866 passed, 1 skipped (the Windows-only network-guard case), rc 0, 8 min 35 s at load 34 falling to 10. The spawn census took 6.4 s there. The census file alone with the step's flags: 56 passed in 16.8 s at load ~30, the shared scan 10.8 s (TASK-42479 measured 4.2 s for the old scan at load 14-20).
- `preflight.sh` rc 0; `ruff check tldw_chatbook --select F` clean; `ruff_gate.py` rc 0.
- No child was run from an unpinned environment: the base content of the three-turn file was run only with its `:3999` test deselected or as the single TASK-33372 node.

**Known, accepted gap: `pytest --noconftest` children** (out of scope). The brief said the only in-repo use is collect-only; there are more. `Tests/CI/test_pytest_root_collection_skips_docs.py` (collect-only) and `Tests/CI/test_pytest_timeout_rearm.py` (runs a hanging test file that imports no app code) are now pinned explicitly. Also: the `PR lane test targets resolve without overlaps` CI step (two contract tests, on GitHub runners), `Tests/RAG/_metadata_import_probe.py` (an in-process `pytest.main(["--noconftest", ...])` inside its own child), and `Tests/Backup_Recovery/run_platform_product.py`'s phases (the opt-in native-credential lane, also out of scope).

**Lessons.** Added a dated update to the TASK-42477 Keychain entry in `backlog/docs/lessons-testing-evidence.md`: the key-based census missed key-less fresh environments, and a `TldwCli` import is not a mounted app.

Files: `Tests/test_keyring_isolation.py`, the 10 pinned test files above, `.github/scripts/task598_external_parakeet_evidence.py`, `.github/scripts/task602_platform_smoke.py`, TASK-42479's task file, `backlog/docs/lessons-testing-evidence.md`. No new test file, so no lane or contract change: the new tests are in `Tests/test_keyring_isolation.py`, already a target of the Architecture guards step.
<!-- SECTION:NOTES:END -->
