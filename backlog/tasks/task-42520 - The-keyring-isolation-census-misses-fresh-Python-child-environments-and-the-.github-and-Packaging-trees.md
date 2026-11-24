---
id: TASK-42520
title: "The keyring-isolation census misses fresh Python-child environments and the .github and Packaging trees"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 17:21
updated_date: 2026-11-24 19:29
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

**AC amendment (2026-11-24, review of `fu/r25-kcgaps`), made before any further code.**
- AC #1 now also requires HOME. `anthropic_subscription._keychain_credential_raw` is gated only by `_home_is_account_home()`. A child started with no HOME resolves `Path.home()` from the password database, counts as the account home, and would run `/usr/bin/security` if it reached `read_claude_code_credential`. The keyring pin plays no part in that path, so a pin alone does not close it.
- AC #7 was added: the review found Python children the census did not recognise, and spawns it cannot judge passed silently.
- AC #8 was added: `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write` failed instead of xfailing when turn one was slow (its TASK-33372 symptom), which made this branch's soak red.

Out of scope: `pytest --noconftest` children, and the opt-in CI native-credential lane, which uses the real backend only on GitHub-hosted runners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census flags any non-copy mapping passed as `env=` to a call that starts Python and lacks the null pin or HOME, whatever other keys it sets; parametrized controls show a fresh environment flagged without either, a pinned one with HOME and a copy passing
- [x] #2 The census is red on the `test_console_three_turn_profile.py:3999` site before its pin and green after, and every other site the widened census finds is pinned or exempted with a reason
- [x] #3 The census scans `.github/` and `Packaging/`, every new finding is triaged, and `Docs/` QA harnesses are left out with a comment that names `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/reproduce.sh`
- [x] #4 The census stays above its floors, re-measured on this tree, and an empty spawn scan cannot pass
- [x] #5 An in-process null pin in a `-c` child source is recognised, with controls in both directions, or the reason it is not is recorded
- [x] #6 TASK-42479's notes carry a dated correction of the "child mounts `TldwCli`" claim
- [x] #7 The census recognises the Python children the review found (an interpreter held in a variable, argv returned by a same-module helper, a starred first element, `list(...)` of an argv, an `sh -c` command that execs `sys.executable`), with one control per shape, and fails when the number of spawns it cannot judge grows past its measured count
- [x] #8 The TASK-33372 xfail in `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write` also accepts the turn-one wait's timeout and stays strict on every other failure
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

**G1: fresh Python-child environments (`Tests/test_keyring_isolation.py`).** Described as it stands after the review round; what the review changed is marked.
- **Which calls count.** The census's per-scope mapping tracker (`_profile_moves`, now `_scan`) also finds calls that pass `env=` and start Python, and judges the mapping they pass. The argv is read as a list, a name bound to one, `list(...)`/`tuple(...)` of one, a same-module function's returned list, or a starred first element that resolves to one (the last three are new in review). It starts Python when its first element is `sys.executable` (also through `str`, `os.fspath`, `Path(...).resolve()`), a `python*` string, or (new in review) a variable whose name contains `python` or is `interpreter`. It also counts with `-m` among its leading options, or (new in review) as `sh`/`bash -c` with an f-string that interpolates `sys.executable`. `asyncio.to_thread(subprocess.run, ...)` and `create_subprocess_exec(sys.executable, ...)` count too. An argv held in a parameter is not recognised (the generic runners `run_child_with_watchdog` and task598's `supervise`); the docstring says so.
- **Where a mapping came from.** Fresh: a literal, a `dict(...)` of literals, or a comprehension. A copy: `{**os.environ}`, `dict(os.environ)`, `os.environ.copy()`, `os.environ` or `env=None`. Unknown: a parameter, another module's helper, a method's result, a closure's variable, or a name bound by tuple unpacking. Or a same-module function's result, judged by its `return`s. `{**base}` carries the base's provenance and pin and is read before the name is rebound, so `e = {**e, ...}` keeps what `e` was. New in review: a comprehension that filters `os.environ.items()` by key, or copies names from `os.environ[n]` (in a comprehension or a `for` loop), carries HOME and the pin when its filter provably keeps them. `keeps` evaluates only comparisons, `and`/`or`/`not`, string constants or module constants holding them, the key's string methods and `... in os.environ`; anything else counts as dropping the key. That is how `test_workspace_tool_executor.py::_isolated_harness_environment`, which copies HOME and the pin by name from the pytest environment, now passes without an edit.
- **Verdict (AC #1 amended in review).** A fresh mapping must set both the null pin and HOME. The pin alone does not close the Keychain path: `_keychain_credential_raw` is gated only by `_home_is_account_home()`, and with no HOME the child's `Path.home()` comes from the password database. A copy carries both; a `pop`/`del` of either undoes that. Any other backend value fails. An in-process pin (G5) stands for the env pin, never for HOME. An unknown mapping is not judged. Copies pass in every tree, not only pytest files, since a copy starts the child with its parent's own backend and HOME, as passing no `env=` does.
- **Controls:** 49 parametrized cases in `test_the_census_reads_each_recognised_python_child_and_its_environment` (renamed in review). Run against the first round's scanner (commit b2dc98a8c6), 12 of the 45 then present were red: pin without HOME, a copy with HOME deleted, two key filters that keep both, the seven new argv shapes, and an in-process pin without HOME. The 4 copy-by-name cases came after. The rest were shown with temporary mutations, each reverted:
  - deleting the `to_thread` branch turned its new control red;
  - a filter evaluator that keeps everything turned 3 red; one that keeps nothing turned 2 red;
  - dropping the pin requirement turned 8 red; dropping the HOME requirement turned 3 red;
  - treating any `sh -c` as Python turned 1 red; ignoring interpreter variables turned 3 red.
- **Spawns the census cannot read** are capped (new in review): `_MAX_UNJUDGED_SPAWNS = 26`, the count measured on 2026-11-24. With the ceiling at 0 the test went red and listed all 26. I spot-checked the builders behind them. Most are a parameter or another module's builder that sets HOME and the pin (`_sanitized_build_env`, `_private_environment`, the roundtrip test's `environment.update(HOME=..., PYTHON_KEYRING_BACKEND=...)`, the task598 and native-credential builders) or a copy. The rest are the product restore launcher's environment, the native-credential lane (out of scope) and a Windows `vcvars` capture in `Packaging/windows/portable/wezterm/build.py`.
- **Red before the pins (first round):** the spawn census named 15 sites, including `Tests/Performance/test_console_three_turn_profile.py:3999 test_target_imports_resolve_only_below_selected_root`; green after the pins below.
- **Review round, red before the fixes:** with the HOME rule and the new argv shapes, the census named 8 sites: `:3999`, `Tests/Canvas/test_runtime_assets.py` (2), `Tests/Library/test_rag_answer_first_query_latency.py::_run`, `Tests/CI/test_plugin_qualification_workflow.py:86`, and `Tests/Tools/test_workspace_tool_executor.py` (3, `str(isolated_runtime_python)` children). Green after these fixes:
  - `:3999` and both `test_runtime_assets` children now get `HOME=str(tmp_path)`. They were the only fresh literals with no HOME.
  - The three `test_workspace_tool_executor` children needed no edit: their builder copies HOME and the pin by name, which the census now reads.
  - Two exemptions in the new `_SPAWN_EXEMPT` (stale pairs fail):
    - `test_rag_answer_first_query_latency.py::_run`: pinned; it drops HOME on purpose so the child can find the real Hugging Face cache, and both child sources (`_SEED`, `_MEASURE`) set HOME and the XDG paths in-process before any app import.
    - `test_plugin_qualification_workflow.py`: its allowlist is the workflow's own, read at run time with `ast.literal_eval`, and it holds HOME (`plugin-qualification.yml:112`); the test pins the backend.
- **Pinned in the first round** (`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` where the environment is built; review correction of the exposure):
  - Three children really dropped the pin. The two `{PATH, PYTHONPATH}` literals in `test_runtime_assets.py` (also HOME-less) and `test_worker_import_closure.py::_probe_environment` (its `PYTHON*` filter drops the pin; it keeps HOME). `:3999` dropped it too, also with no HOME.
  - The other 11 were `os.environ.items()` denylists that kept the conftest's pin. They were pinned explicitly because the census then could not evaluate filters; the pins stay, being explicit:
    - `Tests/CI/test_pytest_root_collection_skips_docs.py`
    - `Tests/CI/test_pytest_timeout_rearm.py`
    - `Tests/Chat/test_sensitive_llm_logging.py` (2)
    - `Tests/Image_Generation/test_comfyui_workflow_distribution.py`
    - `Tests/Library/test_rag_answer_first_query_latency.py::_run`
    - `Tests/Utils/test_startup_splash_policy.py` (4)
    - `Tests/test_real_profile_guard.py`

**G2: `.github/` and `Packaging/`.**
- Both are in `_CENSUS_TREES`. A comment names the `Docs/` exclusion: QA harnesses there are archived, and `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/reproduce.sh` moves HOME without the pin (its `env HOME=...` line launches the app), so re-running it by hand could hit the TASK-32344 authorization hang.
- New findings: two HOME moves, both unpinned: `.github/scripts/task598_external_parakeet_evidence.py::_isolated_environment` (its child imports the transcription service and `TldwCli`) and `.github/scripts/task602_platform_smoke.py::_isolate_process` (moves HOME in-process, then imports app modules). Both now pin; `_isolate_process` restores the key with the others. Red-first env-capture tests in the laned census file, host backend set to `fail`, no child run: `test_the_task598_evidence_child_environment_pins_the_null_backend`, `test_the_task602_smoke_isolation_pins_the_null_backend_and_restores_it`. Red: `assert 'keyring.backends.fail.Keyring' == 'keyring.backends.null.Keyring'` (2 nodes), and the HOME census named both sites; green after. The two trees' Python spawns (one in each) pass or are unknown (`Packaging/windows/portable/wezterm/build.py` passes a captured `vcvars` environment, a copy of the build process's).
- **Floors re-measured** on this tree: 179 profile moves (176 in the first three trees, the same count the base census code gives on this tree; 2 in `.github/`; 1 in `Packaging/`) and, after the review round, 294 recognised Python spawns (290 in `Tests/`, 2 in `scripts/`, 1 each in `.github/` and `Packaging/`; 272 in the first round). The move floor stays 120; the new spawn floor is 200. With `_CENSUS_TREES` emptied by a temporary edit (reverted), both census tests failed: `the census saw only 0 profile moves` and `the census saw only 0 Python spawns`.

**G5: in-process pins.**
- A `-c` child source is read when it is a constant, a module or scope constant, `textwrap.dedent(...)` of one, or an f-string (fields read as `None`), after any leading options (`-I -B -c`). It is pinned when its top level calls `keyring.set_keyring(<null Keyring>())`, the class bound from `keyring.backends.null` and not rebound, before any top-level `tldw_chatbook` import. Nine controls: pinned constant, the same without HOME (fails: the pin never stands for HOME), pin after the app import, macOS backend, rebound name, options before `-c`, f-string, dedent, and a script path (not read). Mutating `_pins_in_process` to always return False turned 4 red; to accept any source turned 3 red; reverted.
- It changes no verdict in today's tree: no child that pins in-process is handed a fresh environment. For the terminal child (an in-memory probe with its env pin removed), the spawn check still passes through the in-process pin, but the HOME check flags `_terminal_child_environment`. The HOME check judges a mapping where it is built, in a helper, and the child source is passed at a different call site; tying them together is cross-scope data flow, not a small change. So the env pin stays. It also covers the code that runs before the in-process pin (`Tests.network_guard`'s import and `install()`).
- TASK-42479's notes now carry a dated correction: the child never builds a `TldwCli` (`Host(App)` borrows `TldwCli.open_recovery_profile`), and its in-process pin dates from its first commit, b5251e9a6e (2026-09-16).

**TASK-33372 xfail (AC #8, review round).** `test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write` failed instead of xfailing when turn one was slow. The harness's 10 s turn-one wait (`run_console_three_turn_profile.py:5620-5625`) catches the timeout `wait_until` raises and re-raises it as "turn-one diagnostics: ...", while the xfail matcher accepted only "mounted sample did not settle before timeout". The reviewer measured base failing 2 of 5 and this branch 3 of 5 at load 14-54, all with that message. My first-round failures (4, all "turn-one diagnostics: gateway_calls=0 ... run_status=streaming") were the same symptom, which my 23 base runs at load 8-18 did not reach. The matcher now also accepts an error whose `__cause__` is that timeout, and stays strict otherwise. I forced the timeout with a temporary `wait_until(timeout=0.0)` default, reverted after:
- with the old matcher the node failed (`turn-one diagnostics`);
- with the new one it xfailed (reported as skipped by the private-profile wrapper);
- with the forced default plus `wait_until` raising "some other harness failure", the new matcher still failed.

**First-round checks** (machine shared with other agents; loads are 1-minute averages).
- `Tests/test_keyring_isolation.py` plus the 10 pinned test files and the tests of the two `.github` scripts (`Tests/CI/test_task598_external_parakeet_evidence.py`, `Tests/STT/test_task602_platform_smoke.py`, `Tests/CI/test_task602_platform_evidence.py`), `-n 4`: one run, then 5 more. Every run had 1096 passed. The census file passed 56 of 56 in all 6. Runs 2 and 3 of the 5 (17:26-17:30 UTC, load 17-28) also failed `test_console_three_turn_profile.py::test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write`, the TASK-33372 harness node (`turn-one diagnostics: gateway_calls=0 ... run_status=streaming`, a message its xfail branch does not match). It failed twice more when run alone at 17:37-17:40 (load 17-21). Its code path is identical to base: that file's only change is the `:3999` environment, in another test. It was **not** reproduced at base: 23 runs with the base file content (alone, alternating with this branch, 6 in parallel, and the whole file under a `Tests/CI` load) all xfailed, at load 8-18. The 20 runs of this branch in the same conditions also all xfailed. TASK-42479 recorded the node red at base at load 57-97.
- One whole-file run of `test_console_three_turn_profile.py` under an added `Tests/CI` load (load rising to 34) failed `test_prepare_workspace_runtime_cleans_resources_on_construction_failure[readiness]` once. The output was filtered, so the message was not kept. Its 3 cases passed 6 of 6 alone at load 38-42, and in the other 8 whole-file runs; it builds the workspace runtime in-process and touches nothing this task changed.
- `Tests/CI` `-n 4`: 1948 passed, 2 skipped (69 s, load ~16).
- `Tests/Architecture` serial: 1507 passed, 5 skipped (15 min 9 s, load 33 falling to 10).
- Architecture guards step through `r19-step.sh` (`-p no:xdist --timeout=180`, 53 targets): 866 passed, 1 skipped (the Windows-only network-guard case), rc 0, 8 min 35 s at load 34 falling to 10. The spawn census took 6.4 s there. The census file alone with the step's flags: 56 passed in 16.8 s at load ~30, the shared scan 10.8 s (TASK-42479 measured 4.2 s for the old scan at load 14-20).
- `preflight.sh` rc 0; `ruff check tldw_chatbook --select F` clean; `ruff_gate.py` rc 0.
- No child was run from an unpinned environment: the base content of the three-turn file was run only with its `:3999` test deselected or as the single TASK-33372 node.

**Review-round checks** (2026-11-24, after the fixes above; loads are 1-minute averages).
- The census file, the 10 pinned test files and the tests of the two `.github` scripts, `-n 4`, 5 runs: 1117 passed, 14 skipped in each, at load 8-15. The TASK-33372 node xfailed in all 5. The census file has 77 tests now (49 spawn controls, 18 HOME-move controls, the two census tests and 8 others).
- `Tests/CI` `-n 4`: 1948 passed, 2 skipped (57 s, load ~11).
- Architecture guards step through `r19-step.sh` (`-p no:xdist --timeout=180`, 53 targets), run alone: 887 passed, 1 skipped (the Windows-only case), rc 0, 6 min 47 s at load 10.6 falling to 7.6. The spawn census took 6.9 s there.
- `Tests/Architecture` serial: 1507 passed, 5 skipped (14 min 27 s, load 7-9).
- `preflight.sh` rc 0; `ruff check tldw_chatbook --select F` clean; `ruff_gate.py` rc 0. `ruff --select F` on the touched test files finds only the 15 findings `test_console_three_turn_profile.py` already has at base.

**Known, accepted gap: `pytest --noconftest` children** (out of scope). The brief said the only in-repo use is collect-only; there are more. `Tests/CI/test_pytest_root_collection_skips_docs.py` (collect-only) and `Tests/CI/test_pytest_timeout_rearm.py` (runs a hanging test file that imports no app code) are now pinned explicitly. Also: the `PR lane test targets resolve without overlaps` CI step (two contract tests, on GitHub runners), `Tests/RAG/_metadata_import_probe.py` (an in-process `pytest.main(["--noconftest", ...])` inside its own child), and `Tests/Backup_Recovery/run_platform_product.py`'s phases (the opt-in native-credential lane, also out of scope).

**Lessons.** Added a dated update to the TASK-42477 Keychain entry in `backlog/docs/lessons-testing-evidence.md`: the key-based census missed key-less fresh environments, a pin without HOME does not close the Keychain path, and a `TldwCli` import is not a mounted app. The review round corrected its exposure count (3 children really dropped the pin, not 15).

Files: `Tests/test_keyring_isolation.py`, the 10 pinned test files above (`test_console_three_turn_profile.py` also carries the TASK-33372 matcher fix), `.github/scripts/task598_external_parakeet_evidence.py`, `.github/scripts/task602_platform_smoke.py`, TASK-42479's task file, `backlog/docs/lessons-testing-evidence.md`. No new test file, so no lane or contract change: the new tests are in `Tests/test_keyring_isolation.py`, already a target of the Architecture guards step.
<!-- SECTION:NOTES:END -->
