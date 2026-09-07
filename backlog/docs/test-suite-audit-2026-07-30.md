# Test-Suite Audit — 2026-07-30

**Trigger**: full `pytest` runs take 1+ hour; asked to find broken, useless, or inefficient tests.
**Method**: three-way static audit (structure/config, slowness patterns, broken/useless patterns) over `Tests/` at `origin/dev` `665ef1c01`, plus one instrumented serial baseline run (junit XML + `--durations=50`, artifact referenced in §8).
**Remediation program**: tasks **task-1450 … task-1465** (§7).

---

## 1. Executive summary

The suite is **11,571 test functions / 23,793 collected items / 401k LOC across 900 files**, and it runs **fully serially** — pytest-xdist is not installed, no CI job or local config uses `-n`. The hour is not caused by a few pathological tests; it is dominated by three systemic costs:

1. **~2,650 full Textual app mounts** (`Tests/UI` alone), 1,344 of which build the *real* `TldwCli` via `_build_test_app()`, each re-parsing the 14,616-line `css/tldw_cli_modular.tcss` bundle.
2. **Two full `gc.collect()` passes after every test** — an autouse fixture in `Tests/conftest.py:164-175` = ~23,000 full-heap collections per run (est. 10–40 min with torch/transformers imported).
3. **Zero parallelism and zero fixture reuse** — 568 fixtures, only 3 session-scoped; file-backed DBs with full schema+FTS5 DDL are rebuilt per test at 354 sites.

Also headline-worthy: **a default `pytest Tests` run currently cannot complete at all on dev, for two independent reasons.** (1) `Tests/Event_Handlers/test_worker_events_contract.py` imports `StreamDone`, which TASK-650 removed from the product module, and the collection error aborts the entire run (fix: task-1456). (2) Even past that, `test_pyaudio_recording_flow` deterministically hangs whenever the optional `webrtcvad` package is installed (it is, locally and in CI), and after the 300s timeout `timeout_method="thread"` **kills the whole pytest process at ~3% progress** (fix: task-1466). A "test run taking an hour+" on a webrtcvad machine was in fact a run that died at 3% after burning 5 minutes on the hang. Additionally, **81 tests have never run in the suite's history** because their files are named `tests_*.py`, which pytest's `test_*.py` pattern never matches (§4.1).

The mkdtemp leaks are not just hygiene: this machine's user temp dir held **324,000+ leaked `tldw*` test sandboxes totalling ~285GB**, which had filled the disk to 4GB free — mid-audit, test collection itself started failing with `OSError: [Errno 28] No space left on device`. Deleting the stale ones (idle 2+ hours) restored 289GB of free space. Killed runs also leak their bootstrap sandboxes (`pytest_sessionfinish` never runs), which compounds under the repo's many concurrent agent sessions; task-1458 fixes the largest per-run source (1,344 leaks per full UI run).

**Measured result of the quick wins** (identical `-n 8 --dist loadscope` commands, back-to-back on the same machine — §8): origin/dev **1h10m32s** → quick-wins branch **13m56s**, a **5.1× reduction**, with strictly better outcomes (259→227 failed, 23,337→23,372 passed, 17→16 errors; every diff triaged — zero regressions attributable to the changes). Ambient load from concurrent agent sessions was present during both runs but not controlled, so treat 5.1× as this-machine evidence, not a lab number; the per-directory A/Bs in §8 decompose it. The structural phase (CSS parse cache, DB template-copy) should push further. A clean *serial* run could not be completed on this shared machine (§8); projecting from 19% progress, it sits around **~3.5 hours** when it survives at all.

---

## 2. Ranked wall-clock drivers

| # | Driver | Evidence | Est. share |
|---|--------|----------|-----------|
| 1 | `Tests/UI` app mounts | 2,649 `run_test()`, 4,138 `pilot.pause()`, 3,851 tests, 138k LOC | ~50–60% |
| 2 | `_build_test_app()` (`Tests/UI/test_screen_navigation.py:785`) | 1,344 real `TldwCli` builds across 60+ modules; each parses the full 14,616-line CSS bundle; ~15 nested `patch()`; leaks one `mkdtemp` per call (no rmtree) — see §1: 324k+ leaked sandboxes had filled this machine's disk | ~25–35% (subset of #1) |
| 3 | Autouse double `gc.collect()` (`Tests/conftest.py:164-175`) | 2 × full collection × ~11,600 tests | ~10–25% |
| 4 | Worst single files | `test_library_shell.py` (11,400 lines, 212 builds), `test_console_native_chat_flow.py` (170), `test_settings_configuration_hub.py` (164), `test_console_internals_decomposition.py` (113, 10.2s literal pauses), `test_mcp_workbench.py` (560 pauses) | ~15% combined |
| 5 | Autouse isolation fixtures (`Tests/conftest.py:345`, `Tests/UI/conftest.py:61`) | tmp_path + 5 env monkeypatches + config import, per test | ~3–5% |
| 6 | Function-scoped file-backed DB fixtures | 354 file-DB sites vs 139 `:memory:`; 146 `CharactersRAGDB()` full-DDL constructions; only 3 session-scoped fixtures suite-wide | ~4% |
| 7 | Hypothesis property files | 4 modules set global profiles at *import time* (effective example counts depend on collection order!); 34 unbounded `@given`; stateful machines to 1,000 DB ops; `time.sleep(0.2)` inside rule bodies | ~3–5% |
| 8 | Explicit sleeps/pauses | ~122s of numeric `pilot.pause(N)`, ~30s of `time.sleep`; 212 fixed-iteration polling loops | ~5% |
| 9 | Subprocess import-weight tests | ~27 fresh-interpreter boots × ~2s | ~2% |
| 10 | `Tests/RAG_Search/conftest.py:648` | session **autouse** fixture runs `transformers.utils.move_cache()` + downloads/loads a HF model whenever transformers is installed (CI installs it in every job) | fixed 5–60s+/session, per xdist-worker later |

Ruled out as causes: parametrize explosions (max 12/file), real network calls (httpx is consistently `MockTransport`-mocked), snapshot testing (none).

Ruled out as a fix: **session-scoped app reuse** — already tried in this repo, reverted, and there is a regression test against the wedged-compositor state it caused (see `_build_test_app`'s docstring).

## 3. Configuration findings

- **Split-brain config**: `Tests/UI/pytest.ini` coexists with `pyproject.toml`'s `[tool.pytest.ini_options]`. When pytest is invoked as `pytest Tests/UI` (CI does), rootdir flips to `Tests/UI`: `asyncio_mode=auto` and `--strict-markers` turn on, the pyproject `timeout=300` turns **off**, and `Tests/conftest.py`'s autouse fixtures don't load. Two `.pytest_cache` dirs on disk confirm both rootdirs in use.
- **Consequence**: repo root runs use pytest-asyncio **strict** mode, and ~46 `Tests/UI` files rely on auto mode — their `async def` tests are **silently not executed** in a full run from repo root. Fixing the config (task-1457) will surface these as new (recovered) tests needing triage.
- **Marker reality vs CI**: only 27/900 files carry `unit`, 40 `integration`, 6 `ui`. CI's `pytest -m unit` job matrix (8 runs) selects those 27 files while `pip install`ing torch/chromadb/playwright each time. **~590 test files are selected by no PR-triggered job in test.yml** — they run only in `python-app.yml`'s bare serial `pytest ./Tests/` (a second, duplicate, unbounded full run) and the manual-dispatch `all-tests` job.
- **Dead marker plumbing**: `Tests/conftest.py` gates on `optional_deps` — zero tests use it (the real marker is `optional`, 1 file). `Tests/README.md` documents the nonexistent workflow. 25 `@pytest.mark.slow` tests are auto-skipped unless `--run-slow` is passed — nothing (local docs, CI) ever passes it, so they never run anywhere.
- **Dependency split-brain**: `pip install -e ".[dev]"` (the documented setup) installs none of: `pytest-mock`, `pytest-cov`, `pytest-xdist`, `chromadb`, `torch`, `sentence-transformers`, `tiktoken`. Result: documented `--cov`/`-n` commands fail on a clean dev install, and the `embeddings_rag`-gated suites (73+ tests) plus `test_token_counter.py`'s tiktoken tests skip forever.

## 4. Broken tests

### 4.1 Never-collected (81 tests, ~72KB of test code)
`Tests/Prompts_DB/tests_prompts_db.py` (67 tests) and `tests_prompts_db_properties.py` (14 tests) are named `tests_*` — pytest's `python_files = test_*.py` has **never matched them**. They are the bulk of that directory's coverage. Enabling them will surface unknown failures → task-1463 (quarantine protocol).

### 4.2 Collection error aborting all runs (task-1456)
`Tests/Event_Handlers/test_worker_events_contract.py:18` imports `StreamDone` from `tldw_chatbook.Event_Handlers.worker_events`; TASK-650 removed it. Every default `pytest Tests` run dies at collection ("Interrupted: 1 error during collection"). The non-streaming error-propagation coverage (task-634's actual regression) is still valid and should be preserved; the streaming-contract half tests removed behavior.

### 4.3 Testing stubs instead of the product
`Tests/RAG/simplified/test_vector_stores.py` (~900 lines) imports `tldw_chatbook.RAG_Search.simplified.vector_stores` (plural — the real module is `vector_store.py`), catches the ImportError, **defines placeholder classes in the test file, and tests those**. Permanently green; tests nothing real. (Owner decision in task-1464: delete vs rewrite against the real module.)

### 4.4 Deterministic hang killing every full run (task-1466)
`Tests/Audio/test_recording_service.py::TestAudioRecordingIntegration::test_pyaudio_recording_flow` stops its loop from inside the chunk callback, but the callback is gated behind VAD speech detection and the synthetic buffer is silence — with `webrtcvad` installed the loop never exits, and pytest-timeout's thread method kills the entire process after 300s. Its sibling `test_sounddevice_recording_flow` fails on clean dev for the same root cause (its 4-sample chunk is smaller than one 20ms VAD frame, so nothing ever reaches the queue) — previously masked because serial runs died at the hang before reaching it. Both were green on webrtcvad-free machines, which is why they ever passed review. Related trap discovered while working around it: `pytest --deselect` **silently ignores** a nodeid that doesn't match (e.g. missing the class name) — a baseline attempt was lost to this.

### 4.5 Orphaned / disabled-by-extension
- `Tests/UI/ingestion_test_helpers.py` (536 lines): imports the deleted `tldw_chatbook.Widgets.Media_Ingest.*` package (now an empty dir); zero importers anywhere in `Tests/`. Dead. (Deleted in task-1455.)
- `Tests/Chatbooks/test_chatbook_ui_integration.py.skip` (635 lines, git-tracked): disabled by renaming; uses `textual.testing.AppTest`, which doesn't exist in Textual 8.x. (Deleted in task-1455.)
- `Tests/UI/test_tools_settings_window.py:14-17` still guards on `AppTest` "not available in Textual 3.3.0" — stale by ~5 majors; line 135 turns it into a permanent skip.

## 5. Useless / unfalsifiable tests

- **174 tests with zero assertions** (no `assert`, no mock-assert, no `raises`). Clusters: `Tests/UI/test_destination_visual_parity_correction.py` (13 — a "visual parity" suite that verifies nothing), `test_bulk_selection_tooltips.py` (6 of 6), tooltip/recovery families, 7 `*_endpoint_wiring` tests in `Tests/tldw_api/test_notes_workspace_client.py`, 6 `test_metrics_*` in `Tests/Scheduling/test_watchlist_check_handler.py`.
- **99 tests asserting only trivia**: `assert True` "documentation tests" (`Tests/UI/test_worldbook_ui.py:30,47,62`, `test_chat_dictionaries_ui.py` same lines), **three placeholder "security tests"** (`Tests/Web_Scraping/test_security.py:278,290,301` — `assert True  # Placeholder`), import-smoke `X is not None` files.
- **143 tests verifying only mock call-graphs** — worst: `Tests/UI/test_chat_window_enhanced_modules.py`'s 10 consecutive `test_*_delegation` tests that mock the delegate and assert the mock was called.
- **27 tests wrap their assertions in `try: … except Exception:` with no re-raise** — every assertion inside is unenforceable. This includes `Tests/Evals/test_integration.py::test_complete_evaluation_flow` and `::test_budget_monitoring_integration` — **the exact two tests `Tests/Evals/TESTING_SUMMARY.md` lists with ✅ as flagship integration coverage**. Full list lives in task-1464.
- **~226 unconditionally dead tests**: 5 fully-skipped modules (two with reasons contradicted by live code — e.g. "ChatWindowEnhanced not currently in use" while three other files actively test it), 15 rotted skips in `test_chat_events_tabs.py` ("Recursion error due to complex mocking", "not implemented"), env-gated live-API suites with no keys anywhere, the 25 never-run `@slow` tests. **Zero `xfail` usage in the entire suite** — known-broken things are hard-skipped and rot invisibly.
- **Stale docs**: `Tests/RAG/README.md` documents 7 test files that don't exist; `Tests/UI/README_TEST_SUITE.md` documents deleted product code and instructs `--cov` of an empty package; `Tests/TEST_RESULTS_SUMMARY.md` and `Tests/RAG_Search/test_results_summary.md` are undated pass/fail baselines contradicting the current code.

## 6. Flakiness notes

- `test_library_shell.py` has a documented CPU-contention flake (task-192 fixed 12 loops in 7 tests); **~91 fixed-iteration `for _ in range(N): await pilot.pause()` polling loops remain** in that file alone, 212 suite-wide — each a latent flake under load and a fixed cost when the condition is already true.
- Hypothesis global-state hazard: whichever of the 4 import-time `settings.load_profile()` modules is imported **last** silently sets `max_examples`/`deadline` for every unannotated `@given` in the session (fix: task-1452).

## 7. Remediation program (backlog tasks)

Quick wins (this wave):

| Task | Scope | Depends on |
|------|-------|-----------|
| task-1450 | `--durations` in addopts + junit outcome-diff script (measurement protocol) | — |
| task-1451 | De-autouse the RAG_Search HF-model fixture; HF offline defaults | — |
| task-1452 | Hypothesis: central dev/ci/thorough profiles via `HYPOTHESIS_PROFILE`; remove import-time `load_profile()` | — |
| task-1453 | pytest-xdist adoption (`-n auto --dist loadscope`), per-worker config sandbox, dep-extra fixes | 1420–1422 |
| task-1454 | Narrow the double-`gc.collect()` autouse (every-N + `requires_cleanup` marker) + fd-leak sentinel | 1420 |
| task-1455 | Mechanically-safe deletions (orphaned helper, `.py.skip` file) + stale-doc fixes | — |
| task-1456 | Fix `test_worker_events_contract.py` (StreamDone import) — unblocks all full runs | — |
| task-1466 | Fix the `test_pyaudio_recording_flow` VAD hang + its sounddevice sibling — unblocks all full runs on webrtcvad machines | — |

Structural (next wave):

| Task | Scope |
|------|-------|
| task-1457 | Config unification: delete `Tests/UI/pytest.ini`, `asyncio_mode=auto` + `--strict-markers` at root; triage the surfaced dormant async tests |
| task-1458 | Extract `_build_test_app` to a shared factory; ExitStack; fix the mkdtemp leak |
| task-1459 | CSS parse-cache spike (shared `Stylesheet._parse_rules` cache keyed incl. variables fingerprint; canary-gated) |
| task-1460/1431/1432 | DB template-copy pattern per directory (ChaChaNotesDB / Media_DB / Chatbooks): build schema once per session, `copyfile` per test |
| task-1463 | Enable the 81 never-collected Prompts_DB tests under a quarantine protocol |

Owner sign-off required:

| Task | Scope |
|------|-------|
| task-1464 | Decision table: rotted skips, the 27 exception-swallowed tests, ~416 low-value tests, stub-testing `test_vector_stores.py`, `@slow` policy (proposal: weekly `--run-slow` CI job). Introduces the xfail convention. |
| task-1465 | CI rework: xdist in CI, drop the 27-file `-m unit` matrix in favor of directory shards, dedupe/delete `python-app.yml` (branch-protection check first), nightly serial+thorough+slow job |

**Verification protocol for every task above**: `--collect-only` count delta itemized in the PR; junit `(nodeid → outcome)` diff against the §8 baseline — pass→fail must be fixed or quarantined via `xfail(strict=False)` + a task, never silently skipped; pass→missing must map to an itemized deletion.

## 8. Baseline measurements

Serial baseline run: worktree at `665ef1c01` (origin/dev), macOS (darwin 24.6.0), Python 3.12 venv, started 2026-07-30 08:45.
Artifacts: `baseline-serial.log` / `baseline-serial.xml` (junit, per-test wall time) — session scratchpad; durable copies attached to the PR for task-1450.

- Collection alone: **~41s** for 23,793 items.
- Attempt 1 aborted: the §4.2 collection error kills a default run (`--continue-on-collection-errors` required until task-1456 lands).
- Attempt 2 aborted at ~3% after ~15 min: the §4.4 hang + thread-method timeout killed the process (this is what a "full run" looks like today on a webrtcvad machine).
- Attempt 3 lost to the `--deselect` silent-no-match trap (§4.4). Attempt 4 (correct class-qualified deselect) is the recorded baseline.
- Directory-level A/B measurements taken during the audit (same command, same machine, comparable concurrent load):
  - `Tests/Notes` (701 tests): **131.9s** with per-test gc (`TLDW_TEST_GC_EVERY=1` — still fewer collections than dev's double-collect) vs **111.7s** with the task-1454 default — ~15% from gc narrowing alone, identical outcomes.
  - `Tests/Notes` under xdist (task-1453 branch, which still carries the per-test double-gc): **87.0s with `-n 2`** — parallelism compounds with the gc win.
  - Hypothesis property suites after task-1452: ChaChaNotesDB + Utils path-validation 37 passed in 6.3s; RAG_Search + Media_DB properties 26 passed in 13.2s.
  - `Tests/Audio/test_recording_service.py` after task-1466: **34 passed in 1.4s** (was: 300s hang, then process kill).
  - `Tests/RAG_Search` after task-1451 (offline HF, no autouse model load): 66 passed, 12 skipped in 25.0s.
- Serial attempt 4 was **killed by an external signal at 19% (~40 min in)** — this machine hosts concurrent agent sessions that run and manage their own pytest processes; a 3.5-hour serial run is not completable here. Projection from 19%/40min: **~3.5h serial**. The recorded yardstick is therefore the **identical-command parallel pair** below.

### Parallel A/B (the recorded before/after)

Identical command both sides: `pytest Tests -p no:cacheprovider -n 8 --dist loadscope --max-worker-restart=3 --continue-on-collection-errors --durations=50 --timeout=300 --deselect <the §4.4 hang>`, back-to-back on the same machine (14 cores; other agent sessions active during both windows).

| | origin/dev `665ef1c01` | quick-wins branch (tasks 1450–1456, 1466) |
|---|---|---|
| Wall time | **1:10:32** | **0:13:56** (5.1×) |
| Outcomes | 259 failed / 23,337 passed / 181 skipped / 17 errors | 227 failed / 23,372 passed / 181 skipped / 16 errors |

**Clean dev fails 259 tests + 17 errors** — concentrated in `Tests/UI` (169), `Tests/Transcription` (36), `Tests/TTS` (11), `Tests/Chat` (13 errors) — the ungated-suite rot §3 predicts.

junit outcome diff (`Tests/junit_outcome_diff.py`), fully triaged:
- **NEW +3**: the task-1456 rewritten contract tests. **VANISHED 1**: dev's collection-error entry for that same module (the error became three passing tests).
- **RECOVERED 39**: the task-1466 sounddevice fix; 31 Transcription edge-case tests; 4 `test_library_shell` tests; and `Tests/test_hypothesis_profile::test_per_example_deadline_is_disabled` — an existing guard for the central Hypothesis profile that was **failing on dev because of the very profile leak task-1452 fixes**.
- **REGRESSED 7 → all cleared**: rerun in isolation they produce the **identical failure set on the quick-wins branch and on clean dev** (6 fail both sides, 1 passes both sides) — they are pre-existing **order-dependent tests** (4 in `Tests/Performance/test_rag_citation_provenance_benchmark.py`, `Tests/UI/test_library_prompts_canvas.py::…unsaved_marker…`, plus two contention-flaky UI/audio tests) that passed on dev's run only by worker-bucket luck; the branch's +3/−6 file changes reshuffled `--dist loadscope` buckets. Follow-up: task-1467.

Artifacts: `dev-parallel.{log,xml}`, `combined-parallel.{log,xml}`, `baseline-serial.log` (attempts 1–4) — session scratchpad, durable copies attached to the task-1450 PR.

## 9. How to run the suite today (until fixes land)

- Always `.venv/bin/python -m pytest` (system python3 is 3.9 and breaks collection; `python -m` also makes the checkout's own package the code under test).
- Add `--continue-on-collection-errors` until task-1456 merges.
- Don't use `-q` — it suppresses the FAILED summary lines in this repo's setup.
- `pytest -m unit` is **not** "the fast subset" — it's 27 of 900 files. There is currently no curated fast subset; until xdist lands, scope runs by directory.
