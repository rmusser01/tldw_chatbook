---
id: TASK-41524
title: "The Evals word bench falls back to http://localhost:8080 when the configured llama.cpp URL is refused"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-18 03:21'
created_date: 2026-11-17 05:27
labels:
- follow-up
- evals
- providers
priority: medium
references:
- tldw_chatbook/Evals
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in endpoint fix r1 (round 15), after the I-3 fix. When the configured llama.cpp URL is refused, the bench shows "not configured", but an existing unsteered target still sends to the hard-coded `http://localhost:8080` fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A refused or missing configured URL sends nothing, and the bench says why
- [x] #2 The fallback address is removed or becomes an explicit setting
- [x] #3 A test shows no request reaches the fallback
<!-- AC:END -->

## Implementation Plan

1. Check at base: `_default_client_factory` falls back to `http://localhost:8080` when `configured_llama_cpp_url` is None (missing or refused URL), and both the sample bench (reusing an existing target) and a saved bench's run reach it.
2. Red first: with an existing llama_cpp target and a refused or missing URL, running the sample bench and an existing bench sends nothing (a recording transport sees no request) and the failure names why.
3. Fix in `sample_bench`: one helper returns the bench URL or the reason; the factory raises that reason instead of falling back; the sample bench resolves the client before it creates its dataset and bench, so a refusal writes nothing.
4. Covering Evals tests, preflight, ratchets.

## Implementation Notes

Checked at base 45b1949ca3: `sample_bench._default_client_factory` used `configured_llama_cpp_url(app_config) or "http://localhost:8080"`. With a refused (TASK-41131) or missing `[api_settings.llama_cpp] api_url`, `configured_llama_cpp_url` is None and the rail reads not configured, yet an existing llama_cpp target still ran: the sample bench reuses it (`resolve_sample_target`) and a saved bench resolves its own, and both built their client on the fallback. The character bench goes through `chat_api_call("llama_cpp")`, whose adapter has no fallback and refuses such a URL already.

**Fix** (`UI/Evals/sample_bench.py`): the fallback is removed (no setting added, per the brief). `_llama_cpp_endpoint` returns the bench's root URL, or None and why: "No llama.cpp endpoint is configured: set [api_settings.llama_cpp] api_url." or "The configured llama.cpp endpoint is refused: <the contract's reason>" (the reason never echoes the URL). `configured_llama_cpp_url` is now its first half, unchanged in behaviour. `_default_client_factory` raises `RuntimeError(why)`, which both bench workers already show as "Could not create the sample bench: ..." / "Could not run the bench: ..." with `markup=False`. `create_and_run_sample_bench` builds its client before it writes anything, so a refusal leaves no orphan dataset or bench; `run_existing_bench` already built it before any send.

**Red/green** (lane file `Tests/Chat/test_local_server_discovery.py`, `test_the_word_bench_sends_nothing_without_a_usable_endpoint[refused|missing]`): an in-memory EvalsDB with an existing llama_cpp target and a saved bench; the real `WordBenchCaptureClient` is built on an `httpx.MockTransport` that records every request. Both entry points must raise with the reason, record no request to any address, and add no dataset or bench row. Red at base: "DID NOT RAISE" for both cases; with the raise check relaxed, the transport recorded 7 requests to `http://localhost:8080/v1/completions` in each case. Green after. It carries `@pytest.mark.bootstrap_profile` (its EvalsDB writes hit `RecoveryRequired: raw_source_selection_changed` under the per-test redirect, as `Tests/UI/test_evals_empty_states.py` does for the same reason).

**Runs** (all through the wrapper, `-n 4` unless noted):
- The new test: base sources red as above, this branch green (both cases), also inside the lane-flag serial run of the whole file (TASK-41523's notes: 982 passed, each new node under 1 s).
- Every test file that uses `sample_bench` (`Tests/Evals/test_evals_screen_worker_diagnostics.py`, `Tests/Evals/word_bench/test_run_existing_bench.py`, `test_runner.py`, and `Tests/UI/test_evals_authoring_e2e.py`, `_bench_editor.py`, `_cell_continuation_e2e.py`, `_continuation_e2e.py`, `_empty_states.py`, `_screen.py`, `_steering_e2e.py`). These UI files are contention-sensitive here: at load 60-100 one command over all ten failed 59 and 92 nodes at this branch and 14 with base sources. Every failure was a timing one (`NoMatches` right after a `select()` and one pause, "condition never became true") or the local `RecoveryRequired` trap. Controls: `test_evals_bench_editor.py` alone failed 5 with each source set at load 40-49; `test_evals_screen.py` + `test_evals_empty_states.py` failed 36 here and 10 at base back to back at load ~55, and **203 passed, 0 failed** here at load ~27 (so no node fails in every run at this branch); the remaining eight files at load ~23: 9 failed, 8 of them failing with base sources too (`RecoveryRequired`, the harness's `EvalsScreen._nodes`), the ninth (`test_add_from_picker_stages_a_row_and_save_persists_it`) passing 3 of 3 alone, as did 4 sample-bench and selection tests run alone 3 times.
- `./scripts/preflight.sh` passed; the module and screen size ratchets passed (235, serial).

Files: `tldw_chatbook/UI/Evals/sample_bench.py`, `Tests/Chat/test_local_server_discovery.py`. No User Guide page describes the bench's endpoint (checked `Docs/User_Guide/lab.md`).
