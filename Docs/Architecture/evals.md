# Evals orchestration

This document describes the benchmarking stack: the orchestrator/runner split, the task-type runners, the EvalsDB schema, the character-probe and word-bench engines, A/B testing, and the Evals screen. There are three distinct engines — classic evals, word bench, and character probe — sharing one DB and one screen.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Orchestrator | `Evals/eval_orchestrator.py` | `EvaluationOrchestrator` (composition: `ConcurrentRunManager`, `ConfigurationValidator`, `ErrorHandler`, `EvalsDB`, `TaskLoader`); `run_evaluation`, `export_results`, `quick_eval` |
| Runner | `Evals/eval_runner.py` | `EvalRunner` facade over `BaseEvalRunner` + task runners: `QuestionAnswerRunner`, `ClassificationRunner`, `LogProbRunner`, `GenerationRunner`; `DatasetLoader`, `MetricsCalculator` |
| Specialized runners | `Evals/specialized_runners.py` | code-execution (subprocess sandbox), safety, multilingual, creative, robustness, math-reasoning, summarization, dialogue, research-report |
| Task loading | `Evals/task_loader.py` | `TaskLoader.load_task` (eleuther/custom/huggingface/csv), templates, validation |
| Character probe | `Evals/character_probe/` | `CharacterProbeRunner` (cards × probes × targets × samples grid), storage over EvalsDB, card snapshots from ChaChaNotes |
| Word bench | `Evals/word_bench/` | logprob capture against llama.cpp completions; `sample_bench.run_existing_bench` |
| A/B testing | `Evals/ab_testing.py` | `ABTestRunner.run_ab_test` (two orchestrator runs gathered), statistical tests, confidence intervals |
| Research scoring | `Evals/research_report_scorer.py` | deterministic citation/grounding metrics for research reports |
| Steering | `Evals/steering.py` | `model_steering` — prefix (raw) xor system_prompt (chat) from the model row's config JSON |
| DB | `DB/Evals_DB.py` | `EvalsDB`, schema v5 via `PRAGMA user_version`; tasks/datasets/models/runs/results/metrics + probe review tables + A/B tables + FTS |
| UI | `UI/Screens/evals_screen.py`, `UI/Evals/evals_state.py`, `UI/Evals/sample_bench.py` | `EvalsScreen(LabScreen)`, pure `EvalsViewModel` read side |

In-tree reference docs: `Evals/EVALS_SYSTEM_REFERENCE.md`, `Evals/DEVELOPER_GUIDE.md`. DB path: `config.get_evals_db_path()` (profile-aware).

## Classic eval run (dataflow)

1. **Configure**: task (file or template; loader validates → DB row), model config (validator → DB row), dataset (format inferred from suffix).
2. **Launch** `run_evaluation`: load task+model rows (typed validation errors), split overrides into model params (`temperature`, `max_tokens`, `top_p`, `top_k`) vs task fields (filtered by signature), auto-name the run `task_model_timestamp`.
3. **Durable run row first**: `create_run` → concurrency manager registration → status `running` → `run_started_callback` lets the UI navigate before completion.
4. **Execute** `EvalRunner.run_evaluation`: load samples (local json/csv or HuggingFace), one asyncio task per sample bounded by a semaphore (default 10); per-sample exceptions become `FATAL_ERROR` sample results — the run continues. Every LLM call is `asyncio.to_thread(chat_api_call, …)` under `wait_for(request_timeout)` — inference reuses the app's normal chat dispatcher, non-streaming, with a fixed extra-kwarg whitelist.
5. **Stream results**: each settled sample is persisted immediately (`store_result`) and forwarded to the progress callback; results are placed positionally so ordering is stable.
6. **Aggregate**: `<metric>_mean`/`_count`, `total_samples`, `error_count`, `success_rate` → `store_run_metrics`.
7. **Finalize honestly**: `failed` if any sample errored **or** stored-result count ≠ expected (re-read from the DB); `cancelled` on CancelledError (re-raised); `finally` always unregisters. `close()` refuses while runs are active.
8. **Export**: JSON/CSV from the orchestrator; richer formats (markdown, LaTeX for A/B) via `Evals/exporters.py`.

## Character probe (distinct path)

`EvalsScreen` composes this engine inline (it does **not** go through `EvaluationOrchestrator`): load bench + probe set → snapshot character cards from ChaChaNotes (no cross-DB FK; text copied for provenance) → resolve targets (steering extracted and validated **before** the first provider call) → write the run-group rows (`pending`) before any provider call → run the cards × probes × targets × samples grid, conversations concurrently under a semaphore, turns sequentially (turn N needs turn N−1). Failures are per-conversation (retained turns + `error`); cancellation stops scheduling but in-flight `to_thread` calls run out. Output is collected conversations for **human review** (annotations, review state in v5 tables) — there is no scoring on this path. The default chat factory is hardcoded to `llama_cpp`.

Word bench is the third engine: logprob capture against llama.cpp completions, driven by `sample_bench.run_existing_bench`; it shares `eval_models` targets, run groups, and the steering reader.

## DB notes

Schema v5 via `PRAGMA user_version` (no version table — contrast ChaChaNotes). `eval_tasks.task_type` CHECK allows only the four basic types; specialized runners are selected via `metadata.category`/`subcategory` or `task_type == "research_report"`, so a specialized task must be stored with a compatible basic type. `eval_models` rows are immutable (no update API — steering is fixed per row). `eval_results` is UNIQUE(run_id, sample_id); run groups (`run_group_id`) pivot the UI's one-row-per-group view with status precedence and an all-cells-failed rollup query.

## Boundaries

- Orchestrator coordinates; execution lives in runners; persistence only in `EvalsDB`; the UI read side is the pure `EvalsViewModel` (no Textual imports).
- All provider traffic flows through `Chat.Chat_Functions.chat_api_call` (sync — hence `to_thread` everywhere).
- Character cards live in ChaChaNotesDB — snapshots, not joins.
- `EvalsScreen` runs word/character benches only; the classic `EvalRunner` path is exercised via `quick_eval`, A/B tests, and service APIs.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| DB "locked" OperationalError | Retryable `DatabaseError`; others non-retryable |
| Per-sample provider failure | Retried per config (honoring `Retry-After`); then a FATAL_ERROR sample — batch continues |
| Orchestrator wiring failure in UI | `EvalsViewModel(db=None)` degrades to empty lists, never raises |
| Legacy evals DB present | Warning only — never migrated |
| Error toasts with user-controlled names | `notify(..., markup=False)` mandatory (Rich-markup crash guard) |

## Governing decisions and docs

ADR-031 (`031-bounded-evaluation-and-tool-worker-execution.md`). Specs: `Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md`, `2026-07-25-evals-console-rebuild-design.md`, `2026-08-01-character-probe-eval-design.md`. Config: `Evals/config/eval_config.yaml` (task types, valid metrics per type).

## Verified gotchas

1. `Event_Handlers/eval_db_operations.py` is schema-drifted legacy — its read/write calls don't match `EvalsDB`'s shapes and always hit the broad except; only its own test references it. The live path is `EvalsScreen` + `EvalsViewModel`.
2. Two different `BaseEvalRunner` classes exist (`eval_runner.py` is the production one; `base_runner.py` is the older ABC still imported by specialized runners).
3. `EvalsScreen` replaced `EvalsWindowV3` because mounting a Textual Screen inside a plain Container renders at `Region(0,0,0,0)` — the Lab frame with region swaps is the fix; region swaps must step around frame-owned collapse headers.
