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
| DB | `DB/Evals_DB.py` | `EvalsDB`, schema version at `SCHEMA_VERSION` via `PRAGMA user_version`; tasks/datasets/models/runs/results/metrics + probe review tables + A/B tables + FTS |
| Configuration | `Evals/__init__.py`, `Evals/config_loader.py` | `_default_config_path`, `_override_config_path`, `EvalConfigLoader` — shipped defaults plus sparse private YAML overrides |
| Recovery | `Evals/recovery.py`, `Backup_Recovery/settings_file_participants.py` | `eval.definitions` owner, exact private-file selection and raw-participant lifecycle |
| UI | `UI/Screens/evals_screen.py`, `UI/Evals/evals_state.py`, `UI/Evals/sample_bench.py` | `EvalsScreen(LabScreen)`, pure `EvalsViewModel` read side |

In-tree reference docs: `Evals/EVALS_SYSTEM_REFERENCE.md`, `Evals/DEVELOPER_GUIDE.md`. DB path: `config.get_evals_db_path()` (profile-aware).

## Configuration and recovery ownership

The default `EvalConfigLoader` reads `Evals/config/eval_config.yaml` as shipped application data and merges sparse `eval_overrides.yaml` beside the effective `config.toml`. `_default_config_path()` selects the package resource; `_override_config_path()` selects the private file through the CLI configuration path policy. Package defaults never enter the private mutable-file lifecycle.

Dictionaries merge recursively; lists and scalars replace. Explicit updates remain overrides even when equal to a shipped default; mutable `get()` results remain drafts whose changes are captured on save. A missing override file uses the current shipped defaults without creating a private file. Primary saves atomically publish only sparse overrides; explicit loader files remain standalone configurations, and explicit export destinations receive the full effective configuration.

Backup/recovery keeps the `eval.definitions` owner for private YAML, separate from the `db.evals` owner for results. Discovery uses the same override selector, records genuine canonical absence as `unused`, and blocks complete backup on nonregular or unreadable overrides. Legacy recovered definitions retain their existing provenance, inactive destinations and activation requirements; startup never automatically adopts or rewrites them.

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

`EvalsScreen` composes this engine inline (it does **not** go through `EvaluationOrchestrator`): load bench + probe set → snapshot character cards from ChaChaNotes (no cross-DB FK; text copied for provenance) → resolve targets (steering extracted and validated **before** the first provider call) → write the run-group rows (`pending`) before any provider call → run the cards × probes × targets × samples grid, conversations concurrently under a semaphore, turns sequentially (turn N needs turn N−1). Failures are per-conversation (retained turns + `error`); cancellation stops scheduling but in-flight `to_thread` calls run out. Output is collected conversations for **human review** (annotations and probe review tables) — there is no scoring on this path. The default chat factory is hardcoded to `llama_cpp`.

Word bench is the third engine: logprob capture against llama.cpp completions, driven by `sample_bench.run_existing_bench`; it shares `eval_models` targets, run groups, and the steering reader.

## Dataset snippet imports and name edits

`UI/Evals/snippet_editor.py::SnippetEditor` retains the dataset heading,
`DatasetRenameRow` and its live name Input while imports refresh only the
read-only `_SnippetResults` summary, snippet table and warning strip. The
existing import service returns the persisted combined snippets; projecting
those results needs no additional DB read. The table owns scrolling and Import
remains outside that scroll region.

Name edits stay widget-local and uncommitted, including whitespace or invalid
names, with their focus, caret and selection intact when import completes.
Only explicit Rename validates and writes the name; a successful Rename posts
`DatasetRenamed` for the screen's heading/catalog reload. Cancelled, malformed
or unreadable imports leave the name edit unchanged. This follows ADR-033's
narrow state ownership and the existing ADR-150/161 layout contracts.

## Word-bench Save publication

`BenchEditor.Saved` carries the successful write's originating editor and raw
form snapshot. The serialized Evals region swap refreshes Catalog and Inspector
from persisted data, then replaces the detail only if that editor and bench
selection still apply and its form is unchanged. Applicability is checked after
the awaited Catalog replacement, immediately before detail replacement. Save
refresh shares the current selection revision so it cannot supersede a pending
user selection whose detail has not mounted yet.

Later accepted input therefore retains the actual child widgets, focus, caret
and selection; it stays unsaved until another Save. The editor advances its
dirty baseline on successful writes so later edits are compared with the
submitted snapshot. An unchanged ordinary Save still reloads cleaned persisted
values and discards pre-existing uncreated target mini-form text. Validation,
write failure and explicit Revert retain their existing paths. This follows
[ADR-011](../../backlog/decisions/011-chatbook-workbench-ui-system.md),
[ADR-033](../../backlog/decisions/033-application-session-state-ownership.md),
[ADR-150](../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../backlog/decisions/161-component-pattern-library.md).

## Evals keyboard scope

ResultsGrid owns `l` (lens), `b` (baseline), `s` (sort) and `e` (export).
`EvalsScreen._grid_shortcut_context` derives both footer hints and F1 entries
from those bindings, the mounted selected grid's loaded readiness, and Textual's
active binding map. Catalog and Inspector focus cannot dispatch grid commands;
text fields and open Select searches consume printable keys. The footer hides
unreachable commands; F1 lists supported commands with an explicit grid focus
requirement when needed, even when the narrow footer cannot fit them. Empty,
unavailable and failed grid loads retain their visible explanation and expose
no inert commands. Focus and help projection perform no fresh DB/config reads.

The existing global F6 delegate calls Evals' pane handler, cycling Catalog →
detail → Inspector through `focus_relative_workbench_pane`, skipping collapsed
or unavailable panes. Tab continues through individual controls. Selecting a
loaded word-bench run focuses its table so its commands are immediately usable.

## DB notes

Schema version at `DB/Evals_DB.py::SCHEMA_VERSION`, tracked via `PRAGMA user_version` (no version table — contrast ChaChaNotes). `eval_tasks.task_type` CHECK allows only the four basic types; specialized runners are selected via `metadata.category`/`subcategory` or `task_type == "research_report"`, so a specialized task must be stored with a compatible basic type. `eval_models` rows are immutable (no update API — steering is fixed per row). `eval_results` is UNIQUE(run_id, sample_id); run groups (`run_group_id`) pivot the UI's one-row-per-group view with status precedence and an all-cells-failed rollup query.

## Boundaries

- Orchestrator coordinates; execution lives in runners; run/results persistence lives in `EvalsDB`, while private configuration YAML lives under the raw-participant lifecycle; the UI read side is the pure `EvalsViewModel` (no Textual imports).
- Classic evals and the default character-probe chat factory use `Chat.Chat_Functions.chat_api_call` through `to_thread`. Word bench uses its own async HTTP capture client in `Evals/word_bench/capture_client.py` for llama.cpp completions.
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

ADR-031 (`031-bounded-evaluation-and-tool-worker-execution.md`). Specs: `Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md`, `2026-07-25-evals-console-rebuild-design.md`, `2026-08-01-character-probe-eval-design.md`. Configuration ownership: [ADR-220](../../backlog/decisions/220-evaluation-defaults-and-private-overrides.md); backup/recovery: [ADR-126](../../backlog/decisions/126-complete-local-backup-and-recovery.md).

## Verified gotchas

1. `Event_Handlers/eval_db_operations.py` was schema-drifted legacy and was deleted (task-34891); the live path is `EvalsScreen` + `EvalsViewModel`.
2. Two different `BaseEvalRunner` classes exist (`eval_runner.py` is the production one; `base_runner.py` is the older ABC still imported by specialized runners).
3. `EvalsScreen` replaced `EvalsWindowV3` because mounting a Textual Screen inside a plain Container renders at `Region(0,0,0,0)` — the Lab frame with region swaps is the fix; region swaps must step around frame-owned collapse headers.
