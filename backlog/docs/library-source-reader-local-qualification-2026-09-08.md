# Library source reader: local qualification

Date: 2026-09-08
Task: [TASK-32029](../tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md)
Decision: [ADR-133](../decisions/133-question-directed-library-reading-experiment.md)
Design: [Source-reader experiment](../../Docs/superpowers/specs/2026-09-07-library-source-reader-design.md)

## Outcome

The fixture-only experiment is implemented. It can prepare a disposable Library
database, run explicitly configured direct/reader requests, and report paired
quality, cost, and latency evidence. It changes no shipped Console behavior.
The approved DeepSeek development run has now completed: 20 calls, an estimated
$0.003328, and four of eight reader attempts returning no evidence. The
[live development report](library-source-reader-deepseek-development-2026-09-08.md)
recommends revision before further evaluation. No blind human grading has run;
the adoption decision is **inconclusive** and TASK-32029 remains In Progress.

A subsequent approved prompt revision repaired both observed negative-evidence
failures in a focused four-call probe. Cumulative usage is 24 requests and an
estimated $0.005122. See the [revision report](library-source-reader-negative-evidence-revision-2026-09-08.md).
The full development comparison, including positive controls, still needs a
rerun; the focused probe does not replace the frozen first-run results.

The corpus contains four development and twelve held-out questions, each with
fixed essential facts and critical-error definitions. Two interleaved repetitions
produce 72 held-out arm rows. An optional separate three-question scenario tests
repeated questions on the same sources in independent contexts; it does not
simulate a continuing conversation or establish multi-turn savings.

## Implemented boundaries

- Complete source-selection validation precedes bounded production Library
  reads. Real SQLite tests cover paging, deleted sources, revision changes,
  Unicode offsets, overlap, and byte/packet ceilings.
- Reader output undergoes strict JSON validation and exact, uniquely located
  quote matching. Host code computes source spans and drops whole findings to
  fit output limits. Exact location is distinguished from semantic support.
- Auxiliary requests contain only fixed instructions, a question, and selected
  evidence. Known model caps override looser operator caps. A timed-out or
  cancelled request remains owned until drainage; late content is discarded.
- Reports retain failed attempts, missing arms, unknown usage, known portions
  of partially observed costs, and late usage. Missing evidence cannot produce
  a financial pass. No model-generated grading is used.

## Review corrections

Local review reproduced and fixed malformed escaped Unicode crashing the
output fitter, known model caps being ignored, a caller overriding the deadline,
partial paid calls disappearing from reported spend, and normalized malformed
usage appearing fully priced. The review also tightened complete-matrix
retention, frozen grading facts, cancellation accounting, and artifact labels.

## Running the harness

Run from the repository root with the project virtual environment. Preparation
and reporting make no model calls. Each output directory must be new:

```bash
.venv/bin/python Helper_Scripts/Benchmarks/library_source_reader.py prepare \
  --output /private/tmp/chatbook-reader-fixtures --include-followup

.venv/bin/python Helper_Scripts/Benchmarks/library_source_reader.py report \
  --prepared /private/tmp/chatbook-reader-fixtures \
  --output /private/tmp/chatbook-reader-report

.venv/bin/python Helper_Scripts/Benchmarks/library_source_reader.py run --help
```

`run` requires explicit main/worker model IDs, endpoint, credential environment
variable name, split, context capacity, and request/token/USD ceilings. It
reserves the whole runnable split before dispatch and refuses insufficient
budgets or unavailable pricing. It supports the existing OpenAI adapter or
`--provider deepseek`, with sensitive logging and no retries. `--dry-run` writes
the complete preflight manifest and grading template without model dispatch.
A 60-second result deadline is distinct from the
adapter's configured transport timeout; shutdown may wait for an outstanding
synchronous call. Unknown usage stops subsequent model calls.

Start with development cases. Freeze prompts, model choices, limits, and scoring
before held-out calls. Inspect the run's request and pricing manifests, conduct
blind human grading, then provide the run directory and completed grades to
`report`. Give reviewers only `blind_grading_packet.json`; keep
`grading_key.json` with the operator. The report maps completed `review_id`
grades back to the frozen arm matrix. Dollar values are estimates from the recorded model-price catalog;
custom endpoint billing and actual served model identity need operator
qualification. An API spending ceiling is not a provider-side billing limit.

The existing `Evals` package imports application configuration even for CLI
help. To isolate that startup behavior, set `TLDW_CONFIG_PATH` to a scratch TOML
file whose `[paths] data_dir` points to an existing private scratch directory.
The qualification below used this arrangement; no live Library was opened.

## Remaining qualification gaps

1. DeepSeek Pro/Flash development execution and a focused negative-evidence
   repair probe are complete within the approved $1/24-request ceiling. A full
   development rerun including positive controls is still needed before
   held-out evaluation; the remaining limitations below still apply.
2. The existing semantic retrieval adapter applies selected IDs before top-k and
   checks indexed revisions and exact text. The CLI currently records
   `index_unavailable`; a disposable fixture index and complete embedding,
   reranking, and index-build accounting are needed for a three-way comparison.
3. No held-out answer quality, cache-effect, cost-saving, or latency claim is
   supported by local contract tests. The baseline corpus is small and mostly
   synthetic; even passing it would justify only a limited pilot.
4. `requests.json` contains exact auxiliary gateway requests, not a capture of
   final HTTP bodies. Local recording-server tests verify the actual adapter
   path. Provider transformations and served identity remain a live qualification
   step; all relevant protocol and pricing metadata must be frozen for that run.
5. Experiment packet/quote records use shared Library ID adapters but are plain
   ephemeral evaluation structures. The planned canonical evidence-type/trace
   integration is deferred: existing trace types require product request/run
   ownership that this harness does not create. No canonical citation persistence
   or product reachability is claimed.

## Verification evidence

**143 targeted tests passed** across the five `Tests/Evals/test_source_reader*`
modules. This includes real SQLite, real loopback HTTP through both auxiliary
transport paths, HTTP 503 with exactly one request, missing raw output usage,
and synchronous-call cancellation. The first combined run caught the explicit
`reasoning_effort="none"` routing bug; the final run passed after omitting that
parameter. Tests used local response servers and recording doubles, with no paid
API calls. Ruff lint and format checks passed for all eleven added Python files.
The full suite was not run. The environment emitted its existing Requests
dependency-version warning.

An additional check of existing Media database path-ownership tests reported
four failures:

```bash
.venv/bin/python -m pytest -q --no-cov Tests/DB/test_core_sqlite_owner_privacy.py \
  -k 'core_owner_rejects_unsafe_namespace_before_raw_sqlite and media'
```

These tests expect `PrivatePathError` as the deepest cause, while the unchanged
Media database constructor scrubs it to `DatabaseError` using `from None`.
Separate real constructor probes confirmed both symlink and hardlink rejection
before any raw SQLite connection, with target bytes, permissions, and timestamps
unchanged. The suspected source-isolation finding was therefore retracted;
the existing exception-expectation failures remain outside this experiment's
additive changes. They are separate from the 143 passing new tests.

Offline `--help`, `prepare --include-followup`, and `report` succeeded using a
separate configuration/data directory. The local artifacts are under
`/private/tmp/library-source-reader-qualification-20260908-01/`; the generated
report correctly returns inconclusive without model attempts or grades. These
temporary paths are local evidence, not committed benchmark results.

## DeepSeek preparation, before the approved live run

The user selected DeepSeek. The prepared development comparison uses
`deepseek-v4-pro` for answers and `deepseek-v4-flash` for the reader, through
`https://api.deepseek.com`. An authenticated, read-only model-list request
returned HTTP 200 and both IDs. No generation request was made. The credential
remains in the existing local configuration and is passed only through the
runner's environment; artifacts contain no key.

The fixture-only bridge reuses the native Chat Completions adapter and sets
`thinking.type=disabled`, following the provider's
[thinking-mode contract](https://api-docs.deepseek.com/guides/thinking_mode/).
The optional native argument preserves existing callers' defaults. The bridge
validates DeepSeek's cache-hit/miss counters before translating them into the
shared normalized usage format. Invalid or missing buckets stop further calls.
No new runtime owner is introduced; ADR-133 and
[ADR-064](../decisions/064-deepseek-dual-api-provider-boundary.md) apply.

Experiment pricing uses the
[current provider table](https://api-docs.deepseek.com/quick_start/pricing/?article_id=article_1779470751466_8)
and its [August price announcement](https://api-docs.deepseek.com/news/news260813/),
verified on September 8. The shared application catalog is unchanged. Preflight
reserves peak prices. Recorded UTC request intervals select peak/off-peak
prices; intervals crossing a boundary retain only an upper bound and cannot
contribute an exact cost or known subtotal. Review caught and fixed valid
boundary-crossing calls being misclassified as unknown usage and stopping the
matrix. Missing token buckets still stop dispatch, including at rate boundaries.

The reviewed dry run prepared four development questions with two repetitions:
24 maximum generation requests, 240,000 reserved tokens, 8,000 input and 2,000
output tokens per request, and a 32,768-token context ceiling. A proposed $1
ceiling covers the conservative $0.9504 reservation. It is a preflight estimate,
not a provider-enforced billing limit. The reviewed manifest is at
`/private/tmp/library-source-reader-qualification-20260908-01/deepseek-development-preflight-reviewed/run_manifest.json`.
It contains `dry_run: true`; no attempts were dispatched.

Latest verification: **163 source-reader tests passed**, including real local
HTTP DeepSeek success/error/cache-accounting checks and three run-level pricing
boundary regressions. **Three existing DeepSeek chat tests passed** (95 other
tests deselected). Ruff lint and format checks passed for all thirteen package
and source-reader test files. No full suite or paid generation was run. The
existing Requests dependency warning remains. The retrieval baseline and human
grading gaps above still prevent an adoption or savings conclusion.

## Approved live run

After the user approved the $1 / 24-request ceiling, the unchanged reviewed
configuration dispatched 20 development calls. Four empty worker results
prevented their main calls. All normalized model costs were available and
totalled $0.003328; all eight unavailable retrieval rows retain unknown cost.
The executed manifest matched preflight except for `dry_run: false`.

Independent artifact review reconciled usage/cost totals, checked the frozen
matrix and source digests, and verified all four accepted quote spans. Direct
reading returned eight answers; the reader returned four, each costing more
than its direct counterpart. No model or human grades were supplied. See the
[live report and frozen evidence](library-source-reader-deepseek-development-2026-09-08.md)
for the results, limitations, and required development improvements. No runtime
code was edited during execution, and no additional paid call was made.
