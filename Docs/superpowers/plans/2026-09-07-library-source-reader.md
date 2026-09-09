# Library Source Reader Experiment Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the independent source/report units and execute integration inline. Track completion here; do not commit unrelated checkout changes.

**Goal:** Build and locally qualify the approved fixture-only reader comparison harness.

**Architecture:** A small `Evals/source_reader` package assembles revision-pinned Library text, validates exact evidence, and invokes the existing sensitive auxiliary gateway. A fixture-only runner records attempts and produces paired comparison reports without touching live Library or Console state.

**Tech Stack:** Python 3.11+, existing SQLite Library service, existing provider gateway, pytest, stdlib JSON/argparse.

**Spec:** [Approved design](../specs/2026-09-07-library-source-reader-design.md)

**Backlog:** TASK-32029. ADR required: yes. ADR path: `backlog/decisions/133-question-directed-library-reading-experiment.md`. Reason: source selection, derived evidence, provider isolation, and accounting contracts.

## Global Constraints

- Fixture data only; no product tool registration or live Library persistence.
- Six sources, 256 KiB raw input, 64 packets, 4,000-codepoint packets with 200 overlap.
- Worker: 24,000 input tokens, 2,000 output tokens, 60-second result deadline; tighter model limits win.
- Host-owned selection, revision checks, no unscoped fallback, no tools/history in auxiliary requests.
- Exact JSON and quotations; 12 findings, three references per finding, 800-codepoint quotations; independent byte/character result limits.
- Unknown cost and failed attempts cannot produce a financial pass.
- Run only targeted tests. Real provider calls need explicit models and spending limits.
- Work is additive in the current non-main checkout. Existing unrelated work remains untouched.

## Task 1: Source assembly and packets

Files: create `tldw_chatbook/Evals/source_reader/sources.py`; test `Tests/Evals/test_source_reader_sources.py`.

Interfaces: immutable `Selection(source_id: str, revision: str)`, `Source(source_id: str, revision: str, title: str, text: str)`, `Packet(packet_id: str, source_id: str, revision: str, start: int, text: str)`; `assemble_sources(service, selections, source_ids, *, direct_enabled=True) -> tuple[Source, ...]`; `pack_sources(sources) -> tuple[Packet, ...]`; `SourceReaderError` with a bounded `code`.

- [x] Write failing tests against real SQLite Library paging, a Unicode overlap, changed revision, invalid full selection and empty body. Example: `with pytest.raises(SourceReaderError, match="invalid_scope"): assemble_sources(service, (), (foreign_id,))`; verify no body query occurs.
- [x] Run `.venv/bin/python -m pytest -q --no-cov Tests/Evals/test_source_reader_sources.py` and observe assertion failures for the missing behavior.
- [x] Implement using `service.invoke("library_get_media", {"id": source_id, "cursor": cursor, "max_chars": 8000})`; validate selection IDs before any call, then validate each response and continuation before retaining content. Recheck revisions before returning.
- [x] Re-run the targeted tests; inspect boundary cases and added files.

## Task 2: Evidence and auxiliary request lifecycle

Files: create `tldw_chatbook/Evals/source_reader/reader.py`, package `__init__.py`; test `Tests/Evals/test_source_reader.py`.

Interfaces: `validate_findings(text: str, packets, *, max_chars=16000) -> dict`; `build_reader_request(resolution, question, packets, *, context_limit, input_limit=24000, output_limit=2000) -> AuxiliaryCompletionRequest`; `ReaderSession(gateway)` with async `complete(request, deadline=60)` and `drain()` plus observable `pending` and late outcome. No generic fleet or router.

- [x] Write failures for fabricated and duplicate quotes, repeated JSON keys, short sources, Unicode offsets and result fitting. Example: `assert validate_findings('{"findings":[]}', packets)["status"] == "no_evidence_found"` and reject a valid quote attached to an unknown packet ID.
- [x] Run the reader tests before implementation. Then implement strict parsing, host-computed spans, deduplication and whole-finding fitting.
- [x] Add request tests that inspect the far end of a real local HTTP request through `complete_auxiliary`, and a gated synchronous adapter showing a deadline cannot create a second call. Implement one owned shielded task and explicit drainage; errors carry no raw provider text.
- [x] Re-run source and reader tests. Verify no inherited tools, unrelated context or source canaries in ordinary logs.

## Task 3: Paired report accounting

Files: create `tldw_chatbook/Evals/source_reader/comparison.py`; test `Tests/Evals/test_source_reader_comparison.py`.

Interfaces: `summarize_comparison(attempts: list[dict], grades: list[dict], expected: list[dict]) -> dict`. Keys are `case_id`, `repeat`, `arm` (`direct`, `retrieval`, `reader`). Attempts carry `status`, `cost_usd` (nullable), `latency_seconds`; grades carry `success`, `critical_error`, `essential_correct`, `essential_total`. Expected keys come from the frozen matrix, not successful attempts.

- [x] Write failures for missing/duplicate arms, NaN/negative costs, unknown spend, missing human grades, and cheap failures. Hand-derived paired costs of direct `1.0`, reader `0.5` must yield ratio `0.5`, not a ratio of unrelated medians.
- [x] Implement complete-matrix validation, per-case paired metrics and quality gates. Return `inconclusive` for missing evidence or unknown pricing, `reject` for failed quality/cost gates, and `pilot_candidate` only when every gate passes. Prefer retrieval when it matches quality and costs less.
- [x] Run `.venv/bin/python -m pytest -q --no-cov Tests/Evals/test_source_reader_comparison.py` and review independently.

## Task 4: Runnable fixture harness and qualification

Files: create `tldw_chatbook/Evals/source_reader/experiment.py`, `Helper_Scripts/Benchmarks/library_source_reader.py`, fixture JSON under `Tests/fixtures/library_source_reader/`, and `Tests/Evals/test_source_reader_experiment.py`.

- [x] Write failing CLI tests for an offline preparation/report cycle, output overwrite refusal, and live execution without a model/spend limit. Run them, then implement `prepare`, `run`, and `report` argparse commands, using explicit fixture manifests and new output directories.
- [x] Import fixture sources into a disposable real MediaDatabase and use the production Library service for all source reads. Freeze source IDs/revisions, question split and expected interleaved arm matrix. Retain exact evidence/request artifacts without credentials.
- [x] Implement direct/reader requests and a source-scoped existing retrieval adapter. Missing indexes or model capabilities are unavailable arms, not empty successful results. Include query embeddings/reranking in accounting or mark their spend unknown. No silent fallback or automatic real calls.
- [x] Provide blank human-grade rows and operator budget gates. Report missing grades/live runs as inconclusive. Include initial 4 development/12 held-out cases with facts fixed before generation and add separate follow-up scenario support.
- [x] Run the package's targeted suites, local recording-server check, lint/format of added code, CLI help, offline preparation, and report generation. Review all new code against the spec, then document observed results and remaining live evaluation in TASK-32029.

## Execution record

- Local implementation and review complete. New suites were run failing before their implementations; review regressions were reproduced before fixes. Final combined result: 143 targeted tests passed; Ruff lint and formatting passed for all eleven added Python files.
- Real loopback HTTP checks qualified both gateway transports, request isolation, no retry after HTTP 503, logging canaries, and malformed raw usage. Offline CLI help, preparation, and report generation passed.
- Prototype deviations: the CLI records retrieval as unavailable until a fixture index is connected; request artifacts capture the auxiliary boundary rather than final wire bytes; ephemeral quote records defer canonical product citation ownership. See backlog/docs/library-source-reader-local-qualification-2026-09-08.md.
- User selected DeepSeek. Pro answers and Flash reads via native Chat Completions with thinking explicitly disabled; the experiment bridge preserves validated cache-hit/miss usage. Existing ADR-133 and ADR-064 apply; no new ADR is required because provider isolation and runtime ownership are unchanged.
- DeepSeek continuation passed 163 source-reader tests, three existing DeepSeek chat tests, and Ruff lint/format checks. Independent review found and reproduced pricing-boundary calls prematurely stopping the matrix; valid bounded costs now allow continuation while incomplete usage still stops dispatch.
- Official model availability was verified with an authenticated read-only list request. A reviewed dry run reserves at most 24 development requests, 240,000 tokens, and $0.9504 at peak pricing under a proposed $1 ceiling. No generation was dispatched. See the local qualification report for provenance and the manifest path.
- The user approved the $1 / 24-request development ceiling. The unchanged reviewed configuration completed 20 calls for an estimated $0.003328. Four empty reader results prevented their main calls; all eight direct attempts returned answers. Independent review reconciled costs and source provenance. The formal report remains inconclusive, and the development recommendation is revise before further evaluation. See backlog/docs/library-source-reader-deepseek-development-2026-09-08.md.
- Negative and absent-information evidence, fixture retrieval, held-out outputs, and human grades remain outstanding. Do not mark the overall experiment task Done on development execution alone.
- Approved negative-evidence revision: clarified only the reader prompt, retaining explicit negation and unspecified-information evidence without turning silence into a fact. The original recorded failure check was red; a focused four-call live probe now retains both required quotations and delivers both answers. Cumulative usage is 24 calls and an estimated $0.005122. The 47 affected reader/runner/transport tests and Ruff checks pass. The probe is separate from the full comparison; see backlog/docs/library-source-reader-negative-evidence-revision-2026-09-08.md.

## Remaining evaluation

- [x] Select DeepSeek Pro/Flash, verify the official endpoint and model availability, test explicit thinking controls/cache accounting, and prepare a capped development dry run.
- [x] Obtain the operator spending ceiling and qualify live development generation, actual usage, and provider behavior, with the recorded artifact/served-identity limits.
- [x] Revise reader handling of explicit negative and unspecified-information evidence and verify the two observed failures in a focused development probe.
- [ ] Repeat the complete development comparison, including positive controls, before using the held-out set; obtain an additional explicit request allowance for any further model calls.
- [ ] Connect a disposable fixture retrieval index and account for embeddings/reranking.
- [ ] Run frozen held-out comparisons, collect blind human grades, and record the adoption decision.
