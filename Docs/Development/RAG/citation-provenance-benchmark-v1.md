# RAG Citation Provenance Benchmark v1

This benchmark records the pre-feature local performance and storage baseline
required by ADR-024. It is deterministic, uses synthetic non-sensitive data,
and makes no network request.

## Commands

Run from the repository root with the project virtual environment active:

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode baseline \
  --samples 30 \
  --warmups 5 \
  --output Docs/Development/RAG/citation-provenance-baseline-v1.json
```

Qualification requires the committed v1 result:

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode qualification \
  --samples 30 \
  --warmups 5 \
  --baseline Docs/Development/RAG/citation-provenance-baseline-v1.json
```

External resolver latency is an explicit, informational measurement:

```bash
python Helper_Scripts/Benchmarks/rag_citation_provenance_benchmark.py \
  --mode external \
  --provider external-http-v1 \
  --external-target https://example.org/health \
  --external-timeout-seconds 10 \
  --samples 30 \
  --warmups 5 \
  --output /tmp/rag-citation-provenance-external.json
```

Baseline and qualification modes accept only the in-process
`mock-local-v1` provider and no base URL. An absent baseline, or one with an
incompatible fixture/result schema or incomplete/non-finite required metric,
is rejected before measurement with a sanitized error. A structurally valid
baseline from an environment outside the recorded envelope may still be
measured, but the result cannot claim a passing qualification.

Qualification runs with fewer than 30 measured samples or five warmups are
permitted as quick diagnostics. They report fixed ineligibility reason codes
and `overall_pass: false`; only runs at or above 30/5 can claim delivery
qualification.

Only external mode may perform network I/O. It requires the
`external-http-v1` provider and an explicit absolute HTTP(S) target without URL
credentials. Its output reports latency separately with `overall_pass: null`;
it does not load, overwrite, or gate against the local baseline.

## Reference environment

The committed result was measured on:

| Component | Value |
| --- | --- |
| CPU / architecture | Apple Silicon-compatible ARM64 (`arm`) |
| Operating system | macOS 15.6 / Darwin 24.6.0 |
| Python | CPython 3.12.11 |
| SQLite | 3.49.1 |
| Provider | deterministic in-process `mock-local-v1` stream |
| Network | disabled |
| Fixture | `rag-citation-provenance-v1` |
| Result schema | 1 |

The supported v1 qualification envelope is the same OS family, ARM64
architecture, Python 3.12 minor line, fixture/result schema, mock provider, and
network-disabled mode recorded in the JSON result. Other environments remain
useful for measurements but are not comparable enough to claim pass.

## Measurement rules

- Five complete warmups are discarded before 30 measured samples.
- Every duration uses `time.perf_counter_ns`.
- Results report the median and nearest-rank p95; minimum values are never
  reported.
- Each sample group receives a fresh temporary `ChaChaNotes.db` and
  `chat_rag_context.json`. No user database or sidecar is read or written.
- Before the first Chatbook import, the runner redirects HOME, XDG config/data,
  and `TLDW_CONFIG_PATH` to its temporary workspace, enables test mode, and
  hides inherited secret-bearing environment variables. It restores the
  original environment after the Console measurement; host configuration,
  credentials, and user data are never loaded or modified.
- SQLite size measurements run `PRAGMA wal_checkpoint(TRUNCATE)` before each
  file-size observation.
- The mocked native first-token path runs through
  `ConsoleChatController.submit_draft` and
  `ConsoleChatController._stream_assistant_response`, including message
  persistence and a deterministic two-chunk stream derived from the fixture
  corpus. The mock applies a fixed 20 ms first-token floor so sub-millisecond
  scheduler and SQLite noise does not dominate the 10% comparison; the
  independent 25 ms regression ceiling remains enforced.
- The pre-feature baseline has one current implementation, so its recorded
  candidate and unchanged no-provenance control are the same measured series.
  Pre-feature qualification also reuses that series because no provenance
  candidate exists yet. Later candidate tasks extend this runner with their
  in-process candidate while retaining this unchanged control. Qualification
  compares candidate p95 directly with both the compatible committed v1
  candidate p95 and the current in-process control; each comparison must stay
  within both first-token ceilings.
- Generation, finalization, inspector, database-growth, and migration workloads
  consume the versioned corpus records and representative shape families. The
  result records coverage and stable corpus-input hashes for the measured
  seams.
- Finalization hashes governed snapshots and serializes bounded immutable
  metadata. Inspector, database-growth, and migration measurements use local
  SQLite proxies.
- External source resolution runs only in explicit external mode and never
  contributes to local pass/fail.

## Frozen budgets

| Family | Pass/fail budget |
| --- | --- |
| First token | Candidate p95 increase is at most 10% **and** 25 ms over the compatible v1/current control |
| Finalization | 8 × 4 KiB snapshots p95 ≤ 75 ms; exactly 4 MiB across at most 64 snapshots p95 ≤ 250 ms |
| Inspector load | Cold p95 ≤ 100 ms; warm p95 ≤ 25 ms |
| Trace size | Immutable aggregate ≤ 256 KiB; snapshot ≤ 64 KiB UTF-8; governed trace payload ≤ 4 MiB |
| Database growth | Per grounded answer ≤ governed bytes × 1.35 + 256 KiB |
| Migration | Median ≥ 100 messages/s; restart produces zero duplicate canonical proxy rows |

## Frozen v1 bounds

The corpus contains deterministic exact-limit and one-unit-over descriptors for
every bound. The runner materializes domain-shaped values and sends each pair
through the applicable validator: exact values are accepted and one-unit-over
values are rejected. The exact 4 MiB governed trace is split into 64 snapshots
of 64 KiB so the aggregate case also satisfies the per-snapshot bound.

| Value | Maximum |
| --- | ---: |
| Immutable aggregate JSON | 256 KiB |
| Governed snapshot text | 64 KiB UTF-8 |
| Governed trace payload | 4 MiB |
| Prompt sets | 8 |
| Evidence entries per prompt | 64 |
| Answer attempts | 8 |
| Citation occurrences | 512 |
| Retrieval candidates per run | 200 |
| Locator JSON | 16 KiB |
| Observation JSON | 8 KiB |
| Error/reason code | 256 characters |
| External opaque ID | 256 UTF-8 bytes |
| Answer-attempt body | 1 MiB UTF-8 |
| Legacy sidecar | 32 MiB |
| Migration batch | 100 messages |

## Current result

All six families pass on the reference environment.

| Metric | Median | p95 | Budget result |
| --- | ---: | ---: | --- |
| Mocked Console first token | 22.787 ms | 24.730 ms | Pass; baseline regression 0% / 0 ms |
| Standard finalization | 0.042 ms | 0.118 ms | Pass |
| Maximum finalization | 2.884 ms | 3.765 ms | Pass |
| Inspector cold load | 0.288 ms | 2.522 ms | Pass |
| Inspector warm load | 0.079 ms | 0.246 ms | Pass |
| SQLite growth per 4 MiB governed answer | 4,227,072 bytes | 4,235,264 bytes | Pass; allowance 5,924,454 bytes |
| Legacy migration | 74,954 messages/s | 100,768 messages/s | Pass; zero duplicate rows |

The maximum trace proxy contains 4,194,304 governed bytes, a 65,536-byte
largest snapshot, and 6,744 bytes of immutable aggregate JSON.

The full machine-readable measurements, budget definitions, individual checks,
qualification eligibility, environment envelope, sample counts, and
external-network exclusion are in `citation-provenance-baseline-v1.json`.

## Limitations

- This is a pre-feature benchmark. Finalization, inspector reads, storage
  growth, and legacy migration are bounded proxies for the later canonical
  repositories, not claims about code that does not exist yet.
- The inspector proxy distinguishes a new SQLite connection from a reused
  connection; it does not flush the operating-system disk cache.
- The deterministic mock isolates Chatbook overhead and is not a model/provider
  latency benchmark.
- External refresh latency depends on resolver, authority, cache, and network.
  It is measured only by the explicit external workflow and never gates local
  answer rendering, finalization, persistence, or inspector opening.
