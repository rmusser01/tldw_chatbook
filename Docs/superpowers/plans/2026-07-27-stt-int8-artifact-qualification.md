# STT INT8 Artifact Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, fail-closed evaluation harness that decides whether the stock Parakeet v2 and v3 INT8 ONNX artifacts qualify as candidates relative to their F32 references and faster-whisper base/int8.

**Architecture:** Keep the evaluator outside the production package in a small `scripts.stt_eval` package. A strict manifest layer resolves an immutable experiment, a preparation layer verifies and publishes the corpus, isolated child workers emit raw model observations, and a deterministic reporting layer computes scores, confidence intervals, and decisions. All inference uses verified local artifacts; only the explicit corpus-preparation command may download declared sources.

**Tech Stack:** Python 3.11+, Pydantic v2, standard-library archive/audio/process primitives, `httpx`, `psutil`, `onnx-asr==0.12.0`, ONNX Runtime CPU, faster-whisper, pytest.

---

## Scope and decision check

This is one evaluation subsystem with one deliverable: conclusive macOS
qualification evidence. It intentionally does not modify production routing,
provider registration, the model browser, or legacy providers.

**ADR required:** yes

**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

**Reason:** ADR-025 already fixes the artifact candidates, comparison baselines,
thresholds, local-only runtime boundary, VAD behavior, and promotion gates. This
plan implements that decision and does not introduce a new architecture choice.

**Approved design:** `Docs/superpowers/specs/2026-07-27-stt-int8-artifact-qualification-design.md`

## File map

### Evaluation package

- `scripts/stt_eval/__init__.py` — package version and public entry points.
- `scripts/stt_eval/__main__.py` — `python -m scripts.stt_eval` entry point.
- `scripts/stt_eval/schema.py` — strict manifest/result models, canonical JSON,
  and experiment/run fingerprints.
- `scripts/stt_eval/io.py` — streaming SHA-256, verified files, atomic JSON/JSONL
  publication, and validated path helpers.
- `scripts/stt_eval/prepare.py` — dry-run preflight, bounded archive extraction,
  audio normalization, deterministic derived fixtures, and corpus receipt.
- `scripts/stt_eval/normalization.py` — versioned language normalization and
  tokenization.
- `scripts/stt_eval/metrics.py` — edit counts, corpus WER/CER, clustered paired
  bootstrap, and scored-sample audit records.
- `scripts/stt_eval/gates.py` — closed-matrix completeness and fail-closed
  artifact/language decisions.
- `scripts/stt_eval/adapters.py` — local-only Parakeet ONNX and faster-whisper
  adapters behind one minimal protocol.
- `scripts/stt_eval/worker.py` — one-model child process and raw JSONL protocol.
- `scripts/stt_eval/resources.py` — Darwin high-water normalization and sampled
  process-tree RSS.
- `scripts/stt_eval/runner.py` — isolated quality, throughput, and memory/reuse
  profile orchestration.
- `scripts/stt_eval/report.py` — run validation, scoring, deterministic report
  generation, and human-readable summary.
- `scripts/stt_eval/cli.py` — `validate`, `prepare`, `verify-models`, `run`, and
  `report` commands.

### Versioned inputs and evidence

- `scripts/stt_eval/manifests/corpus-v1.json` — pinned FLEURS, Common Voice, and
  derived-fixture population.
- `scripts/stt_eval/manifests/models-v1.json` — exact v2/v3 INT8/F32,
  faster-whisper, and VAD files.
- `scripts/stt_eval/manifests/experiment-v1.json` — closed matrix, profiles,
  thresholds, minimums, seeds, and runtime settings.
- `Tests/STT_Eval/fixtures/` — tiny manifests, archives, fake worker, and WAV
  fixtures.
- `Tests/STT_Eval/normalization-golden-v1.json` — auditable examples for English
  and every proposed v3 language.
- `Docs/STT_Evaluation/task-593/README.md` — exact acquisition and reproduction
  procedure.
- `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/` — reviewed
  environment, raw/scored records, machine decision, and Markdown report.

### Tests

- `Tests/STT_Eval/test_schema.py`
- `Tests/STT_Eval/test_prepare.py`
- `Tests/STT_Eval/test_normalization.py`
- `Tests/STT_Eval/test_metrics.py`
- `Tests/STT_Eval/test_gates.py`
- `Tests/STT_Eval/test_worker.py`
- `Tests/STT_Eval/test_runner.py`
- `Tests/STT_Eval/test_report_cli.py`

## Task 1: Strict manifests and stable identities

**Files:**

- Create: `scripts/stt_eval/__init__.py`
- Create: `scripts/stt_eval/schema.py`
- Create: `scripts/stt_eval/io.py`
- Create: `Tests/STT_Eval/test_schema.py`
- Create: `Tests/STT_Eval/fixtures/minimal-experiment.json`

- [ ] **Step 1: Write failing strict-schema tests**

Cover unknown fields and versions, unsafe filenames, missing artifact/VAD
digests, incomplete matrix cells, and the exact 24-language v3 set.

```python
def test_manifest_rejects_unknown_fields() -> None:
    raw = minimal_experiment()
    raw["surprise"] = True
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_v3_matrix_requires_every_language_and_profile() -> None:
    raw = minimal_experiment()
    raw["matrix"] = [
        cell for cell in raw["matrix"] if cell["language"] != "uk"
    ]
    with pytest.raises(ValueError, match="closed comparison matrix"):
        ExperimentManifest.model_validate(raw)
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_schema.py -v
```

Expected: FAIL because `scripts.stt_eval.schema` does not exist.

- [ ] **Step 3: Implement strict Pydantic models and canonical JSON**

Use `ConfigDict(extra="forbid", frozen=True)` on every persisted model. Validate
filenames as single relative path components, SHA-256 as 64 lowercase hex
characters, sizes/counts/durations as positive bounded values, and measurement
profiles as `quality|throughput|memory_reuse`.

```python
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactFile(StrictModel):
    filename: str
    size_bytes: PositiveInt
    sha256: str

    @field_validator("filename")
    @classmethod
    def filename_is_contained(cls, value: str) -> str:
        if Path(value).name != value or value in {".", ".."}:
            raise ValueError("artifact filename must be one contained component")
        return value
```

Serialize fingerprints with UTF-8, sorted keys, compact separators, and no
NaN. Build the experiment fingerprint from the resolved corpus/model/manifold,
normalizer revision, environment, thresholds, seed, and harness revision. Build
the run fingerprint from the experiment fingerprint plus variant, profile, and
effective settings.

```python
def canonical_json(value: BaseModel | Mapping[str, object]) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
```

- [ ] **Step 4: Add verified I/O and atomic writers**

Implement streaming size/SHA-256 verification, `open("x")` staging writes, and
same-directory `Path.replace()` publication. Never follow symlinks and never
accept a path outside its declared root.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/stt_eval/__init__.py scripts/stt_eval/schema.py \
  scripts/stt_eval/io.py Tests/STT_Eval/test_schema.py \
  Tests/STT_Eval/fixtures/minimal-experiment.json
git commit -m "feat(stt-eval): add strict qualification manifests"
```

## Task 2: Safe, deterministic corpus preparation

**Files:**

- Create: `scripts/stt_eval/prepare.py`
- Create: `Tests/STT_Eval/test_prepare.py`
- Create: `Tests/STT_Eval/fixtures/fake_ffmpeg.py`
- Modify: `scripts/stt_eval/schema.py`
- Modify: `scripts/stt_eval/io.py`

- [ ] **Step 1: Write failing preflight and archive-boundary tests**

Test that dry-run lists licenses, source bytes, local inputs, destination, and
free-space requirements without network or extraction. Test rejection of path
traversal, absolute paths, symlinks, devices/FIFOs, unknown and duplicate
members, too many members, one oversized member, cumulative expansion overflow,
digest mismatch, and insufficient staging space.

```python
def test_extract_rejects_cumulative_expansion_before_publish(tmp_path: Path) -> None:
    archive = make_tar(tmp_path, {"a.wav": b"a" * 8, "b.wav": b"b" * 8})
    source = source_descriptor(
        archive, members=["a.wav", "b.wav"], max_uncompressed_bytes=12
    )
    with pytest.raises(PreparationError, match="uncompressed byte limit"):
        prepare_source(source, archive, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()
```

- [ ] **Step 2: Run preparation tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_prepare.py -v
```

Expected: FAIL because preparation functions do not exist.

- [ ] **Step 3: Implement dry-run and explicit execution**

`preflight()` is read-only. `prepare(..., execute=False)` returns the preflight;
only `execute=True` may transfer or extract. Common Voice always requires a
user-supplied archive matching the manifest. FLEURS downloads only the pinned
declared archive URL and only in explicit execution mode.

```python
@dataclass(frozen=True)
class PreparationPreflight:
    transfer_bytes: int
    staging_bytes: int
    required_local_inputs: tuple[Path, ...]
    licenses: tuple[str, ...]


def prepare(request: PrepareRequest, *, execute: bool = False) -> Path | PreparationPreflight:
    plan = preflight(request)
    if not execute:
        return plan
    return _execute_preparation(request, plan)
```

- [ ] **Step 4: Implement bounded streaming extraction**

First scan archive metadata and reject any name/type/count/declared-size
mismatch. During streaming extraction, independently count each output and the
cumulative bytes, recheck free space, and abort before a limit is exceeded.
Open outputs exclusively and publish only the final verified corpus directory.

- [ ] **Step 5: Implement normalized audio and derived fixtures**

Invoke a user-visible `ffmpeg` executable with an argument vector, never a
shell, to produce PCM signed 16-bit, 16 kHz, mono WAV. Record the exact ffmpeg
version in the corpus receipt. Generate seeded noise, silence, beyond-limit,
ten-minute, and long-form concatenations from declared sample IDs. Verify every
output size/digest against the manifest before publication.

- [ ] **Step 6: Run preparation tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_prepare.py -v
```

Expected: PASS with no network access.

- [ ] **Step 7: Commit**

```bash
git add scripts/stt_eval/prepare.py scripts/stt_eval/schema.py \
  scripts/stt_eval/io.py Tests/STT_Eval/test_prepare.py \
  Tests/STT_Eval/fixtures/fake_ffmpeg.py
git commit -m "feat(stt-eval): prepare bounded immutable corpus"
```

## Task 3: Auditable normalization and sample scoring

**Files:**

- Create: `scripts/stt_eval/normalization.py`
- Create: `scripts/stt_eval/metrics.py`
- Create: `Tests/STT_Eval/test_normalization.py`
- Create: `Tests/STT_Eval/normalization-golden-v1.json`
- Create: `Tests/STT_Eval/test_metrics.py`

- [ ] **Step 1: Add one golden normalization fixture per evaluated language**

The fixture covers `en` plus:

`bg hr cs da nl et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk`.

Each row includes raw reference/hypothesis, expected Unicode-normalized text,
case/punctuation/whitespace behavior, word units, character units, and edit
counts. Keep language-specific apostrophes and letters; do not transliterate.

- [ ] **Step 2: Write failing normalization and corpus-rate tests**

```python
@pytest.mark.parametrize("case", load_golden_cases(), ids=lambda row: row["language"])
def test_normalization_golden(case: dict[str, object]) -> None:
    scored = score_pair(case["language"], case["reference"], case["hypothesis"])
    assert scored.normalized_reference == case["normalized_reference"]
    assert scored.word_reference_units == case["word_reference_units"]
    assert scored.word_edits == case["word_edits"]
    assert scored.character_edits == case["character_edits"]


def test_corpus_rate_sums_edits_and_units() -> None:
    assert corpus_rate([(1, 2), (1, 8)]) == pytest.approx(0.2)
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_normalization.py \
  Tests/STT_Eval/test_metrics.py -v
```

Expected: FAIL because normalization and scoring are absent.

- [ ] **Step 4: Implement the versioned normalizer**

Use NFKC, `casefold()`, an explicit Unicode punctuation table/policy, and
whitespace collapse. Compute `normalizer_hash` from canonical JSON containing
the normalizer ID, revision, Unicode form, punctuation policy, unitization, and
per-language overrides.

```python
def normalize(text: str, language: str) -> NormalizedText:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = normalize_punctuation(value, language)
    value = " ".join(value.split())
    return NormalizedText(
        text=value,
        word_units=tuple(value.split()) if value else (),
        character_units=tuple(ch for ch in value if not ch.isspace()),
    )
```

- [ ] **Step 5: Implement deterministic edit counts and scored audit records**

Use a standard two-row Levenshtein dynamic program. Preserve raw and normalized
reference/hypothesis, normalizer ID/hash, word and character units/counts,
edit counts, sample ID, language, cluster ID, and gate populations.

- [ ] **Step 6: Run normalization and metric tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_normalization.py \
  Tests/STT_Eval/test_metrics.py -v
```

Expected: PASS for all 25 language fixtures.

- [ ] **Step 7: Commit**

```bash
git add scripts/stt_eval/normalization.py scripts/stt_eval/metrics.py \
  Tests/STT_Eval/test_normalization.py Tests/STT_Eval/test_metrics.py \
  Tests/STT_Eval/normalization-golden-v1.json
git commit -m "feat(stt-eval): add auditable multilingual scoring"
```

## Task 4: Clustered confidence intervals and fail-closed gates

**Files:**

- Create: `scripts/stt_eval/gates.py`
- Modify: `scripts/stt_eval/metrics.py`
- Create: `Tests/STT_Eval/test_gates.py`
- Modify: `Tests/STT_Eval/test_metrics.py`

- [ ] **Step 1: Write failing paired-bootstrap tests**

Cover a fixed 10,000-replicate result, common cluster resampling across both
arms, speaker-level grouping, missing-pair incompleteness, candidate-minus-
baseline direction, per-language v3 results, WER/CER macro separation, and
upper-97.5-percentile gating.

```python
def test_bootstrap_preserves_candidate_minus_baseline_direction() -> None:
    result = paired_cluster_bootstrap(
        candidate=records(rate=0.20),
        baseline=records(rate=0.10),
        metric="wer",
        seed=593,
        replicates=10_000,
    )
    assert result.point_delta == pytest.approx(0.10)
    assert result.upper_97_5 > 0
```

- [ ] **Step 2: Write failing decision-order tests**

Test:

- any missing cell yields `incomplete`;
- zero surviving v3 languages yields conclusive `fail` and macro
  `not_applicable`;
- one failed language is excluded without choosing F32;
- any global failure blocks all surviving languages;
- a non-empty passing survivor set yields `pass`;
- v2 has no subset behavior; and
- `semantic_default_eligible` is always false.

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_metrics.py \
  Tests/STT_Eval/test_gates.py -v
```

Expected: FAIL because bootstrap and gate evaluation are absent.

- [ ] **Step 4: Implement clustered paired bootstrap**

Sort cluster IDs before seeded sampling. Each replicate samples the same
clusters for candidate and baseline, includes every member of selected
clusters, recomputes summed edits/summed reference units, then records
candidate-minus-baseline. Compute v3 macro replicates by resampling inside each
surviving language and taking an unweighted mean only across one primary-metric
family.

- [ ] **Step 5: Implement explicit thresholds and decision order**

Represent points as rates:

```python
V2_ENGLISH_AGGREGATE = Decimal("0.010")
V2_ENGLISH_SLICE = Decimal("0.030")
V3_LANGUAGE = Decimal("0.040")
V3_MACRO = Decimal("0.015")
INT8_VS_F32 = Decimal("0.005")
MAX_RSS_BYTES = 3 * 1024**3
MAX_REUSE_GROWTH = Decimal("0.15")
```

Both the point delta and adverse confidence bound must satisfy quality
thresholds. Evaluate decisions in the order fixed by the design; never replace
a failed INT8 variant with F32.

- [ ] **Step 6: Run gate tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_metrics.py \
  Tests/STT_Eval/test_gates.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/stt_eval/metrics.py scripts/stt_eval/gates.py \
  Tests/STT_Eval/test_metrics.py Tests/STT_Eval/test_gates.py
git commit -m "feat(stt-eval): evaluate confidence-bound artifact gates"
```

## Task 5: Local-only model adapters and raw child protocol

**Files:**

- Create: `scripts/stt_eval/adapters.py`
- Create: `scripts/stt_eval/worker.py`
- Create: `Tests/STT_Eval/test_worker.py`
- Create: `Tests/STT_Eval/fixtures/fake_worker.py`
- Modify: `scripts/stt_eval/schema.py`
- Modify: `scripts/stt_eval/io.py`

- [ ] **Step 1: Write failing local-artifact verification tests**

Require every declared model and VAD file to be a regular non-symlink file with
the exact size and SHA-256 at each required verification boundary. Cover
same-inode mutation during descriptor-backed consumption and path replacement
around native path-only model loading. Verify that neither adapter receives a
repository ID or enables runtime downloads.

- [ ] **Step 2: Write failing worker-protocol tests**

Cover one model load, fixed sample order, exact sample count, timestamps,
warnings, timings, runtime/model identity, requested/effective VAD
`batch_size=1`, terminal completeness, child crash, timeout, malformed output,
and partial JSONL rejection.

```python
def test_worker_loads_once_and_emits_terminal_record(tmp_path: Path) -> None:
    records = run_fake_worker(tmp_path, sample_count=3)
    assert [row["sample_id"] for row in records[:-1]] == ["s1", "s2", "s3"]
    assert records[-1] == {
        "record_type": "terminal",
        "status": "complete",
        "sample_count": 3,
        "model_load_count": 1,
    }
```

- [ ] **Step 3: Run worker tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_worker.py -v
```

Expected: FAIL because adapters and worker do not exist.

- [ ] **Step 4: Implement the adapter protocol and faster-whisper adapter**

```python
class STTEvalAdapter(Protocol):
    def transcribe(
        self,
        waveform: npt.NDArray[np.float32],
        *,
        sample_rate: int,
        long_form: bool,
    ) -> Observation: ...
    def close(self) -> None: ...
```

For any consumer API that accepts an open descriptor or binary stream, keep the
swap-resistant verified-read context open for the complete consumption
operation and accept its output only after the context's post-consumption
content/stat validation finishes successfully. Native model loaders require
paths, so retain each artifact's pre-load identity token, revalidate the exact
artifact content after model load and again after the run, and mark the run
incomplete if either check detects a change. These checks detect integrity
changes; they are not path authorization or a lease. Do not add a snapshot
service in this task.

Construct `faster_whisper.WhisperModel` from the declared local directory with
CPU, the manifest's thread count, `compute_type="int8"`, and
`local_files_only=True`. Retain and revalidate the model files' pre-load
identity tokens at both required boundaries. Pass the already-decoded float32
waveform rather than a path, materialize all segments inside the timed
boundary, and standardize segment timestamps.

- [ ] **Step 5: Implement the onnx-asr v2/v3 adapter**

Call pinned `onnx_asr.load_model` with the declared local directory,
`quantization="int8"` or `None` for F32, CPU execution provider, and the exact
thread/preprocessor settings from the manifest. For long-form records, create
the pinned VAD wrapper with the declared local VAD path and explicitly pass
`batch_size=1`; record requested and observed effective values. Retain the
pre-load identity tokens for every model and VAD file, revalidate their exact
content after loading and after the run, and make any detected change
incomplete. Do not use a model name that can invoke the resolver when a local
path is missing.

- [ ] **Step 6: Implement the JSONL child protocol**

The parent sends one fixed run request file. The child loads one model and
decodes each prepared WAV outside the recognition timing boundary. Descriptor-
capable WAV consumers must remain inside the swap-resistant read context until
its post-consumption validation succeeds. The worker retains path-only
model/VAD identity tokens through loading and the full run, performs the
required post-load and post-run exact-content revalidations, and emits
`incomplete` instead of accepting observations if any identity changed. This
detects integrity changes and does not turn a path into authorization or a
lease. The child emits one raw sample record at a time, flushes it, then emits
exactly one terminal record. On any exception it emits a sanitized terminal
failure when possible and exits nonzero. The parent publishes only a complete
validated temporary JSONL file.

- [ ] **Step 7: Run worker tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_worker.py -v
```

Expected: PASS using fake adapters only; no production model is loaded.

- [ ] **Step 8: Commit**

```bash
git add scripts/stt_eval/adapters.py scripts/stt_eval/worker.py \
  scripts/stt_eval/schema.py scripts/stt_eval/io.py \
  Tests/STT_Eval/test_worker.py Tests/STT_Eval/fixtures/fake_worker.py
git commit -m "feat(stt-eval): run verified local models in child process"
```

## Task 6: Isolated measurement profiles

**Files:**

- Create: `scripts/stt_eval/resources.py`
- Create: `scripts/stt_eval/runner.py`
- Create: `Tests/STT_Eval/test_runner.py`
- Modify: `scripts/stt_eval/worker.py`
- Modify: `scripts/stt_eval/schema.py`

- [ ] **Step 1: Write failing profile-separation tests**

Prove that:

- `quality` runs the full declared population without the 10 ms sampler;
- `throughput` runs one warm-up and exactly three timed passes without the
  sampler, stores every pass, and gates their median;
- `memory_reuse` enables the sampler, loads once, processes warm-up plus 100
  files, preserves timing only as diagnostic, and computes idle-window medians;
- profile identity changes only the run fingerprint, not the experiment
  fingerprint.

- [ ] **Step 2: Write failing RSS tests**

Cover Darwin `ru_maxrss` byte conversion, descendant rediscovery, maximum
process-tree sum, missing required probes, worker PID changes, model-load counts,
and the 15% post-warm-up growth formula.

- [ ] **Step 3: Run runner tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_runner.py -v
```

Expected: FAIL because runner/resource modules do not exist.

- [ ] **Step 4: Implement process-tree sampling**

Sample the worker and all descendants every 10 ms using `time.monotonic()` and
`psutil.Process.children(recursive=True)`. Sum RSS for live processes on every
sample and retain the maximum. Record sampling gaps and child high-water RSS;
the gate uses the larger. Missing either required value means `incomplete`.

- [ ] **Step 5: Implement profile orchestration**

Use `sys.executable -m scripts.stt_eval.worker` with no shell. Each profile gets
a fresh child. Throughput starts timing at normalized-waveform submission and
stops after result materialization. Preserve three inverse real-time factors
and use `statistics.median()` for the gate. The same predecoded waveform objects
are reused across the three passes. Resource sampling is never created for
quality or throughput.

- [ ] **Step 6: Implement memory/reuse accounting**

Capture one-second 10 ms idle windows after warm-up and after 100 files. Require
the same worker PID and `model_load_count == 1`; compute:

```python
growth = (post_run_median - baseline_median) / baseline_median
```

Keep the maximum across load, warm-up, all files, and both idle windows.

- [ ] **Step 7: Run runner tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_runner.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/stt_eval/resources.py scripts/stt_eval/runner.py \
  scripts/stt_eval/worker.py scripts/stt_eval/schema.py \
  Tests/STT_Eval/test_runner.py
git commit -m "feat(stt-eval): isolate quality throughput and memory runs"
```

## Task 7: Deterministic reporting and CLI

**Files:**

- Create: `scripts/stt_eval/report.py`
- Create: `scripts/stt_eval/cli.py`
- Create: `scripts/stt_eval/__main__.py`
- Create: `Tests/STT_Eval/test_report_cli.py`
- Modify: `scripts/stt_eval/schema.py`
- Modify: `scripts/stt_eval/io.py`

- [ ] **Step 1: Write failing report-completeness tests**

Test matching experiment fingerprints, correct variant/profile run
fingerprints, terminal records, every matrix cell, minimum samples/reference
units/duration, exact pairing, explicit unsupported timestamps, silence,
long-form VAD evidence, and macOS-only platform labels.

- [ ] **Step 2: Write failing deterministic-output tests**

Run report generation twice from the same fake raw records and assert
byte-identical scored JSONL, `decision.json`, `environment.json`, and
`report.md`. Assert every scored row contains the approved audit fields and the
top-level decision says `semantic_default_eligible: false`.

- [ ] **Step 3: Write failing CLI safety tests**

Cover:

```text
validate MANIFEST
prepare MANIFEST --destination PATH [--common-voice-archive PATH] [--execute]
verify-models MANIFEST --model-root ID=PATH ...
run MANIFEST --prepared-corpus PATH --model-root ID=PATH --output PATH
report MANIFEST --runs PATH --output PATH
```

Missing `--execute` must remain a dry run. `run` and `report` reject incomplete
or existing incompatible destinations instead of overwriting them.

- [ ] **Step 4: Run report/CLI tests and verify they fail**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_report_cli.py -v
```

Expected: FAIL because reporting and CLI modules do not exist.

- [ ] **Step 5: Implement report validation and scoring**

Load only complete raw runs with matching identities. Score quality records,
validate timestamps/silence/performance/memory/reuse evidence, execute the
closed matrix and decision order, and write all outputs to same-directory
staging before atomic publication.

- [ ] **Step 6: Implement machine and Markdown reports**

The machine decision contains `pass|fail|incomplete` per artifact, v3 language,
matrix cell, and gate; candidate and excluded language sets; exact deltas and
intervals; and outstanding release gates. The Markdown file summarizes without
claiming cross-platform or semantic-default promotion.

- [ ] **Step 7: Implement the CLI**

Use `argparse`, explicit subcommands, nonzero exit codes for invalid/incomplete
evidence, and sanitized error messages. Keep all commands local-only except
explicit FLEURS transfer in `prepare --execute`.

- [ ] **Step 8: Run report/CLI tests**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_report_cli.py -v
python3 -m scripts.stt_eval --help
```

Expected: tests PASS and help lists all five subcommands.

- [ ] **Step 9: Commit**

```bash
git add scripts/stt_eval/report.py scripts/stt_eval/cli.py \
  scripts/stt_eval/__main__.py scripts/stt_eval/schema.py \
  scripts/stt_eval/io.py Tests/STT_Eval/test_report_cli.py
git commit -m "feat(stt-eval): add fail-closed qualification reports"
```

## Task 8: Curate immutable production manifests and reproduction docs

**Files:**

- Create: `scripts/stt_eval/manifests/corpus-v1.json`
- Create: `scripts/stt_eval/manifests/models-v1.json`
- Create: `scripts/stt_eval/manifests/experiment-v1.json`
- Create: `Docs/STT_Evaluation/task-593/README.md`
- Modify: `Tests/STT_Eval/test_schema.py`

- [ ] **Step 1: Write a failing production-manifest contract test**

Assert no placeholder/zero digest exists; all 24 v3 languages and all three
profiles are closed; every WER- or CER-primary population declares
sample/unit/duration minimums; all model/VAD files have repository, immutable
revision, license, size, and SHA-256; Python, ffmpeg, ONNX Runtime, onnx-asr,
faster-whisper, CTranslate2, execution-provider, and thread settings are pinned;
bootstrap is 10,000 with a fixed seed; and every approved threshold is exact.

- [ ] **Step 2: Run the contract test and verify it fails**

Run:

```bash
python3 -m pytest Tests/STT_Eval/test_schema.py \
  -k production_manifest -v
```

Expected: FAIL because production manifests do not exist.

- [ ] **Step 3: Pin the corpus**

Pin one immutable FLEURS repository commit and its declared test archives for
English and all 24 v3 languages. Select sample IDs before inference, with the
approved deep slices for Spanish, French, German, Polish, Greek, Russian, and
Ukrainian. Pin one Mozilla Data Collective Common Voice English archive for
accent evidence and record its user-supplied filename, size, digest, license,
and member allowlist. Add deterministic clean/noisy/silence/beyond-limit/
ten-minute/long-form recipes and cluster IDs.

Document the one-time curation ledger used before finalizing the manifest:
verify each complete source archive first; derive the member allowlist and fixed
sample IDs only from upstream metadata; run the same bounded extractor,
normalizer, and derived-fixture code in an isolated curation directory; record
the exact command, tool versions, member counts, prepared sizes, and observed
SHA-256 values; then transcribe those values into the versioned manifest.
Delete the curation output and prove the finalized manifest by preparing again
from an empty destination. Only the second, strict hash-matching preparation is
eligible for qualification evidence.

- [ ] **Step 4: Pin model and VAD artifacts**

Record exact files, sizes, and SHA-256 for:

- stock `istupakov/parakeet-tdt-0.6b-v2-onnx` INT8 and F32 at one commit;
- stock `istupakov/parakeet-tdt-0.6b-v3-onnx` INT8 and F32 at one commit;
- faster-whisper base CTranslate2 int8 baseline at one commit; and
- the exact local Silero VAD bundle used by `onnx-asr==0.12.0`.

Compute SHA-256 from downloaded bytes and independently compare repository
metadata/size before entering values. Do not use the community SmoothQuant
artifact in this task.

- [ ] **Step 5: Build the closed experiment manifest**

Declare CPU provider/thread settings, normalizer revision, matrix minimums,
quality/throughput/memory populations, timeout, VAD `batch_size=1`, 10 ms RSS
sampling, three throughput passes, 100-file reuse, all thresholds, and fixed
bootstrap seed.

- [ ] **Step 6: Document exact acquisition and commands**

Explain licenses, Mozilla archive acquisition, expected disk use, model
artifact acquisition, local directory mapping, dry-run, explicit execution,
run, report, output interpretation, and cleanup. State that a pass is only
TASK-593 evidence and cannot promote defaults or remove providers.

- [ ] **Step 7: Validate manifests and tests**

Run:

```bash
python3 -m scripts.stt_eval validate \
  scripts/stt_eval/manifests/experiment-v1.json
python3 -m pytest Tests/STT_Eval -v
git diff --check
```

Expected: manifest validation and all evaluator tests PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/stt_eval/manifests Tests/STT_Eval/test_schema.py \
  Docs/STT_Evaluation/task-593/README.md
git commit -m "docs(stt-eval): pin qualification corpus and artifacts"
```

## Task 9: Produce and review macOS qualification evidence

**Files:**

- Create: `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/environment.json`
- Create: `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/raw/*.jsonl`
- Create: `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/scored/*.jsonl`
- Create: `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/decision.json`
- Create: `Docs/STT_Evaluation/task-593/macos-<experiment-prefix>/report.md`
- Modify: `Docs/STT_Evaluation/task-593/README.md`

- [ ] **Step 1: Verify the reference environment**

Record macOS version, architecture, CPU, memory, Python, ffmpeg, ONNX Runtime,
onnx-asr, faster-whisper/CTranslate2, execution provider, thread settings, and
harness commit. Stop if the resolved environment differs across variant runs.

- [ ] **Step 2: Run corpus preflight without transfer**

Run:

```bash
python3 -m scripts.stt_eval prepare \
  scripts/stt_eval/manifests/experiment-v1.json \
  --destination /ABSOLUTE/PATH/task-593-corpus \
  --common-voice-archive /ABSOLUTE/PATH/common-voice.tar.gz
```

Expected: a dry-run listing licenses, exact transfer/input/staging bytes,
destination, and available space. No file is downloaded or extracted.

- [ ] **Step 3: Obtain explicit approval for the displayed transfer**

Do not continue from the dry-run to `--execute` implicitly. Present the exact
download and staging totals and wait for approval of that preparation command.

- [ ] **Step 4: Prepare and verify the corpus**

After approval, rerun the exact preflight command with `--execute`. Verify the
atomic prepared-corpus receipt and every sample digest.

- [ ] **Step 5: Verify all local model roots**

Run:

```bash
python3 -m scripts.stt_eval verify-models \
  scripts/stt_eval/manifests/experiment-v1.json \
  --model-root parakeet-v2=/ABSOLUTE/PATH/v2 \
  --model-root parakeet-v3=/ABSOLUTE/PATH/v3 \
  --model-root faster-whisper-base=/ABSOLUTE/PATH/whisper-base \
  --model-root silero-vad=/ABSOLUTE/PATH/silero-vad
```

Expected: all required INT8/F32/baseline/VAD files PASS size and SHA-256
verification; no network is used.

- [ ] **Step 6: Run all declared variants and profiles**

Use one output root and the same resolved experiment fingerprint for all runs.
Do not combine or repair mismatched runs. A failed child/profile remains
incomplete until the same declared run is repeated successfully.

- [ ] **Step 7: Generate the report**

Run:

```bash
python3 -m scripts.stt_eval report \
  scripts/stt_eval/manifests/experiment-v1.json \
  --runs /ABSOLUTE/PATH/task-593-runs \
  --output /ABSOLUTE/PATH/task-593-report
```

Expected: every matrix cell is conclusively `pass` or `fail`; no required cell
is `incomplete`.

- [ ] **Step 8: Copy the deterministic evidence into the repository**

Copy only manifests, environment, raw/scored JSONL, decisions, and reports.
Never copy model files, downloaded archives, prepared audio, credentials, or
unnecessary absolute local paths. Re-run report generation against the copied
raw records and assert byte-identical outputs.

- [ ] **Step 9: Review the decision**

Confirm:

- v2 and v3 artifact decisions follow the documented order;
- failed v3 languages are excluded individually;
- global failures block the artifact;
- F32 is comparison-only;
- `semantic_default_eligible=false`;
- platform scope is macOS only; and
- a conclusive fail is reported plainly rather than massaged into a pass.

- [ ] **Step 10: Commit**

```bash
git add Docs/STT_Evaluation/task-593
git commit -m "test(stt): record macOS INT8 qualification evidence"
```

## Task 10: Final verification and Backlog completion

**Files:**

- Modify: `backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md`
- Modify if required: `Docs/STT_Evaluation/task-593/README.md`

- [ ] **Step 1: Run the complete evaluator suite**

Run:

```bash
python3 -m pytest Tests/STT_Eval -v
```

Expected: PASS.

- [ ] **Step 2: Run adjacent STT regression tests**

Run:

```bash
python3 -m pytest \
  Tests/Transcription/test_stt_batch_routing.py \
  Tests/Transcription/test_parakeet_onnx_vertical_slice.py \
  Tests/Local_Ingestion/test_parakeet_v2_installer.py -v
```

Expected: PASS; evaluation work did not alter production behavior.

- [ ] **Step 3: Run static and repository checks**

Run:

```bash
python3 -m compileall -q scripts/stt_eval
python3 -m mypy scripts/stt_eval
git diff --check
git status --short
```

Expected: compile, type, and whitespace checks PASS; status contains only
intentional TASK-593 changes.

- [ ] **Step 4: Perform a self-review**

Review the complete branch diff against all six acceptance criteria and
ADR-025. Search for downloads in `run`/`report`, placeholder digests, silent
fallback, production routing changes, absolute local paths, secrets, and
unsupported platform claims.

- [ ] **Step 5: Update TASK-593**

Only when the real report has no `incomplete` required cells:

- check all acceptance criteria;
- add concise implementation notes with the artifact/language outcomes,
  modified files, tests, evidence directory, ADR link, and any plan deviations;
- set status to `Done`.

A conclusive `fail` is a valid completed qualification result. Any
`incomplete` cell keeps the task `In Progress`.

- [ ] **Step 6: Commit task completion**

```bash
git add 'backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md' \
  Docs/STT_Evaluation/task-593/README.md
git commit -m "docs(backlog): complete TASK-593 qualification"
```

- [ ] **Step 7: Request code review before PR/merge**

Use `superpowers:requesting-code-review`, address every actionable issue, rerun
the affected verification, then create a PR against `dev`. Do not merge with an
incomplete required evidence cell.
