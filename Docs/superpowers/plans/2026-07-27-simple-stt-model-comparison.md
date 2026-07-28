# Simple STT Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one indicative macOS WER/CER and runtime comparison of stock Parakeet v2/v3 INT8, their F32 forms, and faster-whisper.

**Architecture:** Delete the abandoned qualification framework and replace it with one standard-library-heavy helper script. The script reads local WAV cases, loads each model sequentially from an explicit local directory, records every scheduled result or error, and atomically writes one diagnostic JSON report.

**Tech Stack:** Python 3.11+, `onnx-asr==0.12.0`, ONNX Runtime CPU, faster-whisper, standard-library JSON/WAV/Unicode/timing helpers, pytest.

---

## Scope check

This plan has one deliverable and no reusable evaluator architecture. Reusable
multi-engine evaluation is deferred to TASK-1023.

**ADR required:** no

**ADR path:** N/A

**Reason:** The script produces diagnostic evidence and does not change routing,
artifact ownership, storage, or runtime contracts.

**Approved design:** `Docs/superpowers/specs/2026-07-27-simple-stt-model-comparison-design.md`

For this worktree, use the absolute interpreter path in every command because
shell variables do not persist between tool invocations:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

## File map

- Delete: `scripts/stt_eval/`
- Delete: `Tests/STT_Eval/`
- Delete: `Docs/superpowers/specs/2026-07-27-stt-int8-artifact-qualification-design.md`
- Delete: `Docs/superpowers/plans/2026-07-27-stt-int8-artifact-qualification.md`
- Create: `Docs/superpowers/plans/2026-07-27-simple-stt-model-comparison.md`
- Create: `Helper_Scripts/Benchmarks/stt_model_comparison.py`
- Create: `Tests/Helper_Scripts/test_stt_model_comparison.py`
- Create: `Docs/STT_Evaluation/task-593/cases.jsonl`
- Create: `Docs/STT_Evaluation/task-593/report.json`
- Create: `Docs/STT_Evaluation/task-593/README.md`
- Modify: `backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md`

## Task 1: Remove the abandoned qualification framework

- [ ] Delete only the branch-added `scripts/stt_eval`, `Tests/STT_Eval`, old
  qualification design, and old qualification plan.

- [ ] Verify that the approved simple design, this plan, TASK-593, and TASK-1023
  remain present.

- [ ] Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the intended deletions and retained
simple-eval planning files are visible.

- [ ] Commit:

```bash
git add scripts/stt_eval Tests/STT_Eval \
  Docs/superpowers/specs/2026-07-27-stt-int8-artifact-qualification-design.md \
  Docs/superpowers/plans/2026-07-27-stt-int8-artifact-qualification.md \
  Docs/superpowers/plans/2026-07-27-simple-stt-model-comparison.md \
  'backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md'
git commit -m "refactor(stt-eval): drop qualification framework"
```

## Task 2: Implement the one-file comparison

**Files:**

- Create: `Helper_Scripts/Benchmarks/stt_model_comparison.py`
- Create: `Tests/Helper_Scripts/test_stt_model_comparison.py`

- [ ] Write focused failing tests for:

```python
assert normalize_text(" Héllo—МИР! ") == "héllo мир"
assert edit_distance(["a", "b"], ["a", "x", "b"]) == 1
```

Also cover:

- JSONL-relative audio paths and duplicate case IDs;
- PCM signed 16-bit, 16 kHz, mono WAV validation;
- empty reference text only for `tag == "silence"`;
- rejection of missing paths, non-directories, and repository IDs for all five
  model inputs before the output is touched;
- the exact v2/v3/faster-whisper scheduling matrix;
- exact local loader arguments, including Parakeet path/quantization/provider
  and faster-whisper `local_files_only=True`;
- micro WER/CER aggregation;
- report label/environment, audio SHA-256, model filename/size identity,
  per-row elapsed/duration/RTF, aggregate RTF, and separate silence results;
- atomic report replacement;
- expansion of one model-load failure into an error row for every case
  scheduled for that model;
- continued fake-model execution after one transcription error; and
- nonzero completion status when the published report contains an error.

- [ ] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m pytest Tests/Helper_Scripts/test_stt_model_comparison.py -v
```

Expected: FAIL because the helper script does not exist.

- [ ] Implement the minimum script with these functions:

```python
normalize_text(text: str) -> str
edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int
load_cases(path: Path) -> list[dict[str, object]]
scheduled_models(case: dict[str, object]) -> tuple[str, ...]
run_comparison(cases, model_runners) -> tuple[list[dict], dict, bool]
write_report(path: Path, report: dict[str, object]) -> None
main(argv: Sequence[str] | None = None) -> int
```

Use normalized words for WER and normalized non-whitespace code points for CER.
Aggregate edit counts before division. Silence rows record normalized output but
do not contribute a zero-denominator WER/CER.

Load models sequentially so only one heavy model is resident:

```python
onnx_asr.load_model(
    "nemo-parakeet-tdt-0.6b-v2",
    path=v2_dir,
    quantization="int8",  # None for F32
    providers=["CPUExecutionProvider"],
)

WhisperModel(
    whisper_dir,
    device="cpu",
    compute_type="int8",
    local_files_only=True,
)
```

For faster-whisper, fully materialize segments and join their text inside the
timed call. Pass each non-silence case's language explicitly. Never accept a
repository ID in place of a directory.

Record model-load or transcription errors, continue other runnable models,
publish the diagnostic report, and return exit code 1 when any error exists.
Invalid cases or missing model directories fail before the output is touched.

- [ ] Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m pytest Tests/Helper_Scripts/test_stt_model_comparison.py -v
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m ruff check Helper_Scripts/Benchmarks/stt_model_comparison.py \
  Tests/Helper_Scripts/test_stt_model_comparison.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m ruff format --check Helper_Scripts/Benchmarks/stt_model_comparison.py \
  Tests/Helper_Scripts/test_stt_model_comparison.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  Helper_Scripts/Benchmarks/stt_model_comparison.py --help
```

Expected: tests and static checks pass; help lists the case, five model, and
output arguments.

- [ ] Commit:

```bash
git add Helper_Scripts/Benchmarks/stt_model_comparison.py \
  Tests/Helper_Scripts/test_stt_model_comparison.py
git commit -m "feat(stt-eval): add simple model comparison"
```

## Task 3: Run the indicative macOS comparison

**Files:**

- Create: `Docs/STT_Evaluation/task-593/cases.jsonl`
- Create: `Docs/STT_Evaluation/task-593/report.json`
- Create: `Docs/STT_Evaluation/task-593/README.md`
- Modify: `backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md`

- [ ] Create `.benchmarks/stt-task-593/` for uncommitted models and audio.
Record at least one FLEURS test utterance for `en` and each routed v3 language:

```text
bg hr cs da nl et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk
```

Use the FLEURS snapshot at
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Create an uncommitted
`/tmp/task593_fetch_fleurs.py` that streams the first test row for each entry
in this explicit route-to-FLEURS mapping:

```python
{
    "en": "en_us",
    "bg": "bg_bg", "hr": "hr_hr", "cs": "cs_cz", "da": "da_dk",
    "nl": "nl_nl", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr",
    "de": "de_de", "el": "el_gr", "hu": "hu_hu", "it": "it_it",
    "lv": "lv_lv", "lt": "lt_lt", "mt": "mt_mt", "pl": "pl_pl",
    "pt": "pt_br", "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si",
    "es": "es_419", "sv": "sv_se", "ru": "ru_ru", "uk": "uk_ua",
}
```

The temporary script must call `datasets.load_dataset("google/fleurs",
locale, split="test", streaming=True, revision=REVISION)`, cast `audio` to
`datasets.Audio(decode=False)`, take exactly one row, preserve its transcript
as the reference, and invoke ffmpeg to emit signed 16-bit, 16 kHz, mono WAV.
Run it without changing the project environment:

```bash
uv run --no-project --with datasets /tmp/task593_fetch_fleurs.py
```

Keep the acquisition script and source audio uncommitted.

- [ ] Add local derived cases:

- one additional synthetic English accent voice generated with macOS `say`;
- one noisy English case derived from a clean case;
- one 60-second-or-longer English concatenation with matching repeated
  reference text; and
- one silence WAV with an empty reference.

Record every final local case in
`Docs/STT_Evaluation/task-593/cases.jsonl` using paths relative to that file.
Do not commit the audio.

- [ ] Download the three pinned model snapshots into
`.benchmarks/stt-task-593/models/` with `huggingface_hub.snapshot_download`:

```text
istupakov/parakeet-tdt-0.6b-v2-onnx
  revision 0bbb45a3365852604aef28b538a8f066f4ccaa85

istupakov/parakeet-tdt-0.6b-v3-onnx
  revision 8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce

Systran/faster-whisper-base
  revision ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66
```

For each Parakeet snapshot, use this exact allowlist:

```text
config.json
vocab.txt
encoder-model.onnx
encoder-model.onnx.data
decoder_joint-model.onnx
encoder-model.int8.onnx
decoder_joint-model.int8.onnx
```

The external `encoder-model.onnx.data` file is required by the F32 encoder.

- [ ] Run the script once, pointing the INT8 and F32 arguments for a family at
the same pinned snapshot directory:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  Helper_Scripts/Benchmarks/stt_model_comparison.py \
  --cases Docs/STT_Evaluation/task-593/cases.jsonl \
  --v2-int8 .benchmarks/stt-task-593/models/parakeet-v2 \
  --v2-f32 .benchmarks/stt-task-593/models/parakeet-v2 \
  --v3-int8 .benchmarks/stt-task-593/models/parakeet-v3 \
  --v3-f32 .benchmarks/stt-task-593/models/parakeet-v3 \
  --faster-whisper .benchmarks/stt-task-593/models/faster-whisper-base \
  --output Docs/STT_Evaluation/task-593/report.json
```

Expected: one atomic report; exit 0 when all scheduled runs complete or exit 1
with error rows preserved.

- [ ] Write `README.md` with the pinned sources, exact command, result table,
observed failures, and a plain conclusion. State prominently that this is an
`indicative_macos_comparison` and does not open the production promotion gate.

- [ ] Run final verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m pytest Tests/Helper_Scripts/test_stt_model_comparison.py -v
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m ruff check Helper_Scripts/Benchmarks/stt_model_comparison.py \
  Tests/Helper_Scripts/test_stt_model_comparison.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m ruff format --check Helper_Scripts/Benchmarks/stt_model_comparison.py \
  Tests/Helper_Scripts/test_stt_model_comparison.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  -m json.tool Docs/STT_Evaluation/task-593/report.json >/dev/null
git diff --check
```

- [ ] Only after the real report exists, check all TASK-593 acceptance criteria,
add concise implementation notes with the observed model results, set the task
to Done, and commit:

```bash
git add Docs/STT_Evaluation/task-593 \
  'backlog/tasks/task-593 - Qualify-Parakeet-v2-and-v3-INT8-artifacts.md'
git commit -m "docs(stt-eval): record indicative macOS comparison"
```
