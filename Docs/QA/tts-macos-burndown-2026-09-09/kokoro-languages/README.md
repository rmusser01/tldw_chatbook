# Non-English Kokoro qualification — 9 September 2026

The latest observed tuple selection has **14/14 runtime passes**. It combines ten initial tuples with four repaired Japanese/French/Mandarin reruns; it does not imply all fourteen were rerun on the merged source. The repaired runs use captured revision `1edcbb1aee9bacfd7573c53f70b22d51b27f6300` with 2,265 source hashes, represented as 47 overrides/additions and no removals against the earlier manifest. Runtime success remains separate from the raw transcript comparisons below.

| Latest observed tuple | Source run | Runtime | Small ASR exact | Medium ASR exact |
|---|---|---|---|---|
| [pytorch CPU / es](runs/pytorch-cpu-es-final-01/evidence.json) | final-01 | passed | 2/2 | not run |
| [pytorch CPU / fr](runs/pytorch-cpu-fr-final-01/evidence.json) | final-01 | passed | 0/2 | 2/2 |
| [pytorch CPU / hi](runs/pytorch-cpu-hi-final-01/evidence.json) | final-01 | passed | 0/2 | 0/2 |
| [pytorch CPU / it](runs/pytorch-cpu-it-final-01/evidence.json) | final-01 | passed | 2/2 | not run |
| [pytorch CPU / pt-br](runs/pytorch-cpu-pt-br-final-01/evidence.json) | final-01 | passed | 1/2 | 2/2 |
| [pytorch CPU / ja](runs/pytorch-cpu-ja-final-02/evidence.json) | final-02 | passed | 0/2 | 0/2 |
| [pytorch CPU / zh](runs/pytorch-cpu-zh-final-01/evidence.json) | final-01 | passed | 0/2 | 0/2 |
| [onnx CPU / es](runs/onnx-cpu-es-final-01/evidence.json) | final-01 | passed | 2/2 | not run |
| [onnx CPU / fr](runs/onnx-cpu-fr-final-02/evidence.json) | final-02 | passed | 0/2 | 2/2 |
| [onnx CPU / hi](runs/onnx-cpu-hi-final-01/evidence.json) | final-01 | passed | 0/2 | 0/2 |
| [onnx CPU / it](runs/onnx-cpu-it-final-01/evidence.json) | final-01 | passed | 2/2 | not run |
| [onnx CPU / pt-br](runs/onnx-cpu-pt-br-final-01/evidence.json) | final-01 | passed | 2/2 | not run |
| [onnx CPU / ja](runs/onnx-cpu-ja-final-02/evidence.json) | final-02 | passed | 0/2 | 0/2 |
| [onnx CPU / zh](runs/onnx-cpu-zh-final-02/evidence.json) | final-02 | passed | 0/2 | 0/2 |

Counts are receipt-normalized matches over complete phase clips; raw differences remain unchanged. The latest selection contains 28 phase clips; preserving the superseded ONNX Japanese output makes 30 successful clips across all captured attempts. A second recognizer result is another view of existing audio, not another generated clip.

The latest small-ASR observations are **11/28 normalized-exact**. Completed medium ASR covers **9 tuples, 18 clips and 6 normalized-exact matches**. For one explicitly defined combined observation, select the latest runtime per tuple and use medium for the entire tuple whenever available, otherwise small: **16/28 normalized-exact**. This uses nine medium tuples and five small-only tuples, with no per-clip choice of the better result. Both recognizers remain visible in the table and raw receipts.

Medium ASR matches both complete French clips in each engine and both PyTorch Portuguese clips (2/2 per tuple). Mandarin in both engines produces complete Traditional-script text corresponding to the Simplified input; the strict verifier still reports `content_review_required`, with no script conversion applied. Hindi retains lexical/spelling differences, and Japanese retains lexical/homophone uncertainty. The repaired Japanese transcripts contain the requested Japanese content rather than the initial ONNX character descriptions; these comparisons do not replace native-language listening.

The initial fourteen CPU language tuples produced **eleven runtime passes and three failures**. All fourteen workers joined cleanup with zero recorded final owners. The successful tuples each completed Speech Lab file playback and one Console sink drain: **22 clips**, of which the captured multilingual Whisper-small receipts report **11 normalized-exact**, **7 raw-exact**, and **11 with all three anchors**. These are separate runtime and content outcomes; this baseline is not a blanket language-quality pass.

| Initial runtime tuple | Voice | Runtime | Small ASR normalized-exact clips |
|---|---|---|---|
| [pytorch CPU / es](runs/pytorch-cpu-es-final-01/evidence.json) | `ef_dora` | passed | 2/2 |
| [pytorch CPU / fr](runs/pytorch-cpu-fr-final-01/evidence.json) | `ff_siwis` | passed | 0/2 |
| [pytorch CPU / hi](runs/pytorch-cpu-hi-final-01/evidence.json) | `hf_alpha` | passed | 0/2 |
| [pytorch CPU / it](runs/pytorch-cpu-it-final-01/evidence.json) | `if_sara` | passed | 2/2 |
| [pytorch CPU / pt-br](runs/pytorch-cpu-pt-br-final-01/evidence.json) | `pf_dora` | passed | 1/2 |
| [pytorch CPU / ja](runs/pytorch-cpu-ja-final-01/evidence.json) | `jf_alpha` | failed | not run |
| [pytorch CPU / zh](runs/pytorch-cpu-zh-final-01/evidence.json) | `zf_xiaobei` | passed | 0/2 |
| [onnx CPU / es](runs/onnx-cpu-es-final-01/evidence.json) | `ef_dora` | passed | 2/2 |
| [onnx CPU / fr](runs/onnx-cpu-fr-final-01/evidence.json) | `ff_siwis` | failed | not run |
| [onnx CPU / hi](runs/onnx-cpu-hi-final-01/evidence.json) | `hf_alpha` | passed | 0/2 |
| [onnx CPU / it](runs/onnx-cpu-it-final-01/evidence.json) | `if_sara` | passed | 2/2 |
| [onnx CPU / pt-br](runs/onnx-cpu-pt-br-final-01/evidence.json) | `pf_dora` | passed | 2/2 |
| [onnx CPU / ja](runs/onnx-cpu-ja-final-01/evidence.json) | `jf_alpha` | passed | 0/2 |
| [onnx CPU / zh](runs/onnx-cpu-zh-final-01/evidence.json) | `zf_xiaobei` | failed | not run |

[The original matrix](matrices/matrix-languages-all-all-final-01.json) retains exact commands, exits and deadlines. [The derived summary](summary.json) records each phase denominator, raw comparison, memory snapshot and source revision. Although commands include `--repeats 1`, the selected `--scenarios playback` ran only Lab and Console warmup; these language receipts do not claim active-inference cancellation or repeated Console-loop coverage. The separate [English qualification](../kokoro-english/README.md) records those checks.

The three initial runtime failures remain intact. PyTorch Japanese failed initializing MeCab because the installed full-UniDic path lacked `mecabrc`; its [raw worker log](/private/tmp/tts-macos-burndown/harness/pytorch-cpu-ja-final-01/worker.log) retains the native error. ONNX French and Mandarin reported retryable generation failures. The separate [frontend baseline probe](diagnostics/onnx-frontend-baseline.json) records unsupported eSpeak language names `fr` and `zh`. No successful audio or content receipt is invented for these failed tuples.

ONNX Japanese completed playback but produced the wrong character-description content. The [frontend probe](diagnostics/onnx-frontend-baseline.json) contains English “Chinese” and “Japanese letter” phonemes instead of model-aligned Japanese reading. Both [small-ASR transcripts](runs/onnx-cpu-ja-final-01/content.json) miss every expected anchor and repeat character-description-like syllables. Their source WAV hashes are identical while the transcripts and segment coverage differ; both raw recognizer results remain unchanged. This content failure is kept distinct from the three runtime failures.

Other raw differences remain review evidence rather than assigned pronunciation defects. French small ASR writes `obdorée` for `aube dorée`; Hindi has several substitutions; the PyTorch Portuguese Console clip writes `plateada` for `prateada` while its Lab clip matches. PyTorch Chinese ASR writes traditional characters against a simplified-character prompt. Italian matches the receipt's case/punctuation/whitespace comparison while retaining apostrophe, punctuation and capitalization differences. No script conversion, lexical substitution or pronunciation judgment is applied during curation. Each complete raw transcript, segment time, expected text, anchor position and audio hash is in its unchanged `content.json`.

The 22 clips contain 17 distinct audio byte hashes. Eleven real `afplay` children exited zero and eleven physical sink drains recorded nonzero callback frames. Successful native calls identify PyTorch CPU or ONNX `CPUExecutionProvider`, and exit before phase settlement. All successful playback durations cover the source within the harness's 250 ms tolerance. Source WAV and device callback hashes remain separate because device blocks may contain padding. These observations do not include an acoustic recording, subjective listening or a long-running memory-leak test. Finite before/phase/after memory observations are retained without a broad leak claim.

All initial runs used WAV at 24 kHz mono, speed 1.0 and the requested voices shown above. The source checkpoint/voice and recognizer downloads are pinned in [asset provisioning](provenance/provision.json), [Kokoro metadata](provenance/kokoro-metadata.json), [Whisper metadata](provenance/whisper-metadata.json) and [input phrases](provenance/phrases.json). Runtime receipts retain package versions, native module hashes, actual asset paths and the model alias resolution. The initial installed tuple includes Python 3.12.11, kokoro 0.9.4, kokoro-onnx 0.6.1, Misaki 0.9.4, torch 2.14.0 and onnxruntime 1.29.0. ASR used local multilingual Whisper-small on CPU int8 with faster-whisper 1.2.1 and ctranslate2 4.8.2.

The [full UniDic preparation record](provenance/unidic/asset-provenance.json), [archive verification](provenance/unidic/verification.json), [installed-package inspection](provenance/unidic/installed-package-before.json) and [preparation README](provenance/unidic/README.md) preserve the original deferred-installation state and its source/license provenance. The separately provisioned OpenJTalk dictionary did not satisfy Misaki's default Cutlet/fugashi route. [Initial Whisper-medium provisioning](provenance/medium-provision.json) records the original download allowlist; it is not proof that recognizer initialization succeeded.

All 28 initial source maps match the existing [2,253-file English manifest](../kokoro-english/provenance/source-hashes.json), at captured HEAD `2e3389e694e93592a1c66e5c3416bf29a1057d6c` plus the hashed working changes. Later merged-source reruns retain their own revisions and map overrides rather than relabeling this baseline. Runtime JSON factors only the two repeated hash maps into a relative reference with exact overrides/removals. Rehydrating them and removing `packaging` reconstructs every original JSON value. ASR evidence hashes continue to identify the original raw runtime JSON.

Later receipts explicitly added after completion:

- [onnx-cpu-fr-final-02](runs/onnx-cpu-fr-final-02/evidence.json): runtime exit 0; small: content_review_required, medium: content_passed
- [onnx-cpu-ja-final-02](runs/onnx-cpu-ja-final-02/evidence.json): runtime exit 0; small: content_review_required, medium: content_review_required
- [onnx-cpu-zh-final-02](runs/onnx-cpu-zh-final-02/evidence.json): runtime exit 0; small: content_review_required, medium: content_review_required
- [pytorch-cpu-ja-final-02](runs/pytorch-cpu-ja-final-02/evidence.json): runtime exit 0; small: content_review_required, medium: content_review_required

The first medium-ASR batch failed recognizer initialization in **9 jobs and produced zero transcripts**: its provision allowlist omitted `vocabulary.txt`. [The original command receipt](asr/medium-review-final-01.json) and each startup log remain separate from lexical comparisons. The vocabulary was subsequently provisioned from the same pinned revision; see [the correction receipt](provenance/supplements/medium-vocabulary-provision.json). The later `content-medium.json` files are attached only through their completed attempt output hashes, never retroactively to these failed jobs.

Medium-ASR attempt command receipts: [medium-review-final-01.json](asr/medium-review-final-01.json), [medium-review-final-02.json](asr/medium-review-final-02.json). Completed transcripts remain alongside each run as `content-medium.json`; original small-ASR results are retained.

The [applied UniDic link receipt](provenance/supplements/link-receipt.json) records the later task-local dictionary attachment; the original deferred preparation record is preserved above. Medium ASR uses the pinned revision `08e178d48790749d25932bbc082711ddcfdfbc4f` with the separately hashed vocabulary correction. Completed medium jobs with exit 1 produced full transcripts requiring strict comparison review; they are distinct from the earlier initialization failures, which produced none.

The [final process-release receipt](provenance/supplements/final-process-release.json) records all 26 listed PIDs absent: four repaired-run workers, their four file players and eighteen children from the two medium-ASR attempts. Its exact `ps` command exited 1 with empty stdout/stderr. This is a captured check of those owned PIDs; the earlier fourteen workers retain their own joined cleanup receipts.

Supplemental [negative code-review probes](provenance/supplements/probe-results.json) preserve the initially wrong dependency-error classification and the PyTorch model-entry-after-Stop race. The [accepted ten-case re-review](provenance/supplements/rereview-probe-results.json) verifies four ONNX Japanese/Mandarin dependency failures and PCM/WAV/timestamp cancellation and close controls against actual production routing and the official pipeline, with fake frontends/models. It records joined workers, suppressed model entry/output after Stop and preserved shared identities; same-backend successors pass for cancellation, while explicit close is terminal. Its relevant source hashes match all four repaired live runs. These are boundary checks without real speech. The [targeted regression log](provenance/supplements/regression-green.log) records 197 passed and 2 skipped. [ADR-142](../../../../backlog/decisions/142-kokoro-east-asian-phonemization-across-engines.md) documents the frontend and request-local model-entry decisions.

Native-language review of Hindi and Japanese is tracked in [TASK-32169](../../../../backlog/tasks/task-32169%20-%20Review-Hindi-and-Japanese-Kokoro-speech-content-differences.md); unmatched ASR is not automatically classified as a pronunciation defect.

[Package provenance](package-provenance.json), [semantic/copy verification](verification.json), [package index](package-index.json) and [raw local log hashes](provenance/raw-local-logs.json) support review. Full original runs remain under `/private/tmp/tts-macos-burndown/harness/`. Audio, model/dictionary binaries, archives, private profiles, application source archives and full application logs are omitted; the small recognizer startup-error and targeted-test logs are retained. Curation ran no application imports, models, ASR or audio and changed no production code or Backlog status.
