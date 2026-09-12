# Task 1 report — self-contained audio and speech port

## Outcome

Status: **DONE_WITH_CONCERNS**

The source audio/speech implementation was ported from committed source `258beb6120a5e8c84d4f83b12a39427733301e57` onto dev base `89d803a25aab216e061a049e60d9680d0722282b`. Task-owned self-contained software tests pass. Twenty-eight source tests remain composition-bound to Chat modules/test fixtures assigned to later tasks; none of those out-of-scope paths were imported.

ADR required: no new ADR.
ADR path: existing ADR-094 prerequisite, as authorized in the Task 1 brief.
Reason: this task directly ports the already-decided audio/process contracts and does not make a new architectural decision.

## Exact imported inventory and blob identity

The 59 explicit paths plus all 324 vendored files are recorded in the sibling [task-1-hash-inventory.tsv](task-1-hash-inventory.tsv). It contains each path, source Git blob, current Git blob, and identity result. Across all 383 entries, 382 are byte-identical. The sole permitted adaptation is `native/voice_aec/pyproject.toml`, whose package version changed from `0.1.9.0` (source blob `7c2ef295fbac0130f8497a1c1bd6ed091d16fd1e`) to the app version `0.2.0` (current blob `6885c3d9d8a40a2ecff173c5c321abdcfd0e97f7`).

The complete `native/voice_aec/vendor` directory was also imported. Its source Git tree is `0f6fb63fc3cca0098b599da1e7d62a695af07d1f`, containing 324 files. Every one of the 324 current file blobs was compared with the corresponding source blob and matched. Together with the 59 explicit paths, the audit checked 383 imported files and found only the permitted `pyproject.toml` adaptation.

Qualification/build identity files were intentionally not regenerated or edited:

- `tldw_chatbook/Audio/voice_build_identity.json`: SHA-256 `c56c99ed5b6b96dd99734dd4be0b109758042d50a657bc4d2b9e967f176fcc2a`, identical to source.
- `tldw_chatbook/Audio/voice_qualification_manifest.json`: SHA-256 `473ed30857d879dabaa81e512c337f8f9f2edf25bf2df520dd358cdfe3156cf6`, identical to source; all platforms remain unqualified.

## Shared speech modules and affected tests

Only the source voice hunks were applied. Source/current Git blobs are recorded below.

| Path | Source blob | Current blob | Result |
|---|---|---|---|
| `Tests/Audio/test_dictation_lazy_transcription.py` | `1594ecfb3cbd415693b51eb2833dc923bbf195cf` | `1594ecfb3cbd415693b51eb2833dc923bbf195cf` | source hunk/file exact |
| `Tests/Audio/test_dictation_speech_resumed.py` | `75ee6834eebdb95a181d7792ef2013357ba285ea` | `75ee6834eebdb95a181d7792ef2013357ba285ea` | source hunk/file exact |
| `Tests/STT/test_transcription_service_facade.py` | `b2c63581f9e8db816b07f3e14bc1bd551bbb2c84` | `b2c63581f9e8db816b07f3e14bc1bd551bbb2c84` | source hunk/file exact |
| `Tests/TTS/test_pcm_stream_plan.py` | `e3f4edcbd4a227f6b63e92f20493a8b2458a0ac3` | `e3f4edcbd4a227f6b63e92f20493a8b2458a0ac3` | source hunk/file exact |
| `Tests/TTS/test_tts_request_admission.py` | `6c0c286d1fd93d57cdb174be1919c55051919c26` | `6c0c286d1fd93d57cdb174be1919c55051919c26` | source hunk/file exact |
| `Tests/Utils/test_fd_protection.py` | `b4beda5989fd63cbaf818f6feb7071a852708a44` | `b4beda5989fd63cbaf818f6feb7071a852708a44` | source hunk/file exact |
| `tldw_chatbook/Audio/dictation_service_lazy.py` | `9313af7cbd3d675f034b0a716a7f926a49ecc2ab` | `a32e64f487689a1212d8f44ae23ebc4e460bc555` | adapted: retained dev recorder_factory while adding transcript_engine |
| `tldw_chatbook/Local_Ingestion/transcription_service.py` | `0fc3533b968e343a7cf1df5f41850bd010dca343` | `0fc3533b968e343a7cf1df5f41850bd010dca343` | source hunk/file exact |
| `tldw_chatbook/STT/executor_worker.py` | `bb8cfb7682cf91830ecaab8c0579560ef68d09c6` | `bb8cfb7682cf91830ecaab8c0579560ef68d09c6` | source hunk/file exact |
| `tldw_chatbook/TTS/TTS_Generation.py` | `f7e036d66302e6f6e9843eba86074d93d73cded0` | `f7e036d66302e6f6e9843eba86074d93d73cded0` | source hunk/file exact |
| `tldw_chatbook/TTS/pcm_stream.py` | `961a5dde0312b25771711cabe1a17e950c0f3195` | `961a5dde0312b25771711cabe1a17e950c0f3195` | source hunk/file exact |
| `tldw_chatbook/TTS/request_admission.py` | `4ab494d4a32a664b080021931e4951efb827c363` | `4ab494d4a32a664b080021931e4951efb827c363` | source hunk/file exact |
| `tldw_chatbook/Utils/fd_protection.py` | `c09d59629f5729fce834f36004b2138aa93ba6bd` | `c09d59629f5729fce834f36004b2138aa93ba6bd` | source hunk/file exact |

The `LazyLiveDictationService` adaptation is intentional and limited: the source `AudioFrame`/rolling-engine seam was added while dev's existing `recorder_factory` constructor parameter, storage, documentation, and recorder construction behavior were retained. The corresponding existing recorder-factory regression test was included in verification.

The six modified shared test files are source-byte-identical. No source-branch deletions, unrelated formatting changes outside those files, current privacy/trace UI, app wiring, package-data wiring, or optional-dependency changes were imported.

## RED evidence

New source tests were imported before production implementation.

1. Initial audio contract run:

```bash
../../.venv/bin/python -m pytest -q \
  --basetemp=/tmp/tldw-task1-green-self-contained.cDbeuq \
  Tests/Audio/test_duplex_contracts.py \
  Tests/Audio/test_rolling_transcript.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Audio/test_voice_process_protocol.py \
  Tests/Audio/test_voice_process_io.py \
  Tests/Audio/test_voice_process_types.py \
  Tests/STT/test_resident_buffer_runtime.py
```

Result: collection stopped with five expected missing-contract errors: `duplex_contracts`, `rolling_transcript`, and `voice_process_io` were absent.

2. Shared dictation seams:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> \
  Tests/Audio/test_dictation_lazy_transcription.py \
  Tests/Audio/test_dictation_speech_resumed.py \
  -k 'bare_new_service_has_no_rolling_engine or admitted_frame_hook or rolling_hook'
```

Result: 3 failed as expected: missing `AudioFrame`, missing `transcript_engine` constructor support, and missing default attribute.

3. Resident buffer/facade:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> \
  Tests/STT/test_resident_buffer_runtime.py \
  Tests/STT/test_transcription_service_facade.py \
  -k 'resident or buffer_owner or public_constructor'
```

Result: 18 failed as expected because `ResidentBufferRuntime` was absent.

4. Normalized PCM:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> Tests/TTS/test_pcm_stream_plan.py
```

Result: one collection error because `PcmStreamError` and `iter_normalized_pcm_frames` were absent.

5. Hands-free synthesis:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> \
  Tests/TTS/test_tts_request_admission.py -k hands_free
```

Result: 5 failed as expected because `synthesize_hands_free` and `TTSHandsFreeAudioUnavailableError` were absent.

6. Bounded descriptor guard:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> \
  Tests/Utils/test_fd_protection.py -k bounded_guard
```

Result: 1 failed as expected because `protect_file_descriptors` did not accept `timeout`.

## GREEN evidence

Core targeted self-contained run:

```bash
../../.venv/bin/python -m pytest -q --basetemp=<isolated> \
  Tests/Audio/test_dictation_lazy_transcription.py \
  Tests/Audio/test_dictation_speech_resumed.py \
  Tests/Audio/test_dictation_recorder_factory.py \
  Tests/Audio/test_duplex_contracts.py \
  Tests/Audio/test_rolling_transcript.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Audio/test_voice_process_protocol.py \
  Tests/Audio/test_voice_process_io.py \
  Tests/Audio/test_voice_process_types.py \
  Tests/STT/test_resident_buffer_runtime.py \
  Tests/STT/test_transcription_service_facade.py \
  Tests/TTS/test_pcm_stream_plan.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/Utils/test_fd_protection.py \
  -k 'not audio_and_admitted_frames_validate_optional_speech_onset
      and not preroll_speech_onset_can_follow_its_pcm_end
      and not contract_modules_import_without_native_audio_dependencies
      and not bootstrap_omitted_closed_stt_options_preserve_none_defaults
      and not invalid_stt_options_rejected_by_parser_before_child_factory
      and not public_enum_aliases_preserve_identity_and_settings_values
      and not parent_policy_rejects_fabricated_context
      and not parent_classifies_originals_and_core_receives_only_handles
      and not promotion_adapter_projects_exact_outcomes_without_hidden_commit
      and not unknown_cleanup_receipt_suspends_the_draft
      and not parent_provider_failure_is_projected_before_core_reduction'
```

Result: **584 passed, 24 deselected, 3 existing dependency/deprecation warnings in 7.29 s**.

Supplemental task-owned local STT/process run:

```bash
../../.venv/bin/python -m pytest -q \
  --basetemp=/tmp/tldw-task1-green-stt-only.ZUEdhL \
  Tests/Audio/test_local_voice_stt_process.py \
  Tests/Audio/test_parakeet_voice_worker.py \
  -k 'not dead_prewarm_worker_does_not_advertise_rolling_fallback \
      and not native_adapter_sends_pcm_without_importing_mlx \
      and not uncertain_prewarm_cannot_offer_fallback'
```

Result: **53 passed, 4 deselected, 2 existing dependency warnings in 22.83 s**.

Total non-overlapping verified software cases: **637 passed**.

A diagnostic run without composition deselection produced **584 passed and 24 failed**; all 24 failures stopped at missing Chat modules/test helpers. The analogous supplemental diagnostic produced **105 passed and 4 failed**, with all four stopping at the missing speculative voice session module. No Task 1 implementation assertion failed in either diagnostic.

Static verification:

```bash
../../.venv/bin/ruff check <all changed/imported Python paths>
git diff --check
```

Result: Ruff reported `All checks passed!`; `git diff --check` produced no output.

Per instruction, no full suite, native GIL-hold group, soak, physical qualification, live app/audio, model/provider request, or native rebuild was run.

## Remaining composition dependencies

The following direct missing production paths were observed in the 28 deliberately deselected tests and exist in the source commit. They belong to later Chat/UI integration tasks:

- `tldw_chatbook/Chat/console_speculative_voice.py`
- `tldw_chatbook/Chat/console_speculative_voice_session.py`
- `tldw_chatbook/Chat/console_voice_attempts.py`
- `tldw_chatbook/Chat/console_voice_controls.py`
- `tldw_chatbook/Chat/console_voice_eligibility.py`
- `tldw_chatbook/Chat/console_voice_promotion.py`
- `tldw_chatbook/Chat/console_voice_settings.py`

The direct missing source test-helper paths are:

- `Tests/Chat/test_console_speculative_voice.py`
- `Tests/Chat/test_console_voice_effect_barrier.py`
- `Tests/Chat/test_console_voice_eligibility.py`
- `Tests/Chat/test_console_voice_process.py`

The dependency-light import test specifically needs `tldw_chatbook/Chat/console_voice_settings.py`. It did not expose a need to relocate the current `tldw_chatbook.Audio` package initializer; the failure is the direct absence of that Chat module, so no broad initializer/UI change was made.

Task 4 still owns app optional-dependency and package-data wiring for `tldw-voice-aec`. Native qualification/build identity remains deliberately unchanged and unqualified throughout this PR. Task 4 must not regenerate qualification authority.

## Self-review

- Worktree allowlist audit found `unexpected=0`.
- All new paths were absent before import; no existing dev path was overwritten by the mechanical blob import.
- Shared edits are limited to the seven named production modules and six affected test files.
- Existing `recorder_factory` behavior is retained and tested.
- No source deletions or unrelated source-branch changes were applied.
- Vendor/native ABI source was not redesigned.
- All test invocations used isolated temporary `--basetemp` directories.
