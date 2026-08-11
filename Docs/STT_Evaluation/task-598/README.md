# TASK-598 focused macOS external-mode evidence

Evidence label: `isolated_macos_external_mode_in_memory_production_path`.

This is focused evidence for TASK-598 on one Apple-silicon macOS host. It
demonstrates the production app-owned source-service → coordinator → executor →
ONNX CPU path with a descriptor-valid, user-owned external Parakeet v2 INT8
root. It is not mounted-picker or microphone evidence and is not cross-platform
evidence. Exact host, descriptor, artifact, transcript, timing, provenance, and
invariance data is in [`macos-evidence.json`](macos-evidence.json).

## What passed

- A scratch profile, configuration, data directory, and managed artifact store
  were established before importing the application. The real profile and real
  managed store had identical snapshots before and after the probe.
- The scratch managed store contained the exact verified Silero VAD closure and
  no managed Parakeet artifact, readiness record, or activation record.
- A descriptor-valid v2 INT8 external root was verified and used in place. The
  runtime did not copy it into the managed store, activate it, mutate its
  required files, or use the network.
- The real app-owned `ParakeetSourceService`,
  `LocalSttDispatchCoordinator`, and `LocalSttExecutor` transcribed known
  16 kHz mono PCM through `onnx-asr==0.12.0` and
  `onnxruntime==1.27.0` with `CPUExecutionProvider`.
- The exact transcript was: “The local transcription stack is working on this
  Mac and does not need a network connection.”
- Provenance kept `artifact_root` null and named only the exact managed VAD
  dependency. External and VAD file sizes, SHA-256 digests, and modification
  times were invariant. The bounded 120-second probe completed without timing
  out and left no child process running.

The input was loaded into memory from the previously verified TASK-602
synthetic WAV. It was not selected through a mounted Textual picker and was not
captured from a microphone. Because the app was deliberately unmounted, one
caught final Library top-up callback could not marshal to an active UI;
transcription and ordered shutdown still completed.

## Exact changed-test union

The union was derived from
`git diff --name-only --diff-filter=ACMR origin/dev...HEAD` and contained the
24 changed Python test files recorded in the JSON. Each run used one explicit
command with the repository virtual environment and no keyword filtering:

```text
../../.venv/bin/python -m pytest Tests/App/test_submit_library_ingest_job.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_parakeet_v2_artifact.py Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py Tests/Model_Artifacts/test_service.py Tests/STT/test_dispatch_coordinator.py Tests/STT/test_local_stt_executor.py Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_external.py Tests/STT/test_parakeet_onnx.py Tests/STT/test_parakeet_sources.py Tests/STT/test_transcription_service_facade.py Tests/UI/test_destination_shells.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_library_ingest_canvas.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_curated_view.py Tests/UI/test_model_external_view.py Tests/UI/test_model_installed_view.py Tests/UI/test_parakeet_v2_install_ui.py Tests/Wizards/test_first_run_speech_step.py Tests/Wizards/test_first_run_speech_step_state.py -q
```

Initial evidence result at `e18a4e4d7`: **6 failed, 1248 passed, 1 skipped,
16 warnings in 356.88 seconds**.

Final reviewed-head result at `bdd2f25e3`: **3 failed, 1262 passed, 1 skipped,
9 warnings in 376.78 seconds**. The two UI test-double failures were corrected
and passed in this union; the earlier First Run union-order failure also passed.
Two localhost artifact fixtures were still denied permission to bind
`127.0.0.1` by the sandbox. The remaining dispatch import-state assertion
passed in isolation in 0.37 seconds. The union was not run again.

## Static verification

- Ruff check on all 49 changed Python files retained two findings in
  `Tests/Library/test_library_ingest_runner.py`; both reproduce from the
  `origin/dev` versions. The baseline has six Ruff findings.
- Ruff format check reported 33 current files would reformat; all 33 also fail
  from `origin/dev`, whose baseline reports 35. No bulk format was applied.
- `py_compile` passed for all 25 changed package modules with bytecode outside
  the repository.
- `git diff --check origin/dev...HEAD` passed.
- Added-line path/privacy, placeholder, source logging, implicit activation,
  runtime download, managed-copy, and unmanaged-VAD scans passed. The only path
  hits were synthetic test paths and the intended `Path.home()` picker
  default.

## Status and open gates

TASK-598 remains **In Progress**. This host supports the available macOS arm64
external-mode evidence, but the exact changed-test union is not fully green in
the sandbox.
Native source verification, copy/delete, and real ONNX CPU smoke remain open on
Linux x86_64, Linux aarch64, Windows x86_64, and macOS x86_64. No Linux or
Windows readiness claim is made.
