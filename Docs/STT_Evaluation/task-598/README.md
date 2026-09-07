# TASK-598 external-mode evidence

## Native platform matrix

The four remaining wheel-supported targets passed the bounded native workflow
at commit `9e006d3e618bf228305e3de575d1a4aac699a1d3` in
[GitHub Actions run 31553729188](https://github.com/rmusser01/tldw_chatbook/actions/runs/31553729188):

| Target | Python | ONNX Runtime | v2 total / inference | v3 total / inference |
| --- | --- | --- | --- | --- |
| Linux x86_64 | 3.12.13 | 1.28.0 | 35.79 s / 3.88 s | 13.65 s / 3.74 s |
| Linux aarch64 | 3.12.13 | 1.28.0 | 44.03 s / 3.05 s | 17.86 s / 3.00 s |
| Windows x86_64 | 3.12.10 | 1.28.0 | 31.00 s / 5.97 s | 27.48 s / 6.16 s |
| macOS x86_64 | 3.12.10 | 1.23.2 | 48.76 s / 6.31 s | 37.13 s / 6.23 s |

Each lane installed the declared native transcription extra without a CI-only
runtime pin, provisioned the exact managed Silero VAD, verified external v2 and
v3 INT8 descriptor roots, deleted an optional managed copy, and transcribed
generated four-second zero PCM through `CPUExecutionProvider`. Every lane kept
`artifact_root` null, recorded only the exact managed VAD dependency, left the
external/cache/store/source preference unchanged, retained no managed
Parakeet readiness or active selector, shut down cleanly, and uploaded a
path-private JSON artifact that passed the repository validator. Generated
zero PCM proves runtime execution and lifecycle behavior, not transcription
quality. Normalized results and source-artifact digests are in
[`platform-evidence.json`](platform-evidence.json).

Together with the macOS arm64 smoke below, all five platform gates required by
TASK-598 AC7 have evidence.

## Focused macOS arm64 smoke

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
`git diff --name-only --diff-filter=ACMR origin/dev...HEAD`. The initial
evidence run contained the 24 changed Python test files recorded in the JSON;
the final closeout union contained 25 after the CI evidence test was added.
Each run used one explicit command with the repository virtual environment and
no keyword filtering:

```text
../../.venv/bin/python -m pytest -q Tests/App/test_submit_library_ingest_job.py Tests/CI/test_task598_external_parakeet_evidence.py Tests/Library/test_library_ingest_runner.py Tests/Local_Ingestion/test_parakeet_v2_artifact.py Tests/Local_Ingestion/test_transcription_service_parakeet_buffer_wav.py Tests/Model_Artifacts/test_service.py Tests/STT/test_dispatch_coordinator.py Tests/STT/test_local_stt_executor.py Tests/STT/test_parakeet_dispatch.py Tests/STT/test_parakeet_external.py Tests/STT/test_parakeet_onnx.py Tests/STT/test_parakeet_sources.py Tests/STT/test_transcription_service_facade.py Tests/UI/test_destination_shells.py Tests/UI/test_first_run_wizard_live_contract.py Tests/UI/test_library_ingest_canvas.py Tests/UI/test_llm_screen_lab_adoption.py Tests/UI/test_model_artifact_widgets.py Tests/UI/test_model_browser_state.py Tests/UI/test_model_curated_view.py Tests/UI/test_model_external_view.py Tests/UI/test_model_installed_view.py Tests/UI/test_parakeet_v2_install_ui.py Tests/Wizards/test_first_run_speech_step.py Tests/Wizards/test_first_run_speech_step_state.py
```

Initial evidence result at `e18a4e4d7`: **6 failed, 1248 passed, 1 skipped,
16 warnings in 356.88 seconds**.

Final Task-12 union at `9e006d3e6`, after adding the CI evidence tests, reached
**1361 passed, 2 failed, 1 skipped, 11 warnings, and 2 teardown errors in
348.52 seconds**. The two errors were TASK-598 mounted-test fixtures that had
not neutralized an existing periodic Ollama localhost availability probe under
the repository network guard; both exact nodes pass after the test-only fix.
The two failures are the same unrelated `mcp-tools-workspace-save` missing-
tooltip assertion for the `mcp` and `tools_settings` aliases. They reproduce
independently, and both the audit and MCP implementation are unchanged from
`origin/dev`. The task does not alter that separate MCP baseline defect.

All directly affected dispatch and mounted Models surfaces then passed in one
focused run (**89 passed**), and the two final standalone mounted nodes passed
together. The earlier dispatch import assertion was made suite-order
independent and passes in the union.

Final closeout at `37dfd74c9` reached **1361 passed, 2 failed, 1 skipped, and
10 warnings in 364.05 seconds**. Both failures were the localhost artifact
fixtures being denied permission to bind an ephemeral `127.0.0.1` port by the
sandbox; they were the only non-passes. The exact two nodes then passed
**2 passed in 1.31 seconds** when localhost binding was permitted. The MCP
workspace-save tooltip baseline was resolved separately under TASK-15531, and
both route aliases pass in the final union.

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

## Status

All TASK-598 behavior and all five wheel-supported platform gates are now
evidenced. AC1-7 are satisfied and the Backlog task is **Done**. The final
changed-test union has no product failure: its only two failures were sandbox
socket-bind denials, and both exact localhost nodes pass when that permission
is available.
