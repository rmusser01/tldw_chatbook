# TASK-602 focused macOS evidence

Evidence label: `native_macos_focused_smoke`.

This is focused implementation evidence for TASK-602 on one Apple-silicon
macOS host. It is not a quality benchmark, does not cover Windows or Linux,
and does not open the semantic-default promotion gate reserved for TASK-605.
Exact host, artifact, audio, timing, and result data is in
[`macos-evidence.json`](macos-evidence.json).

## What passed

- The existing managed acquisition service downloaded and verified exact v2
  INT8, v3 INT8, and v2 F32 roots plus the pinned Silero VAD dependency in a
  temporary store. VAD was reused after the first closure.
- Real `onnx-asr==0.12.0` CPU inference transcribed synthesized English with
  v2 INT8 and v2 F32 and synthesized French with v3 INT8. The v3 result kept
  requested `fr`, effective `auto`, null detected language, and the
  `requested_language_not_enforced` warning.
- A 40-second input used the managed VAD and one ASR batch per segment.
  Cancellation observed immediately before the second segment prevented that
  second batch. The same loaded runtime handled two follow-up jobs.
- A real long-form request without a managed VAD failed with
  `artifact_incompatible` and the `retry_faster_whisper` action.
- The focused union passed 450 tests. The localhost acquisition test was run
  separately with permission to bind an ephemeral loopback port and passed.

## Deliberate exclusions and open gates

The focused union deselected the unrelated, pre-existing optional-feature
inventory failure for the `frontmatter` extra. That mismatch predates TASK-602
and is not part of this work stream.

Ruff passed on the changed files except for two unrelated pre-existing findings
in `Tests/Library/test_library_ingest_runner.py` (an unused local at line 2842
and an unused local import at line 3079). Compileall, TOML/JSON parsing, and
`git diff --check` passed.

Windows and Linux native wheel/install/runtime gates remain open because those
hosts are unavailable. TASK-602 therefore remains in progress even though the
macOS implementation evidence is green.
