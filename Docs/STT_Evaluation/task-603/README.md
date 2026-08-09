# TASK-603 focused macOS Console dictation evidence

Evidence label: `partial_focused_macos_console_evidence`.

This is focused implementation evidence for TASK-603 on one Apple-silicon
macOS host. It includes a real Console Mic/capture/caret smoke, but it is not a
speech quality benchmark, cross-platform evidence, or completion of the release
gate. Exact host, artifact, runtime, test, review, and limitation data is in
[`macos-evidence.json`](macos-evidence.json).

## What is evidenced

- A real `TldwCli` dictation factory routed in-memory PCM through the one
  app-owned coordinator and executor to Parakeet v2 INT8 ONNX on CPU. The
  4.7025-second synthetic English fixture returned its expected sentence in
  4.407 seconds without network access or PCM disk staging.
- At rebased commit `24a2ba3cf`, a physical Console Mic press opened the host
  PyAudio device, captured playback of the same fixture, and sent 159,360 PCM
  bytes through the app-owned coordinator/executor. The exact expected text was
  inserted at the existing caret, the draft was not sent, Mic returned to idle,
  and no user-visible failure was emitted.
- The live smoke exposed and fixed two production blockers: deferred dictation
  now honors `transcription.parakeet_onnx_model_dir`, and the first Python 3.12
  STT worker spawn is protected from Textual's `stderr.fileno() == -1` capture.
- The immutable TASK-602 artifact store was reused without acquisition or
  mutation. The pinned v2 INT8 root, Silero VAD dependency, readiness closure,
  file hashes, and a managed read lease were verified.
- At the earlier evidence commit `f8827ddff24b7415acc1b7f40dc40564b55a014d`,
  148 focused coordinator/executor/runtime/facade tests and ten exact
  app/Chat/UI contract nodes passed. The review-found oversized one-shot gap
  has focused red/green and full coordinator-file evidence.
- Final changed-package `py_compile` and `git diff --check` passed. Changed-file
  Ruff reports only two proven pre-existing findings in the changed Library
  test file.
- Whole-branch review found one Important boundedness defect, fixed it before
  the live Console follow-up, and approved the focused re-review with no
  remaining finding.
- Post-rebase follow-up verification passed 94 executor/facade tests and seven
  exact limit/resume/batch-ordering nodes. Focused review of the two live-smoke
  fixes found no Critical, Important, or Minor issues.

## Deliberate limitations and open gates

The mandated changed-test union is not green: its one permitted run reached
96% and the interpreter aborted with exit 134 during a background retained
Parakeet MLX native warm-up. The exact active node passed alone, which makes the
abort load/order-sensitive but does not replace the missing union summary.

The real Mic smoke establishes capture, exact caret insertion, unchanged
message count, idle recovery, and shared executor ownership. Hard-limit/explicit
resume and batch-to-dictation ordering are covered by focused mounted/contract
tests rather than a 60-second physical capture. A destructive real retry failure
was not induced. Windows and Linux remain untested, and the dead legacy
implementation remains intentionally retained for TASK-605.

TASK-603 remains **In Progress**. AC1–AC5 have direct implementation, runtime or
focused user-surface/contract evidence. AC6 remains open for representative
Windows/Linux evidence and the full release gate. This is partial mergeable
implementation evidence, not release-gate completion.
