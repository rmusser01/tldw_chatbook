# TASK-603 partial focused macOS implementation evidence

Evidence label: `partial_focused_macos_implementation_evidence`.

This is focused implementation evidence for TASK-603 on one Apple-silicon
macOS host. It is **not** live Console microphone acceptance evidence, a speech
quality benchmark, cross-platform evidence, or completion of the release gate.
Exact host, artifact, runtime, test, review, and limitation data is in
[`macos-evidence.json`](macos-evidence.json).

## What is evidenced

- A real `TldwCli` dictation factory routed in-memory PCM through the one
  app-owned coordinator and executor to Parakeet v2 INT8 ONNX on CPU. The
  4.7025-second synthetic English fixture returned its expected sentence in
  4.407 seconds without network access or PCM disk staging.
- The immutable TASK-602 artifact store was reused without acquisition or
  mutation. The pinned v2 INT8 root, Silero VAD dependency, readiness closure,
  file hashes, and a managed read lease were verified.
- At final rebased commit `f8827ddff24b7415acc1b7f40dc40564b55a014d`,
  148 focused coordinator/executor/runtime/facade tests and ten exact
  app/Chat/UI contract nodes passed. The review-found oversized one-shot gap
  has focused red/green and full coordinator-file evidence.
- Final changed-package `py_compile` and `git diff --check` passed. Changed-file
  Ruff reports only two proven pre-existing findings in the changed Library
  test file.
- Whole-branch review found one Important boundedness defect, fixed it at the
  final commit, and approved the focused re-review with no remaining finding.

## Deliberate limitations and open gates

The mandated changed-test union is not green: its one permitted run reached
96% and the interpreter aborted with exit 134 during a background retained
Parakeet MLX native warm-up. The exact active node passed alone, which makes the
abort load/order-sensitive but does not replace the missing union summary.

The full Textual app mount produced no retained user-surface markers. This
evidence therefore does not claim a real Mic press, microphone capture, caret
insertion, unchanged message count, visible Library-busy ordering, live hard
limit behavior, or a real retry failure. Windows and Linux remain untested, and
the dead legacy implementation remains intentionally retained for TASK-605.

TASK-603 remains **In Progress**. Only AC1–AC3 have direct implementation,
runtime, focused-test, and review evidence; AC4–AC6 remain open. This is partial
mergeable implementation evidence, not release-gate completion.
