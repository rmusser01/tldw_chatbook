# TASK-603 bounded Console dictation evidence

TASK-603 combines two complementary evidence sets:

- [`macos-evidence.json`](macos-evidence.json) records the real Apple-silicon
  macOS Console Mic/capture/caret smoke and native Parakeet v2 INT8 ONNX CPU
  path.
- [`platform-evidence.json`](platform-evidence.json) records deterministic
  bounded-dictation contracts on Linux x86_64, Linux arm64, Windows x86_64,
  macOS arm64, and macOS x86_64.

The hosted matrix is control-plane evidence. It does not claim runner audio
hardware, microphone capture, model download, or native ONNX inference. The
physical macOS smoke supplies the real-device proof, while TASK-602 supplies
the same five platforms' native Parakeet ONNX evidence.

## Five-platform contract evidence

The passing release-candidate commit is
`f609966d732b18b806e735294e6c41da6269d196`. GitHub Actions run
[`31627875630`](https://github.com/rmusser01/tldw_chatbook/actions/runs/31627875630),
attempt 1, passed all five lanes:

| Evidence lane | Hosted runner | Python | Result | Pytest duration |
| --- | --- | --- | --- | --- |
| Linux x86_64 | `ubuntu-24.04` | 3.12.13 | passed | 7.132 s |
| Linux arm64 | `ubuntu-24.04-arm` | 3.12.13 | passed | 6.454 s |
| Windows x86_64 | `windows-2022` | 3.12.10 | passed | 7.076 s |
| macOS arm64 | `macos-15` | 3.12.10 | passed | 5.001 s |
| macOS x86_64 | `macos-15-intel` | 3.12.10 | passed | 8.797 s |

Every lane ran the same ten exact nodes. Together they cover the derived
60-second PCM byte ceiling, ordered coalescing under one pending request,
pending cancellation without active-batch preemption, bounded shutdown while
dictation waits, heavy-only Library gating, dictation-first terminal handoff,
cooperative shutdown ordering, the unsupported-streaming fallback, the
mounted visible limit transition, and explicit Mic-button resume without
hands-free auto-reopen or auto-send.

Each lane's JUnit output was normalized into a strict path-private JSON
document. A lane passes only when the pytest process succeeds and every exact
required node appears once, from its expected module, with a passed outcome.
All five documents independently validated, shared the exact commit/run/URL,
and were aggregated only by
`.github/scripts/task603_dictation_evidence.py`. The committed aggregate also
validates with:

```bash
python .github/scripts/task603_dictation_evidence.py \
  --validate Docs/STT_Evaluation/task-603/platform-evidence.json
```

## Honest RED history

Two earlier new-SHA runs remained red and were not aggregated:

1. Run
   [`31626730904`](https://github.com/rmusser01/tldw_chatbook/actions/runs/31626730904)
   at `ad963d16f50044707e2929432ae0e49ca463af69` exposed that four selected
   async/Textual tests needed the repository's narrow `allow_network` marker
   on Windows. Python's Proactor event-loop setup owns an internal loopback
   socket pair, which the test network guard otherwise blocked. The other
   four lanes passed. No production or guard policy changed.
2. Run
   [`31627372105`](https://github.com/rmusser01/tldw_chatbook/actions/runs/31627372105)
   at `ccb98fb4accdb46a7d68293871f4bf20f062f50c` passed nine Windows nodes but
   exposed one mounted-test ordering race: a simulated Mic-button click crossed the
   preceding posted `VoiceFinal` repaint and never reached the button handler.
   The test now waits for the observable partial-to-final UI round trip before
   that click. The other four lanes again passed; production remained
   unchanged.

The normalizer correctly preserved both Windows artifacts as failed evidence.
The passing run is a third, brand-new run bound to the final corrected SHA,
not a retry of either red run.

## Local verification on the release candidate

The exact required selection passed locally: `10 passed` with only existing
dependency/deprecation warnings. The final evidence/semantic ratchet file
passed `53 passed`. The scoped network-guard suite passed `9 passed` when
allowed to create its own temporary Unix and localhost sockets; its first
sandboxed run was denied by the host before repository policy was involved.

Scoped Ruff lint, changed-range Ruff format checks, `py_compile`, and
`git diff --check` passed. Whole-file checks still report established unrelated
lint/format debt in legacy test files; those files were not broadly rewritten.

## Physical macOS evidence retained

The earlier Apple-silicon evidence remains the hardware proof. A physical
Console Mic press opened the host PyAudio device, captured playback of the
known fixture, routed 159,360 PCM bytes through the app-owned coordinator and
executor to Parakeet v2 INT8 ONNX on CPU, inserted the exact expected text at
the existing caret without sending, returned Mic to idle, and emitted no
user-visible failure. The immutable TASK-602 artifact store and managed Silero
VAD dependency were reused without acquisition or mutation.

That smoke is intentionally not repeated on hosted runners: their deterministic
tests have no audio-device claim, and the workflow installs no STT model/runtime
extra, downloads no model, and runs no inference.

## Scope and follow-on

This evidence closes TASK-603's bounded dictation release gate. It does not
promote a default provider or remove the retained legacy implementation; that
work remains TASK-605.
