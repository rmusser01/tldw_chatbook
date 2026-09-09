# Kokoro PyTorch repair and native runtime validation

**Goal:** Establish real speech and recovery evidence for the three next testing
priorities accepted by the user, repairing reproducible defects.

**Architecture:** Keep the existing provider registry, managed native lifecycle,
and Global/Studio/Console contracts. Replace only Kokoro's placeholder PyTorch
implementation with the official model and language pipeline.

**Tech stack:** Python 3.12, PyTorch, Kokoro 0.9.4, Textual, sounddevice, pinned
audio.cpp 0.5.1 CPU/Metal builds, independent faster-whisper content checks.

**Spec:** User-authorized follow-up validation; Backlog TASK-32111 (Kokoro runtime),
TASK-32112 (native recovery), and TASK-32113 (additional native models).

ADR required: yes
ADR path: backlog/decisions/140-official-kokoro-pytorch-runtime.md
Reason: optional runtime/dependency and Python-compatibility choice. Native test
work preserves ADR-023, ADR-039 and ADR-050 and requires no additional ADR.

## Constraints

- Work only in the owned isolated TTS worktree and task-local environments.
- Use targeted tests; no full-suite sweep. Preserve user config and unrelated work.
- Serialize live inference/speaker runs. Validate full audio content independently;
  byte counts, nonzero samples and a device drain do not establish intelligibility.
- Distinguish real runtime results from fixtures, observer failures, unavailable
  assets, selected runtime quantization and actual quantized model-file tensors.

## Steps

1. Reproduce the old PyTorch loader failure with hash-verified official weights.
   Read upstream KModel/KPipeline source and record supported Python versions.
2. Add failing adapter tests that catch ignored upstream output, lost segments,
   wrong language/voice/speed, invalid blends, missing runtime and blocking model
   loading. Test doubles stop at the external model/pipeline boundary only.
3. Implement the upstream wrapper and backend initialization/download repair.
   Remove placeholder model/tokenization/random-waveform paths. Run those tests
   and the existing Kokoro delivery, limits, downloads and settings regressions.
4. Validate real CPU and MPS Speech Lab and two Speak replies; add British voice
   and supported speed coverage. Decode and independently transcribe complete
   saved files, record actual device callbacks and complete cleanup.
5. In parallel preparation, build a real native interruption/recovery harness.
   Execute CPU then Metal stop, in-flight cancel, owned-child termination and
   same-message retry against immutable source; independently check successful
   replies and assert every task/lease/owned child is released.
6. Qualify PocketTTS with an authorized synthetic reference, then the meaningful
   additional Supertonic configuration. Record asset types and any real blockers.
7. Update user guidance, QA evidence, task notes and ADR index. Run focused lint,
   packaging and tests, review the final diff, then create the authorized dev PR,
   address review, rebase on current dev and merge after required checks.
