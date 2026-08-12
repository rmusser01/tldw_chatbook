# TASK-602 native Parakeet platform evidence

Evidence label: `task602_native_parakeet_matrix`.

TASK-602's required native CPU matrix passed on one executable commit. The
machine-readable result is
[`platform-evidence.json`](platform-evidence.json).

## Passing run

- Tested commit: `60d8b73b9c9223cef696e9bc3577d186af7e26be`
- Workflow run: [31618353807](https://github.com/rmusser01/tldw_chatbook/actions/runs/31618353807)
- Run attempt: `1`
- Trigger: the explicit `task-602-platform-evidence` PR label
- Python: 3.12 on every lane

| Evidence lane | Native host | ONNX Runtime | Result |
| --- | --- | --- | --- |
| `linux-x86_64` | Linux x86_64 | 1.28.0 | Passed |
| `linux-aarch64` | Linux aarch64 | 1.28.0 | Passed |
| `windows-x86_64` | Windows x86_64 | 1.28.0 | Passed |
| `macos-arm64` | macOS arm64 | 1.28.0 | Passed |
| `macos-x86_64` | macOS x86_64 | 1.23.2 | Passed |

Every lane resolved `onnx-asr==0.12.0`, `faster-whisper==1.2.1`, and
`ctranslate2==4.8.1`, selected `CPUExecutionProvider`, completed cleanup, and
passed all required checks:

- package resolution and the cheap runtime probe;
- exact managed Parakeet v2 INT8 CPU inference;
- exact managed Parakeet v3 INT8 CPU inference;
- managed long-form Silero VAD;
- cancellation before the second segment batch;
- same-identity resident batch reuse with held artifact leases; and
- normalized `retry_faster_whisper` recovery wiring.

The exact shared artifact identities were:

- Parakeet v2 INT8 revision
  `0bbb45a3365852604aef28b538a8f066f4ccaa85-vad-b3e3ee3cce4c`, closure
  fingerprint `d52f16e6505c8efc3e5a9178f597e2414814ae44e677ea0cd75b317a240effc0`;
- Parakeet v3 INT8 revision
  `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce-vad-b3e3ee3cce4c`, closure
  fingerprint `9ec622539e4e11990aef699c7c43f4e9f05c0d5c0e8235abec04f0ced8bbb1e8`;
- Silero VAD F32 revision
  `b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6`.

## Evidence integrity

Each named platform artifact was downloaded to a separate temporary directory
and independently validated with the checked-in normalizer. The aggregate was
then created only through that normalizer and validated again:

```bash
python .github/scripts/task602_platform_evidence.py \
  --validate-aggregate Docs/STT_Evaluation/task-602/platform-evidence.json
```

The aggregate requires exactly the five expected platforms, the same tested
commit, workflow run, attempt, canonical URL, and passed status. It excludes
local paths, commands, transcripts, exceptions, environment values, PIDs,
handles, credentials, and temporary names.

## Fixture attribution

The smoke used PyTorch Audio's 16 kHz mono VOiCES tutorial sample,
`Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav`, under CC BY 4.0. The runner
downloaded it from the PyTorch tutorial-assets host and required SHA-256
`c65fcd726d6b08c82c1e5dc7558f863cd8d483e3ed2f4a7bcf271dc1865ada14` before
inference. The fixture is not committed to this repository.

## Failed executable attempt retained for audit

The first labeled run,
[31616382421](https://github.com/rmusser01/tldw_chatbook/actions/runs/31616382421),
tested commit `aea7bd11be687383dd6f156de94bba51472156b3`. Linux aarch64 failed before
runtime load because the evidence adapter supplied an internal artifact lease
key object where the production executor request requires the public
three-string artifact reference. The strict normalizer recorded the lane red,
and the remaining lanes were cancelled rather than retried.

The adapter was corrected with a focused regression test, the complete local
affected gate passed, and a brand-new five-lane run
[31617299767](https://github.com/rmusser01/tldw_chatbook/actions/runs/31617299767)
passed on commit `2542cdb43da7b7d74416e979be412595efc6922e`. A later PR review removed one
unreachable duplicate raise and completed a public docstring. Although these
changes did not alter runtime behavior, they changed the executable evidence
file, so the final five-lane run was repeated on the reviewed commit rather
than combining results across commits. The initial failure was an
evidence-runner defect, not a native runtime or production containment
failure.

## Scope

This evidence closes TASK-602's five-platform native wheel/runtime gate. The
earlier Apple-silicon focused smoke in [`macos-evidence.json`](macos-evidence.json)
remains historical implementation evidence. Neither record promotes semantic
defaults, removes the legacy provider, adds accelerator support, or turns this
expensive release gate into general CI; those decisions remain outside
TASK-602.
