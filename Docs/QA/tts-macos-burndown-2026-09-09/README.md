# macOS ARM TTS qualification — 2026-09-09

This work extends the earlier Kokoro, Chatterbox and audio.cpp playback
qualification with real cancellation, repeated playback, additional providers,
languages and Linux ARM coverage. Model inference, ASR and physical playback
run serially on an Apple M5 Max with 128 GB memory and macOS 26.5.2. Linux uses
an isolated ARM container; it has no physical Linux audio device.

## Evidence

| Area | Evidence and current result |
| --- | --- |
| Kokoro English | [CPU, MPS and ONNX](kokoro-english/README.md): 33 complete successful clips with normalized-exact ASR, three real mid-inference cancellations, recovery and bounded repeated playback. |
| Native audio.cpp shutdown | [CPU and Metal repair](native/README.md): reproduced crashes, repaired request lifetime, real shutdown and successor playback, four complete normalized-exact source/device transcripts. The approved application runtime baseline is unchanged. |
| Non-English Kokoro | [Fourteen engine/language tuples](kokoro-languages/README.md): all now generate and play. Repaired French ONNX locale routing, Japanese/Mandarin ONNX phonemization and Japanese setup guidance. Original failures and complete multilingual transcripts remain visible. |
| AllTalk and Higgs | [Provider receipts](providers/README.md): AllTalk p225/p226 playback passed, with unresolved content differences. Higgs passed playback, joined Stop, successor and three exact transcripts on the isolated PortAudio repair; its two original native cleanup failures are retained. |
| PortAudio | [Native sink controls](portaudio/README.md): six corrected drain/Stop/successor controls passed on the exact candidate library, including actual callback PCM and clean process exits. The installed default library is unchanged. |
| Linux ARM | [Headless ONNX qualification](linux-arm/README.md): real synthesis, inference-overlap Stop, successor and two full exact transcripts passed in Docker. Physical Linux playback remains untested. |
| Pre-review integration | [Installed wheel and targeted checks](integration/README.md): CPU, MPS and ONNX passed nine complete clips and three real Stop/recovery controls on the earlier 2,274-file wheel. These receipts remain frozen at their original source revision. |
| Final review qualification | [Review fixes and clean installed wheel](review/README.md): 405 targeted tests passed, followed by 81 overlapping migration/resolver/provider tests. All 2,275 Python files match source, wheel and installation. Five real playback tuples passed; English yielded nine exact clips and three Stop/recovery controls. All 23 recorded owned processes exited and user configuration stayed unchanged. |

Each package distinguishes successful runtime behavior, full-content comparison,
failed attempts and cleanup. Source manifests bind the tested uncommitted code;
the checkout's HEAD alone does not identify the tested implementation. Audio,
models, binaries and private profiles remain in the task-owned local evidence
directory and are not committed.

Upstream patches, the PortAudio license, the UniDic catalog, the macOS startup
sample and raw command logs retain their original bytes, including whitespace.
The authored files pass the whitespace check; seven hashed source/output
snapshots and four raw post-review pytest failure logs are excluded from that
check to preserve their recorded identity.

The [opt-in Kokoro runner](../../Development/TTS/Live_Validation.md) provides
repeatable production-path playback, cancellation and content checks. These
checks exercise the mounted Speech Lab and trusted Console delivery path;
they do not claim complete Console navigation or acoustic microphone loopback.
The final review wheel was built after rebasing onto `dev` at `41c4a5858f`;
the earlier integration package targets `a36fc6133c`. Each package identifies
its exact source revision and byte hashes. The custom test host does not
open a TTS profile repository, so these runs do not qualify the separately
changed SQLite profile-helper lifecycle.

## Follow-up boundaries

Ten fix/qualification tasks are complete. Eleven new follow-ups remain:

| Backlog task | Remaining prerequisite or outcome |
| --- | --- |
| TASK-32153 | Physical Linux audio device; Docker already passed headless ONNX qualification. |
| TASK-32154 | Deliberately configured OpenAI cloud TTS credentials and quota. |
| TASK-32155 | Deliberately configured ElevenLabs credentials and quota. |
| TASK-32156 | Windows CPU/ONNX runtime, physical playback device and a Windows-compatible guarded content reader. |
| TASK-32157 | Real CUDA host for Kokoro. |
| TASK-32158 | Real CUDA host for Chatterbox. |
| TASK-32163 | Reviewed adoption of the qualified audio.cpp request-drain patch into approved runtimes. |
| TASK-32166 | Reviewed PortAudio repair distribution and default library selection. |
| TASK-32167 | Independent review of AllTalk VITS content differences. |
| TASK-32168 | A supported Higgs serve-engine cancellation API that stops between decoder steps. |
| TASK-32169 | Native-language review of Hindi and Japanese Kokoro content differences. |

Existing audio.cpp Windows parity and model-recipe expansion remain in
TASK-13208 through TASK-13212. They are retained rather than duplicated; this
work does not qualify every declared audio.cpp model family or package.

Shipping native repairs is separate from qualifying an isolated build. The
audio.cpp request-drain patch is retained for reviewed runtime adoption. The
PortAudio shutdown repair is an open upstream change whose library selection
and distribution must be resolved before it becomes an application default.
Higgs's qualified Stop joined 512 decoder forwards in 96.400 seconds; it is
safe ownership evidence, not responsive cancellation. Mandarin transcripts
preserve Traditional/Simplified script differences. Hindi, Japanese and AllTalk
content uncertainty is not hidden by changing expected text or weakening the
verifier. The language package defines its combined recognizer-selection policy
and retains every small/medium result, including recognizer setup failures.
