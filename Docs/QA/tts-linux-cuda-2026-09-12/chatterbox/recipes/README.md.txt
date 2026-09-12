# Task-owned Chatterbox CUDA qualification

This standalone runner derives the mounted Lab/Console lifecycle from the
reviewed `scripts/validate_live_tts.py`. It changes no production source and
does not replace model results, audio encoding or playback.

Run `build_runner.py --source /checkout/scripts/validate_live_tts.py` to regenerate
`qualify_chatterbox.py`. The generated evidence contains the SHA256 of that exact
source. The generator rejects missing source anchors. Format/lint cleanup used
Ruff from the main checkout: `ruff check --fix DIR; ruff format DIR`.
Transfer all four runtime files together: `qualify_chatterbox.py`,
`chatterbox_observation.py`, `observe_child.py`, and the generator for provenance.

Example for the prepared task-owned remote environment (replace the interpreter
and installed package paths with the verified Chatterbox wheel environment):

```sh
/abs/chatterbox-venv/bin/python -I /abs/driver/qualify_chatterbox.py \
  --model-cache /home/ml-user/tts-linux-cuda-20260912/cache/huggingface/hub \
  --expected-package-root /abs/chatterbox-venv/lib/python3.12/site-packages/tldw_chatbook \
  --reference-dir /home/ml-user/tts-linux-cuda-20260912/models/synthetic-reference \
  --voice synthetic --format wav --scenarios playback,cancel,repeat --repeats 3 \
  --output /home/ml-user/tts-linux-cuda-20260912/runs/new-chatterbox-cuda \
  --play-audio
```

**Isolation note:** The standalone runner explicitly admits its own task-owned
sibling helper directory under `-I`; it does not add the working directory or
checkout. The installed package root and all source hashes are still verified.
`--model-cache` is the HF **hub cache** root, not `HF_HOME`. Use the installed
default voice by omitting `--reference-dir` and `--voice`; references are optional
synthetic local `.wav` stems. `--voice synthetic` now selects only the reference
stem for an additional final `speech-lab-reference` phase. All baseline Lab and
Console requests use the real catalog's `default` voice. The extra phase invokes
the existing file-picker completion callback, selects the real `custom` option,
presses Generate and Play, and requires the actual child `audio_prompt_path` and
hash to match the copied reference. It never adds synthetic stems to a catalog.
With all scenarios and three repeats there are six default successful clips,
one cancelled phase, and one additional successful reference clip. Full-content
verification must include that seventh clip too.
Missing local references fail admission. Cache
downloads/network activity are prohibited by offline environment and audit hook.

The actual registered backend launches its installed `chatterbox_process.py`.
The task observer changes that launch to a local shim that imports the original
native classes and then `runpy` executes that exact installed script. Its hash,
the launcher's hash, child PID and runtime paths/hashes are recorded. Its normal
stdin/stdout protocol and original native argument/result objects are preserved.
A fail-closed guard rejects an in-process fallback; it does not force another
backend path. No delays, replacement audio or alternate model loaders are used.

The original `from_pretrained(device)` resolves its real cached assets. Each
resolved asset filename, path, resolved path and hash is retained; all parameters
of `t3`, `s3gen` and `ve` must be on `cuda:0`. Device name, memory, compute
capability and Torch/CUDA build versions come from the child, not the parent.
Retain driver and physical GPU identity in the host manifest separately.

`generate_calls` contain outer request text/reference metadata. Only the original
inner `t3.inference` and `s3gen.inference` calls populate `native_calls`, which
drive Stop overlap. Each inner call synchronizes CUDA before entry and after
return/error; `host_returned_at` and synchronized `exit_at` are separate. No
per-layer profiler is active during inference. A one-shot profiler installs an
observer around the original audio-send function when original `main` begins,
then disables itself. This records synchronized allocator samples after actual
audio delivery, and parent settlement waits for that sample.

Production cancellation terminates and reaps the child. Interrupted native calls
are marked `completion_kind: process_terminated` only after process reap **and**
NVIDIA's compute process query excludes that exact PID. They do not claim a
natural synchronized native return. Successful calls retain
`completion_kind: native_return` and `cuda_synchronized: true`. Successor speech
uses the new real registered child. Final cleanup checks every launched PID and
its NVIDIA release receipt, not just the currently attached backend process.
After reap, `child_gpu_release_pending` retains observation ownership while
NVIDIA still reports the child PID. Settlement waits within `cleanup_timeout`;
a live orphan native call remains an error. GPU disappearance timestamps are
observations made by settlement, not a claim of release at the earlier Stop
return timestamp.

Parent RSS is separate from `child_memory`. Child receipts contain pre-model,
post-model, post-native, post-delivery and (on natural shutdown) pre-process-exit
CUDA allocated/reserved/current/peak samples. These are cumulative PyTorch
allocator counters per child, not total GPU memory or a universal leak test.
`nvidia-smi` release samples provide an independent process-context observation.
Killed children have no fabricated post-exit allocator snapshot.

The original complete audio/hash/frame/drain evidence and separate full-content
ASR workflow remain. Runtime pass still means content pending. Physical output
and human listening confirmation remain separate evidence.

Local controls: `python -m pytest DIR/test_observation.py -q` (8 passed).
They use synthetic models/records only; no speech, ASR, CUDA, NVIDIA command,
installation or remote operation is performed. Real host qualification remains
required, especially dependency compatibility and the actual child process path.

`probe_reference_ui.py` separately mounted the real pane and registered catalog
using a disposable private profile, invoked the supported picker callback and
Generate, and captured the admitted event before any synthesis handler. It saw
`default` and `custom` choices and an exact `custom:<selected_path>` request.
An import guard excluded Torch/Chatterbox runtime imports. Its local fixture file
was deliberately not audio, since this check concerns UI admission only; the
hardware run must use the real supplied synthetic WAV. The retained result is
`reference-ui-probe.log`. ADR-028 keeps legacy exact voice IDs distinct from
catalog availability; ADR-051's typed clone-reference profile initially applies
to audio.cpp, so this runner uses neither a fabricated catalog nor that profile
contract to route Chatterbox references.
