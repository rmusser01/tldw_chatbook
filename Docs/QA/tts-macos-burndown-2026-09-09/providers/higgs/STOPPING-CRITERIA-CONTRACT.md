# Pinned Higgs V2 Stop contract

The underlying model supports a per-request `StoppingCriteriaList`, but the selected serving engine does not expose that argument. This is a read-only source finding, not a tested cancellation improvement. Exact file hashes and AST signatures are in `stopping-criteria-contract.json`.

The actual CPU run `live-wav-portaudio-02` retained its native ownership correctly, but Stop took 96.400 seconds while the in-flight generate call completed 512 forwards. No cancelled audio escaped and the successor started after that call exited.

At pinned Higgs revision `05a145bb490501b534563bf51bf2f7aa2326b271`:

- [`HiggsAudioServeEngine.generate`](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/serve/serve_engine.py#L341) has a closed signature. It accepts `stop_strings`, but no stop event, `stopping_criteria`, or arbitrary keyword arguments. Its call to `self.model.generate` passes a fixed argument set.
- The same source defines [`AsyncStoppingCriteria(threading.Event)`](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/serve/serve_engine.py#L153). It checks `is_set()` and does not depend on scores, but this class is unused in that engine file.
- [`HiggsAudioModel.generate`](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/model/higgs_audio/modeling_higgs_audio.py#L1933) accepts `**kwargs` and forwards them to `super().generate`. Installed Transformers 4.46.3 explicitly documents the `stopping_criteria` argument, combines it with the default criteria, and passes the resulting list to `_sample`.
- The custom Higgs [`_sample` loop](https://github.com/boson-ai/higgs-audio/blob/05a145bb490501b534563bf51bf2f7aa2326b271/boson_multimodal/model/higgs_audio/modeling_higgs_audio.py#L1908) evaluates that list after each real forward and token update. On this CPU path, a true criterion ends the loop before another normal iteration. It cannot interrupt the forward already running; an event set before the first iteration also needs an entry check to avoid starting unnecessary work.

An early stop needs more than a keyword pass-through. The loop can return no audio sequences before audio generation begins. The serving engine handles an empty list by setting the waveform to `None`, then unconditionally indexes `outputs[1][0]` at line 411. It also decodes partial delayed audio tokens before returning. A change must qualify that unwind, suppress cancelled partial audio and cancellation-related exceptions, reset per-request state before a successor, and keep the native lease until the actual function exits.

A request-owned, version-bounded bridge to the underlying documented criterion or a serving API extension is plausible. This would expand the currently selected engine call contract and needs its own design/test review. Do not change process-global generation defaults or infer that the existing joined Stop is responsive.
