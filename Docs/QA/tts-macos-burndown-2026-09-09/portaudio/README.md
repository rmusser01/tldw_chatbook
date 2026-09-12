# PortAudio CoreAudio qualification — 9 September 2026

The corrected `controls-02` run passed all six production `StreamingPcmSink` controls using the pinned private PortAudio candidate. The worker exited zero in **20.56 seconds**; every stream, notification thread, queued notification and live-sink reference was released. Its exact PID was absent after exit. [Runtime receipt](runs/controls-02/evidence.json), [controller](runs/controls-02/controller.json), [release check](runs/controls-02/ownership-release.json).

| Control | Cycle 1 | Cycle 2 |
|---|---|---|
| Complete first saved Higgs WAV | 117,120 source frames; identical callback PCM hash | 117,120 source frames; identical callback PCM hash |
| Stop during second WAV, repeated three times | 0.116791 s; 11.72 s still queued | 0.111009 s; 11.72 s still queued |
| Complete successor second WAV | 97,920 source frames; identical callback PCM hash | 97,920 source frames; identical callback PCM hash |

The actual output device was **MacBook Pro Speakers**, using sounddevice 0.5.6 and NumPy 1.26.4 in the dedicated CPython 3.12.11 environment. Both Stops began after at least half a second of nonzero callback frames. Every native `abort()` and `close()` returned; each phase recorded the correct `SinkDrained` or `SinkStopped` event and zero values for all five resource counters. These observations cover physical callback delivery, not an acoustic recording. The six finite controls do not establish an absence of all timing races or memory leaks. [Derived summary](summary.json).

The library containing the actual bound `Pa_GetVersion` function was resolved with `dladdr` to the candidate prefix's `libportaudio.19.8.dylib`. Its SHA256 was **48d9fcaff582ff65c8a51d67570bf69ce900d51278841d563e19b7ef73746586**, before and after the controls. The reported version string still says `PortAudio V19.7.0-devel, revision unknown`; that string alone cannot identify this binary. The five relevant application source files also retained identical start/end hashes. No model was initialized.

The candidate source is `1240f06f0227ae3d0b4967f41012221c3d7cb6cf`, built with Apple Clang 17 for arm64 in Release mode. Configuration, build and installation into the **task-owned prefix** all exited zero. [Build receipt](provenance/build-receipt.json), [upstream patch](upstream/1175.patch), [license](upstream/LICENSE.txt). [Upstream PR #1175](https://github.com/PortAudio/portaudio/pull/1175) remained open and unmerged at the recorded check. Its deferred property-listener handling addresses the lock inversion reported in [issue #1174](https://github.com/PortAudio/portaudio/issues/1174). [Status and compiler provenance](provenance/upstream-status.json).

Both earlier full Higgs attempts remain **failed cleanup**: workers 4020 and 8182 completed generation and delivered the complete Console PCM, but never emitted the terminal drain notification and required owned termination. Their distinct native samples show the opposing `FinishStoppingStream`/`AudioOutputUnitStop` and `startStopCallback`/`AudioUnitGetProperty` paths. This is consistent with the upstream lock-order diagnosis; the controls are finite supporting evidence. [Original Higgs report](../providers/higgs/README.md), [first native sample](../providers/higgs/runs/live-wav-01/worker-sample.txt), [second native sample](../providers/higgs/runs/live-wav-02/worker-sample.txt), [runtime/library comparison](../providers/higgs/provenance/portaudio-comparison.json). Passing source-content checks in those runs did not turn native teardown failures into successes.

The first candidate attempt, `controls-01`, passed all sink controls and exited zero, but set only XDG variables. Importing the sink also imported application configuration, which read the default user profile and ensured its chat-dictionary directory existed. That run is **isolation-limited** and is not the final AC2 receipt. Its [runtime](runs/controls-01/evidence.json), [controller](runs/controls-01/controller.json), [release record](runs/controls-01/ownership-release.json) and [original launcher](recipes/controls-01.py.txt) remain unchanged; [raw stderr](/private/tmp/tts-macos-burndown/portaudio/runs/controls-01/stderr.log) retains the observation.

The corrected launcher writes `TLDW_CONFIG_PATH`, `paths.data_dir` and explicit database settings before application import. It verifies all resolved paths and installs a Python audit guard against access to the default configuration/data directories. The [import-only check](isolation/evidence.json) exited zero without importing sounddevice or a model runtime, and [its controller](isolation/controller.json) joined the child. `controls-02` passed with that guard active and no denied accesses. No user profile contents are copied into this package.

Exact commands used, with the prepared task-local runtime and candidate prefix:

```sh
python3 /private/tmp/tts-macos-burndown/portaudio/qualify_sink_controls.py --output /private/tmp/tts-macos-burndown/portaudio/runs/isolation-preflight-02 --check-isolation
python3 /private/tmp/tts-macos-burndown/portaudio/qualify_sink_controls.py --output /private/tmp/tts-macos-burndown/portaudio/runs/controls-02 --play-audio
```

Another run needs fresh output directories and an explicitly granted exclusive audio slot. The [corrected launcher](recipes/controls-02.py.txt) retains a 90-second worker deadline, exact-PID sampling on timeout and bounded terminate/kill/join cleanup; forced termination cannot pass. [Original preparation](provenance/controls-preparation.json), [corrected preparation](provenance/controls-02-preparation.json).

This package qualifies the sink controls in [TASK-32165](../../../../backlog/tasks/task-32165%20-%20Qualify-the-PortAudio-CoreAudio-stop-deadlock-repair-on-macOS-ARM.md). Full repaired-library Higgs cancellation, successor recovery and content checks were still separate work when it was curated. The candidate has not been installed system-wide or adopted as a default runtime. Adoption requires its own runtime/dependency decision; the existing [ADR-023 ownership boundary](../../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md) is unchanged.

[Copy provenance and raw local artifact hashes](package-provenance.json), [verification](verification.json), [package index](package-index.json). Audio, models, binaries, profiles and source archives are omitted. Full build logs, device input WAVs and original native samples remain at the local evidence paths recorded in provenance.
