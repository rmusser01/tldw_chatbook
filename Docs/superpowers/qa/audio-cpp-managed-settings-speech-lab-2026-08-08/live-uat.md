# Managed audio.cpp Settings and Speech Lab live UAT

Date: 2026-08-09 (America/Los_Angeles)

Task: TASK-3795

Application commit: `2d9fe9401a6182e113102d903a6483553bcc8fde`

## Scope and isolation

- Ran the supported Homebrew `audio-cpp` 0.5.1 prebuilt server through the browser-served Textual application with an isolated Chatbook profile and data directory.
- Used an isolated copy of an existing two-model `server.json`; the copy bound to a task-only loopback origin and the original remained untouched.
- Used separate task-only loopback origins for Managed and External verification. The pre-existing user-owned audio.cpp process remained outside Chatbook ownership and healthy throughout the run.
- Pre/post hashes of the user-provided binary and original `server.json` matched.
- Evidence below intentionally omits configured paths, ports, process IDs, roleplay text, raw diagnostics, model file locations, and profile data.

## First-time setup and generation

1. Opened Global Settings → Speech & TTS, chose `audio.cpp`, selected **Managed local server**, used binary detection, and selected the isolated `server.json` copy.
2. Saved the category. Validation completed without launching a process, opening the task listener, probing the server, refreshing the catalog, or synthesizing audio.
3. Opened Speech Lab from Settings. The runtime card showed the saved Managed generation pending over the active External generation and offered **Start & Test Connection**.
4. Started deliberately. Exactly one owned server process and one Managed listener appeared. The card reported Managed/Running/Available with saved and active generations matched.
5. The catalog exposed both configured models: `supertonic-3` and `pocket-tts-en`.
6. Generated a character-roleplay response with `supertonic-3`. Speech Lab preserved a complete current result and exposed Play, Stop, Export, and Save-as-profile actions.
7. Exported the complete result and activated Play through the Speech Lab current-result controls.

Generated WAV metadata:

- Container: RIFF/WAVE
- Encoding: PCM, 16-bit, mono
- Sample rate: 44,100 Hz
- File bytes: 716,794
- Audio data bytes: 716,750
- Duration: 8.126417 seconds
- File mode: owner-only (`0600`)
- User audible confirmation: **confirmed during the live UAT**

## Lifecycle and recovery

- **Save while running:** changed the managed health interval and saved. The original owned process remained the sole listener; Settings stated that active configuration was unchanged; Speech Lab showed the newer saved generation pending; and the current WAV remained available.
- **Restart/apply:** selected **Restart & Apply Settings**. The old owned process was reaped, exactly one replacement appeared, saved and active generations matched, and the current WAV remained available.
- **Unexpected exit:** terminated only the exact task-owned process. Speech Lab moved to Unavailable, retained the current WAV, did not retry in the background, and offered recovery. **Start & Test Connection** created exactly one replacement and restored Running/Available.
- **Explicit shutdown/lazy restart:** **Shut down server** removed the process and listener, preserved the current WAV, and disabled Generate. The next deliberate **Start & Test Connection** started exactly one process.
- **External apply:** started a separate task-owned 0.5.1 External server, then saved External mode while Managed remained active. Saving did not stop or relaunch the managed child. Speech Lab truthfully showed External saved over Managed active; **Apply Settings & Stop Managed Server** then reaped the owned child, closed the Managed listener, and moved the card to External without launching a managed replacement.
- **Configured External origin:** with no Managed listener present, **Test Connection** succeeded against the configured task-only External origin. Generate then produced a new complete WAV while the External harness emitted model-execution activity. The task-only External server was the sole available application target during both operations.
- **Dormant Managed settings:** switched back to Managed and confirmed the previously selected binary and `server.json` remained populated. Saving again launched nothing. A deliberate **Start & Test Connection** created exactly one managed child while the current WAV remained available.
- **Final app cleanup:** closing Chatbook reaped the final owned managed child and closed its listener. The separate External harness and browser-serving process were then stopped. No task-owned listener or audio.cpp process remained, while the pre-existing user-owned process remained healthy.

## Result

The supported audio.cpp 0.5.1 run passed the objective first-time setup, validation-only save, multi-model discovery, complete-WAV generation and audible playback, saved-versus-active handoff, restart/apply, unexpected-exit recovery, explicit shutdown/lazy restart, configured-External-origin, dormant-value, ownership, preservation, artifact-integrity, and final-cleanup journeys at the exact application commit above. The user confirmed hearing the generated Speech Lab result.
