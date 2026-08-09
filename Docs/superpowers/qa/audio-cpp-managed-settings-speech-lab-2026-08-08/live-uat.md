# Managed audio.cpp Settings and Speech Lab interim live UAT

Date: 2026-08-08 (America/Los_Angeles)

Task: TASK-3795

## Scope and isolation

- This was an early compatibility run against audio.cpp 0.4. It does not
  satisfy the release-0.5 application-commit or configured-External-origin
  evidence gates in the governing managed-lifecycle specification.
- Ran the rebased feature branch through the browser-served Textual application with an isolated Chatbook profile and data directory.
- Used the Homebrew `audio-cpp` 0.4 prebuilt server and an isolated copy of an existing two-model `server.json`; the copy bound to a task-only loopback port.
- The pre-existing external audio.cpp process remained outside Chatbook ownership throughout the run. Pre/post artifact hashes matched, and its health response continued to report two models.
- Evidence below intentionally omits configured paths, ports, process IDs, roleplay text, raw diagnostics, model file locations, and profile data.

## First-time setup and generation

1. Opened Global Settings → Speech & TTS, chose `audio.cpp`, selected **Managed local server**, used binary detection, and selected the isolated `server.json` copy.
2. Saved the category. Validation completed without launching a process, opening the task listener, probing the server, refreshing the catalog, or synthesizing audio.
3. Opened Speech Lab from Settings. The runtime card showed the saved Managed generation pending over the active External generation and offered **Start & Test Connection**.
4. Started deliberately. Exactly one owned server process and one task listener appeared. The card reported Managed/Running/Available with saved and active generations matched.
5. The catalog exposed both configured models: `supertonic-3` and `pocket-tts-en`.
6. Generated a character-roleplay response with `supertonic-3`. Speech Lab preserved a complete current result and exposed Play, Stop, Export, and Save-as-profile actions.
7. Activated Play. The platform WAV player received the generated file.

Generated WAV metadata:

- Container: RIFF/WAVE
- Encoding: PCM, 16-bit, mono
- Sample rate: 44,100 Hz
- Data bytes: 534,956
- Duration: 6.065261 seconds
- File mode: owner-only (`0600`)
- User audible confirmation: **pending**

## Lifecycle and recovery

- **Save while running:** changed a managed health setting and saved. The owned process did not restart; Speech Lab showed the newer saved generation pending over the active generation, and the current WAV remained available.
- **Restart/apply:** selected **Restart & Apply Settings**. The old owned process was reaped, exactly one replacement appeared, saved/active generations matched, and the current WAV remained available.
- **Unexpected exit:** terminated only the exact task-owned process. Speech Lab moved to Unavailable, retained the current WAV, and offered recovery. **Start & Test Connection** created exactly one replacement and restored Running/Available.
- **Explicit shutdown/lazy restart:** **Shut down server** removed the process and listener, preserved the current WAV, and disabled Generate. The next deliberate **Start & Test Connection** started exactly one process.
- **External apply:** saved External mode while Managed remained active. Saving did not stop or relaunch the child; Speech Lab truthfully showed External saved over Managed active. **Apply Settings & Stop Managed Server** then reaped the owned child, closed the task listener, and moved the card to External without launching another process.
- **Dormant Managed settings:** switched back to Managed and confirmed the previously selected binary and `server.json` remained populated. Saving again launched nothing. A deliberate start created one child.
- **Final app cleanup:** closing Chatbook reaped that final owned child and closed its listener. The pre-existing external process remained healthy. The browser harness and local serving process were then stopped, leaving no task-owned listener or audio.cpp process.

## Result

This interim 0.4 run exercised the objective first-time setup, catalog, generation, lifecycle, recovery, ownership, preservation, and cleanup journeys. The final Managed re-entry exposed a primary action whose value was correct but whose live width still matched its shorter initial label; that issue was fixed with an explicit layout refresh and a rendered-geometry regression. Release-gate completion still requires a fresh supported-release run that records the exact application commit, proves External traffic used only the configured origin, and obtains the user's subjective audible confirmation.
