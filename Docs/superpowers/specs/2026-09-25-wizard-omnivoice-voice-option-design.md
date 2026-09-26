# First-run wizard: OmniVoice as a Voice option — design

Status: approved in conversation 2026-09-25; awaiting written-spec review.

## Goal

A new user can pick a fully local, offline voice — no server, no account —
in the first-run wizard's **Voice** step, alongside PocketTTS, Official
OpenAI and Custom compatible. Success: after the wizard, OmniVoice is the
default TTS provider, its model is installed, and it actually speaks — with
a stable voice across replies.

Out of scope: choosing OmniVoice voice profiles or language in the wizard
(Speech Lab owns that); a pip-install button (the step shows the command,
as the RAG and Speech steps do); refactoring the Speech step's install
machinery.

## Context (as found on dev 461668df00)

- `VoiceSetupStep` (`UI/Wizards/FirstRunSetupWizard.py`) is built around
  OpenAI-compatible HTTP endpoints; its pure state lives in
  `UI/Wizards/first_run_voice_step_state.py` (`VoiceSetupDraft`,
  presets, `build_voice_setup_save_event`, `run_voice_sample`). It saves
  on Next via `commit()` → `STTSSettingsSaveEvent` and awaits the reply.
- OmniVoice (PR #2825) is an in-process provider needing the
  `omnivoice_tts` pip extra and a 1.1 GB curated artifact
  (`TTS/omnivoice_artifact_catalog.py`), installed through
  `ArtifactAcquisitionService.preflight/provision` with the shared
  `ModelInstallModal` (consent + Terms) and `ModelInstallProgress`.
- `SpeechSetupStep` already installs a curated model (Parakeet) in the
  wizard via thin `run_parakeet_preflight/provision` wrappers and a
  `@work(thread=True, group=..., exclusive=True, exit_on_error=False)`
  install worker; Next never blocks on install state.
- Without a reference clip OmniVoice improvises a voice per request (UAT:
  178 Hz vs 231 Hz on two default-voice runs). The engine reads
  `[OmniVoiceSettings] seed`, but no Settings binding writes it yet.

## Design (approach A: a fourth service in the Voice step)

### User-facing flow

The service row gains **OmniVoice** (short label; the subtitle says it is
local). Choosing it hides the endpoint/model/voice fields and the Advanced
disclosure and shows a local panel; choosing another service reverses that.
The step subtitle becomes "…PocketTTS or OmniVoice run locally, no account
needed…".

Panel states, read off the UI thread when the panel is shown ("Checking…"
while reading):

1. **engine_missing** — the extra is absent: "OmniVoice needs its local
   engine: `pip install "tldw_chatbook[omnivoice_tts]"`, then run setup
   again from Settings ▸ Diagnostics ▸ Run Setup Wizard." Install button
   visible but disabled.
2. **model_missing** — "Downloads the OmniVoice model (1.1 GB, one time)."
   plus **Install voice model** → shared consent dialog (with the Terms
   line) → shared progress bar.
3. **ready** — engine and model present (managed install or a configured
   `model_root`): "OmniVoice is installed — runs offline on this computer."

**Test and Hear** is enabled only when ready; while generating it says
"Generating locally (first run loads the model)…". **Use as default** +
Next makes OmniVoice the default provider and saves a fixed seed (below).
Skip, decline, and failures never block Next.

### Components

1. `TTS/omnivoice_artifact_catalog.py` (install plumbing, no UI)
   - `omnivoice_setup_state() -> Literal["engine_missing", "model_missing", "ready"]`:
     extra probed with `find_spec` (no import); model resolved by the
     engine's own rules (configured `model_root` layout, else the active
     managed artifact).
   - `run_omnivoice_preflight()` / `run_omnivoice_provision(report, *, progress)`:
     thin `ArtifactAcquisitionService` wrappers passing the pinned
     `omnivoice_onnx_source_map()`, mirroring the Parakeet wrappers.
2. `UI/Wizards/first_run_voice_step_state.py` (pure state)
   - `VOICE_PRESET_OMNIVOICE`.
   - `build_omnivoice_save_event(*, speed, seed, request_id, reply_to)`:
     `STTSSettingsSaveEvent` whose settings contain **only**
     `OMNIVOICE_SEED` (no endpoint/credential keys) and whose preferences
     are `provider_id="omnivoice"`, model `omnivoice-int8hq`, voice
     `default`, format `wav`, `commit_defaults_after_handoff=True`.
   - `run_omnivoice_sample(text, *, speed, seed, service)`: synthesizes through
     the app's TTS service —
     `service.generate_audio_stream(build_legacy_speech_request(...))`,
     the path briefing audio uses — so the shared cached backend is used
     (no second ~1.2 GB engine) and the real runtime path is proven.
     Returns the existing `VoiceSampleResult` (wav).
3. `Event_Handlers/STTS_Events/stts_events.py`
   - `OMNIVOICE_SEED` → `[OmniVoiceSettings] seed` Settings binding
     (provider `omnivoice`).
4. `VoiceSetupStep` (`UI/Wizards/FirstRunSetupWizard.py`)
   - Fourth radio, local panel (status `Static`, install `Button`,
     `ModelInstallProgress`), preset switching, off-thread state read on
     show, install = preflight worker → `ModelInstallModal` → provision
     thread worker (the Speech step's worker shape) → state re-read.
   - OmniVoice branches in **Test and Hear** and `commit()`; `commit()`
     posts the OmniVoice save event only when "Use as default" is checked
     (nothing else to save) and otherwise returns success without a save.

The Speech step, the model browser and the engine are unchanged.

### Voice stability (the seed)

When OmniVoice is saved as default the wizard writes a fixed seed so the
default voice does not change per reply. Rule: if `[OmniVoiceSettings]
seed` is already set, keep it (a re-run never changes an existing voice);
otherwise generate one random integer in `[0, 2**31)` the first time the
step needs it. The **Test and Hear sample uses the same seed** (passed per
request as `extra_params["seed"]`, which the engine and request admission
accept), so the voice the user hears is the voice they save. (Added during
planning: without this the sample would use a different random voice.)
**Plan task 1 measures it with the real model** (one seed, several
different sentences: median F0 spread and Whisper intelligibility). If the
voice holds, ship the seed. If it does not, keep the seed (still reduces
drift) and change the ready-state copy to: "Replies may vary in voice
until you create a voice profile in Voice Cloning." — never claim a
stability the measurement did not show.

**Measurement (2026-09-25):** Ran the plan task 1 probe against
`ct03/omnivoice-onnx-int8hq` revision `65c840ba966f4b50cd6bd73f234fb1eba72f9a16`
(32 diffusion steps, English), synthesizing 4 sentences once with
`OMNIVOICE_SEED=1234567` and once with no seed, and transcribing each clip with
faster-whisper ("base", int8, CPU). Median F0 per clip:
- Seeded: `[164, 101, 127, 106]` Hz — spread (max−min)/mean = **50.3%**
- Unseeded: `[135, 213, 215, 110]` Hz — spread = **62.8%**

Per-sentence transcript agreement (both runs): sentence 1 ("Good morning,
here is your daily summary.") and sentence 2 ("The meeting has been moved
to three o'clock on Thursday.") transcribed correctly modulo punctuation
and digit/word numerals; sentence 3 ("I found four articles that match
your search.") transcribed exactly; sentence 4 ("Would you like me to read
the next section aloud?") was mis-transcribed as "...allowed?" in **both**
the seeded and unseeded run — an identical Whisper homophone slip in both
conditions, not something the seed changed.

Decision by the stated rule (seeded spread ≤ 12% AND ≤ half the unseeded
spread AND every transcript matches): the seeded spread (50.3%) is far
above the 12% ceiling, and the seed only narrows the spread from 62.8% to
50.3% (not to half), so the first two conditions already fail —
**`SEED_STABLE = False`**. The numbers are unambiguous; no clips needed to
be listened to. Per the fallback above, the wizard keeps the fixed seed
(it still reduces drift, 62.8% → 50.3%) but the ready-state copy is:
"Replies may vary in voice until you create a voice profile in Voice
Cloning."

### Errors and concurrency

- Preflight failure (offline, disk) → shared `install_failure_message`
  text + Retry. Consent declined → back to model_missing silently.
  Download/verification failure → failure text + Retry (acquisition cleans
  partial files). Sample failure → "Couldn't play a test sample — you can
  still save and test later in Speech Lab." Save failure → the existing
  save-result reply handling.
- Install and sample workers are grouped with `exit_on_error=False`; Test
  and Hear is disabled during an install; switching presets or leaving the
  step cancels an in-flight sample (existing behaviour) but not a download,
  which continues as in the Speech step and is re-read when the panel is
  shown again.
- A concurrent Parakeet and OmniVoice install target different artifacts
  in the same store (per-artifact locks); verified once in live UAT.

## Testing

- Units (no app): `omnivoice_setup_state` for all three outcomes;
  `build_omnivoice_save_event` shape (provider, format, seed present, no
  endpoint/credential keys); wrappers pass the pinned sources through
  (fake acquisition service); the `OMNIVOICE_SEED` binding maps to
  `[OmniVoiceSettings] seed`.
- Step tests (wizard harness): preset switch swaps panels; each state
  renders the right copy and button availability; install runs
  consent → provision → ready with fakes; `commit()` posts the OmniVoice
  event only with "Use as default"; Next is never blocked.
- Live UAT: fresh profile → wizard → OmniVoice → install → Test and Hear
  (Whisper) → save → Speech Lab defaults to OmniVoice with the seed; the
  Voice step at 80×24 (shorten "Custom compatible" to "Custom" if it
  clips); one concurrent Parakeet + OmniVoice install.
- Update the first-run setup page in `Docs/User_Guide/` (content +
  "Verified against" stamp).
