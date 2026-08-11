---
id: TASK-15421
title: >-
  TTS Playground cannot express custom OpenAI model/voice and ignores saved
  exact preferences
status: To Do
assignee: []
created_date: '2026-08-11 12:00'
labels:
  - tts
  - speech
  - lab
  - ux
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT on `origin/dev` `82b595049` (2026-08-11), custom OpenAI Base URL
pointing at a mock OpenAI-compatible server.

The Speech Lab TTS Playground's quick controls for provider OpenAI are closed
catalogs: Model offers only "TTS-1 (Standard)" / "TTS-1-HD (High Quality)" and
Voice only the six official OpenAI voices. There is no way to type a custom
model or voice name, even though:

- Settings ▸ Speech & TTS Global defaults were saved as Exact `mock-model` /
  Exact `mock-voice`, and
- Studio TTS Preferences were saved as Model policy Exact with Exact model ID
  `studio-model` (the pane itself correctly shows the inherited global values).

On Generate the playground sent `model: "tts-1", voice: "alloy"` — its dropdown
defaults — on first open, after Refresh, after saving studio preferences, and
after a full app restart. **Format and Speed do seed from the saved preferences
(Format showed WAV), which makes the model/voice divergence look like the saved
settings simply don't work.** Against a server that rejects unknown model names
(the docs' pocket-tts case in reverse), the playground can never produce audio;
against a permissive server it silently tests the wrong model/voice. The
playground worked fine at the transport level: requests hit the custom Base URL
keyless with no org header.

Two candidate shapes (pick one deliberately): seed the quick controls from the
effective exact selection and render non-catalog values as selectable entries,
or add an explicit free-text "Exact…" affordance mirroring the Studio pane's
Exact model/voice ID fields.

Minor adjacent polish seen on the Studio pane: the "Exact model ID" /
"Exact voice ID" inputs render as a 2-row bordered box whose content line is
invisible while focused (typed text only appears after focus leaves — it had
silently accepted a double paste as `studio-modelstudio-model`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A user whose saved OpenAI selection is an exact non-catalog model/voice can generate in the TTS Playground with those exact values sent to the server
- [ ] The playground's displayed model/voice never silently diverges from what its Generate request sends
- [ ] Typed text in the Studio Exact model/voice ID inputs is visible while the field is focused
<!-- AC:END -->
