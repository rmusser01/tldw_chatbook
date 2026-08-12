---
id: TASK-15421
title: >-
  TTS Playground cannot express custom OpenAI model/voice and ignores saved
  exact preferences
status: Done
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
- [x] A user whose saved OpenAI selection is an exact non-catalog model/voice can generate in the TTS Playground with those exact values sent to the server
- [x] The playground's displayed model/voice never silently diverges from what its Generate request sends
- [x] Typed text in the Studio Exact model/voice ID inputs is visible while the field is focused
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: pure tests on `controls_from_catalog` (pin custom OpenAI model/voice as
   "(no catalog check)" options, generation allowed, no silent swap; scope
   guard: other legacy providers keep the fallback) + an integration test
   mounting `SpeechPlaygroundPane` with seeded exact custom axis values.
2. GREEN, three coupled layers, each found by the previous layer's test:
   catalog resolution (pin, mirroring the existing audio_cpp pin idiom),
   `_apply_catalog` seeding (legacy providers ignored saved axis values in
   favour of hardcoded `LEGACY_DEFAULT_MODELS`/`VOICES`), and the readiness
   gate (vetoed any OpenAI model outside the official catalog, leaving
   Generate disabled).
3. CSS: keep the exact-ID inputs' compact row visible under focus.
4. Live TUI verification against a request-recording mock server.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The saved custom selection was being discarded at THREE independent layers,
each of which had to be fixed for one Generate to work (each fix's test
exposed the next layer):

1. **`UI/stts_playground_catalog.py` `controls_from_catalog`**: for provider
   openai, a selected model/voice absent from the static official catalog is
   now pinned as an extra `"<id> (no catalog check)"` option (the established
   honesty wording) and stays selected, instead of being silently replaced by
   the first catalog entry; the official reference model supplies the
   format/speed shape. `_legacy_voices` grew a `pin_missing` flag (openai
   only). Scope: other legacy providers deliberately keep the fallback —
   their models are fixed by local engines, and a passing guard test pins
   that.
2. **`UI/Speech/speech_catalog_mixin.py` `_apply_catalog`**: legacy providers
   read hardcoded `LEGACY_DEFAULT_MODELS`/`LEGACY_DEFAULT_VOICES` where
   audio_cpp read the saved config — a new provider-guarded
   `_seeded_axis_value` consults the pane's seeded axis values/defaults
   (which already encode global→Studio precedence and exact-mode discipline
   via `_seed_axis_defaults`) before the hardcoded fallback. The
   display-label projection (`_legacy_voice_choices` replace) also carried
   the voice pin across instead of clobbering it.
2b. **Stale-carryover launder** (found by the pre-existing lifecycle tests
   going red): on a provider switch, the transient control snapshot and the
   Select-mirrored `axis_values` both pair the new provider id with the
   previous provider's still-displayed model/voice; the first-catalog
   fallback used to clean those up, and the pin would have preserved them.
   `_seeded_axis_value` therefore reads `axis_defaults` ONLY (written once
   at construction from saved config; the session dict can lie), and
   `_apply_catalog` launders non-catalog openai values that do not match
   that saved seed back to seed-or-default. The playground has no free-text
   entry, so the saved seed is the only legitimate custom id.

3. **`UI/Speech/speech_synthesis_mixin.py` `_generation_readiness_error`**:
   the final model-in-catalog check no longer vetoes openai (nothing can
   "disappear" from a static catalog; a non-catalog id there is the pinned
   custom one), which was keeping Generate disabled with a false staleness
   message.

Verified live against a request-recording mock OpenAI-compatible server:
the playground axes render "studio-model (no catalog check)" /
"mock-voice (no catalog check)" (Studio-over-global precedence intact, the
override marker correctly absent), and Generate sends exactly
`model=studio-model, voice=mock-voice` — displayed == sent. The result
panel reports the same. Tests: 3 new pure tests + extended integration test
(axes AND Generate enabled); the full speech/stts/playground UI sweep and
`Tests/TTS` pass.

**AC3 (exact-ID input visible while focused) is NOT closed.** Root cause of
the invisible text is a focus border re-added at higher specificity than the
compact row's `border: none` (`_forms.tcss` `Input:focus`, solid box glyphs
confirm it), which at `height: 1` paints only the border row. The added
`Input.speech-setting-control:focus` rule (0,2,1 — beats 0,1,1) fixes this
in every harness probe against the real bundle and the real
`SpeechSettingsPane` (blurred AND focused: height 1, no border, focus shown
via `$ds-input-focus-bg`), but the live TUI still draws the solid focus
border despite provably loading the same bundle — cascade-impossible with
the loaded stylesheet, so something else is in play (suspect: app-level
stylesheet/theme handling). Needs an instrumented `textual --dev` session;
evidence recorded here so the next attempt starts warm: border clears on
Tab (real blur), Escape does not blur, glyphs are `solid` not `tall`,
run_test cannot reproduce.
<!-- SECTION:NOTES:END -->

## AC3 Addendum (2026-08-11, follow-up session)

<!-- SECTION:NOTES2:BEGIN -->
The live-vs-run_test "divergence" dissolved: there was none. The mechanism
was the reset-tier accessibility rule `*:focus { outline: solid
$ds-focus-accent }` — Textual paints outlines OVER the widget's outermost
rendered lines (the rule's own comment warns of this), and a height-1
input's outermost line IS its only content line, so focus replaced the
typed value with the outline's box characters. The border rules were a red
herring throughout: every earlier probe asked `styles.border` (correctly
empty) and never captured a rendered frame — the obscuring was present in
run_test all along. Fix: `outline: none` added to the existing
`Input.speech-setting-control:focus` rule, keeping the `$ds-input-focus-bg`
tint as the content-safe focus cue (the opt-out + recolour pattern the
reset comment itself prescribes, TASK-1160 precedent). Pinned by a
rendered-frame regression test (`export_screenshot`) in
`Tests/UI/test_speech_live_render_defects.py` — the file for defects only a
live run exposed, which this almost was. Live-verified: typed text visible
while focused.
<!-- SECTION:NOTES2:END -->
