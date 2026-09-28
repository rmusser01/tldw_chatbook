---
id: TASK-32928
title: OmniVoice ONNX int8hq standalone TTS backend
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-24 02:35'
updated_date: '2026-09-25 03:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Docs/superpowers/specs/2026-09-23-omnivoice-onnx-tts-backend-design.md per ADR-180
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 omnivoice generates speech in Speech Lab
- [x] #2 cloning works from a reference profile
- [x] #3 artifact installs via model browser with license consent
- [x] #4 targeted tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-09-23-omnivoice-onnx-tts-backend.md task-by-task (12 TDD tasks; ADR-180 is the decision record)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All 12 plan tasks implemented on branch feat/omnivoice-onnx-tts (ADR-180; spec 2026-09-23; plan 2026-09-23). 87 unit tests + 2 env-gated integration tests green. PENDING: manual live verification (install artifact via model browser, run Tests/TTS/test_omnivoice_integration.py with OMNIVOICE_ONNX_ROOT set, listen to outputs vs model-card demos) — owner: Robert. Subagent quota exhausted mid-run; Tasks 8-12 controller-implemented inline with ledgered reviews (see .superpowers/sdd ledger + final review pending).

Review + UAT pass (2026-09-24, PR #2825 review). Real-model UAT against ct03/omnivoice-onnx-int8hq@65c840ba found the engine could not synthesize at all; fixed, with regressions pinned in Tests/TTS/test_omnivoice_review_fixes.py (each verified to FAIL on the original PR head):
- tokenizers.Tokenizer.encode returns an Encoding, not list[int] -> every real synthesis raised TypeError (fakes returned lists).
- The ct03 LM declares audio_mask/position_ids as (batch, seq) and the encoder audio as (batch, 1, samples); the adapters fed rank-3/rank-2 -> ORT INVALID_ARGUMENT. Adapters now honour declared ranks; static feeds cached per request.
- Loudness/reference handling now matches upstream (boost refs quieter than 0.1 RMS before encoding, scale back after; trim to whole codec frames; add_punctuation on the transcript). "auto"/"none" language -> the prompt's None.
- Timeout budget scales with the whole sequence (reference included), 30 s floor: a 19.5 s reference + 1.7 s line took 32.5 s wall (old budget ~10 s).
- Task cancellation now signals the sampler and waits for the thread before releasing the single-flight lock; close()-cancel raises CancelledError instead of a silent empty stream.
- Cloning without a transcript is refused (request_invalid).
- Settings Save for OmniVoice persisted nothing (no _TTS_SETTING_BINDINGS) and refused a blank model_root; Speech Lab generation was rejected by request admission (numeric options range-checked to [0,1]); the Lab's pre-filled knobs overrode Settings. All fixed.
Live evidence (tmux, isolated HOME profile): Settings save round-trips to [OmniVoiceSettings]; Lab generates (Whisper transcript exact); profile clone F0 115 Hz vs reference 111 Hz (default voice 231 Hz); bad knob value warns without crashing. Script-level: RTF 2.3-2.5 plain, ~4 cloning on this Mac; Spanish exact.
AC #3 verified live: Lab ▸ Models ▸ Curated ▸ omnivoice-onnx-int8hq ▸ Review and install… ▸ Install downloaded 1,081.9 MiB, verified every pinned SHA-256 and promoted the artifact; with model_root blank the engine resolves the managed copy and synthesis transcribes exactly. (An earlier "tmux cannot click it" note was wrong: the click targeted the wrong column, and later misses were a zsh no-word-split bug in the driver script. In-process Pilot clicks on the modal registered 10/10.)
Driving the flow exposed three UX defects, fixed here: (1) the consent panel never rendered the descriptor's usage_notice, so the user saw only "License: other" -- now a "Terms:" line (shared ModelPlanPanel, benefits every curated model); (2) `outline: heavy` on one-row curated/installed buttons painted over the label, so a focused button read as an empty bar -- now a fill + underline focus style; (3) declining at the consent modal recomposed the list and dropped the user at the top of a 51-card catalog -- focus now returns to the declined row.
License correction: the LM weights are Apache-2.0 (model card, GitHub repo, ct03 export README, and this repo's audio.cpp catalog entry all agree), not CC-BY-NC; the binding terms are the Boson Higgs Audio 2 Community License on the required audio tokenizer (100,000 annual-active-user cap, Llama 3 AUP, "Built with Higgs Materials" attribution). Consent text, user guide, spec and ADR-180 updated.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-32903, colliding with the older
"Scope-corrections-to-eight-existing-tier-1-tasks" task (added 2026-09-21
22:35 in 418eeca1d2, "file the tier-2 code review work streams"), which
arrived first. This task was added 2026-09-23 19:36 (873f7e23a3, ADR-180
commit on the OmniVoice branch, PR #2825). Per the owner rule decided
2026-08-21 in TASK-19601 (**the older arrival keeps the id regardless of
status; the younger task renumbers with a provenance note**), it renumbered
to TASK-32928 — one above the highest id on any ref at renumbering time
(32927). The other TASK-32903 holder keeps the id.
