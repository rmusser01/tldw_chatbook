---
id: TASK-32149
title: Qualify non-English Kokoro speech on macOS ARM
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 05:34'
updated_date: '2026-09-09 09:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exercise the supported non-English Kokoro voices with real local model output and playback so language-routing fixtures are backed by language-appropriate evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Available Spanish, French, Hindi, Italian and Portuguese voices generate and play through real supported Kokoro engines with exact voice and language provenance.
- [x] #2 Japanese and Chinese runtime extras and local speech are exercised where the host can support them, with actionable failures fixed or concretely tracked.
- [x] #3 Complete source and playback evidence plus language-capable independent transcription preserve raw differences and do not claim pronunciation quality from an English-only recognizer.
- [x] #4 Targeted regressions cover any discovered application defects and remaining asset, provider or language-quality blockers have explicit backlog records.
- [x] #5 French aliases and Japanese/Mandarin frontends reach the selected ONNX model correctly; missing language dependencies or Japanese dictionary data have actionable setup errors, and cancellation during phonemization cannot start later inference.
- [ ] #6 Concurrent Japanese or Mandarin ONNX requests do not construct duplicate cached frontends or invoke a shared frontend concurrently; cancellation and backend close retain real frontend work until it finishes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/142-kokoro-east-asian-phonemization-across-engines.md; extends ADR-140 and ADR-023.
Reason: Real ONNX Japanese output exposed the need to select the model-aligned optional Misaki frontend while preserving the selected inference engine and dependency isolation.

1. Inventory local language voices, runtime extras and supported language aliases; preserve pinned asset/package provenance and isolated configuration.
2. Provision only missing public language assets/extras and a local multilingual recognizer in task-owned caches, keeping playback and heavy inference serialized.
3. Exercise Spanish, French, Hindi, Italian and Portuguese, then Japanese and Chinese where feasible, through production resolution/generation and physical playback.
4. Independently transcribe complete clips with a multilingual recognizer, retain source/PCM hashes and raw differences, and separate transport success from pronunciation/content uncertainty.
5. For each discovered application defect, add a failing targeted regression before a minimal fix; record concrete remaining limitations in Backlog and the qualification report.
6. Add failure-first regressions for French aliases, model-aligned ONNX Japanese/Mandarin phonemes, missing Japanese dictionary guidance, and cancellation during frontend work. Implement ADR-142 within the retained worker and rerun affected live languages. Preserve initial failures and check lexical differences with an additional local multilingual recognizer where useful.

7. PR #2545 review: reproduce concurrent frontend construction/invocation with controlled threads, serialize frontend cache construction and use inside retained native workers, document the dictionary error contract, and verify responsive event-loop cancellation and join behavior. ADR required: no new ADR; implement the retained frontend ownership already specified by ADR-142 and ADR-023.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified all fourteen CPU engine/language tuples for Spanish, French, Hindi, Italian, Brazilian Portuguese, Japanese and Mandarin. Preserved the baseline's three runtime failures and Japanese ONNX character-description output. Added model-aligned optional Misaki Japanese/Mandarin phonemization for ONNX, normalized French to fr-fr, and supplied fixed setup guidance for missing extras and UniDic data. Work remains in retained workers; cancellation during G2P prevents later inference. ADR required: yes; implemented backlog/decisions/142-kokoro-east-asian-phonemization-across-engines.md within ADR-023/140.

Real repaired-language playback passed after rebase, with exact engine/voice/model/dictionary provenance and clean process exits. Complete small and medium ASR receipts preserve all differences: medium matches both French engines and Portuguese; Mandarin differs only by Traditional/Simplified script, while Hindi/Japanese content requires native-language review. The strict verifier is unchanged and no general pronunciation claim is made. Separate backlog work records unresolved language quality. Failure-first language/lifecycle tests and independent ten-case review probes pass. Docs/QA/tts-macos-burndown-2026-09-09/kokoro-languages/README.md retains all original/final attempts, raw transcripts, provisioning corrections and release evidence; the developer runbook documents setup.
<!-- SECTION:NOTES:END -->
