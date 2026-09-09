---
id: TASK-32159
title: Honor configured Kokoro engine and voices in the Speech Lab
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 06:09'
updated_date: '2026-09-09 14:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live qualification found the fresh Speech Lab ONNX switch overrides the configured PyTorch engine. Official non-English voice choices also need to remain selectable through normal catalog projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh Speech Lab uses the configured Kokoro ONNX or PyTorch engine, and the generated request matches the displayed switch; an explicit session toggle still overrides it.
- [x] #2 Official available Kokoro language voices can be selected without being silently replaced by an English default during catalog projection.
- [x] #3 Mounted targeted regressions and real speech validation cover inherited engine and non-English voice selection; saved global configuration is not changed by Lab use.
- [x] #4 The explicit Kokoro language selector includes Hindi, Italian and Brazilian Portuguese as well as the existing supported languages, and emits the selected language with its voice.
- [x] #5 The Settings Kokoro default-voice selector uses the same complete official catalog as the Speech Lab and preserves non-English configured selections.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-039 Global/Studio TTS ownership and ADR-140 official Kokoro runtime apply.
Reason: Repair inherited initial controls and the static official voice catalog within existing settings and provider boundaries.

1. Add a mounted failing regression for fresh Kokoro engine inheritance and verify exact official voice IDs against pinned runtime assets.
2. Seed the session engine switch from its configured setting while preserving explicit user toggles; extend the existing static voice choices for official supported language voices as necessary.
3. Exercise catalog projection and actual generated request, including a non-English configured voice; ensure no settings write or generation occurs during mounting.
4. Rerun targeted UI tests and serialized real CPU/MPS/ONNX playback using the corrected harness that applies the same axis seeds as the production Lab owner.

5. PR #2545 review: reproduce the stale Settings voice list with a mounted selector regression, use the shared official catalog, and verify non-English selection persistence plus existing engine inheritance. ADR required: no; apply existing ADR-039/140 without creating a new settings surface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The fresh Speech Lab engine switch now inherits the configured Kokoro engine while preserving explicit session toggles. The static catalog contains the 54 official voices, and the language selector exposes all nine supported language choices including Hindi, Italian and Brazilian Portuguese. Catalog projection preserves the requested voice and generated engine/language selection; global settings remain unchanged. Existing ADR-039/140 apply; no new settings ownership boundary was introduced.

Mounted UI regressions, independent review and real CPU/MPS/ONNX English plus non-English playback pass. The initial harness voice mismatch was separately traced to missing production axis-default seeding and retained as a harness failure, not a product voice regression. The final installed wheel also passed playback and Stop/recovery; source/voice identity and unchanged user config are recorded in Docs/QA/tts-macos-burndown-2026-09-09/{kokoro-english,kokoro-languages,integration}/README.md.

PR #2545 review replaced the stale Settings Kokoro voice list with the shared complete official catalog. A mounted regression selects representative voices across all nine supported languages without writing a blend file; configured engine inheritance and session overrides remain covered. Post-review evidence: Docs/QA/tts-macos-burndown-2026-09-09/review/README.md. The post-rebase targeted selection passed 405 tests with one explicit MPS skip; the final migration/resolver/provider selection passed 81 overlapping tests. The clean installed wheel matches 2,275 Python files (2,266 application plus nine profile-core files). Five serialized tuples passed generation/playback and joined cleanup; English CPU/MPS/ONNX produced nine exact full transcripts and three real Stop/recovery controls. Japanese/Mandarin raw content differences remain review-required. All 23 recorded owned PIDs exited and user config stayed unchanged. No new maintained-source Ruff diagnostics; existing debt and frozen QA snapshots are documented separately. Existing ADRs remain applicable.
<!-- SECTION:NOTES:END -->
