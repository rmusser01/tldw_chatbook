---
id: TASK-20937
title: Add deterministic GGUF variant guidance to Remote Models
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22 19:43'
updated_date: '2026-08-22 20:23'
labels:
  - models
  - ui
  - ux
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users compare eligible GGUF variants using exact filename, byte-size, shard-count, and conservatively recognized quantization facts without implying runtime compatibility or machine fit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each candidate keeps its exact filename primary and shows human-readable size, shard count, and a conservatively recognized quantization token with bounded plain-language guidance; unknown filename patterns are labeled not identified and are never guessed.
- [x] #2 Users can filter candidates locally by filename or recognized quantization and sort by source order, size ascending, size descending, or quantization without issuing another network request.
- [x] #3 Changing the filter clears a selection that is no longer visible and requires an explicit visible reselection; changing sort order preserves the exact selected candidate.
- [x] #4 The screen explicitly labels guidance as filename-derived and general and continues to say runtime compatibility and machine fit are not verified.
- [x] #5 Existing Hugging Face attribution, provenance, consent, verification, installation, completion, and runtime handoff behavior remain unchanged, and no one-option provider selector is added.
- [x] #6 Guidance controls, rows, selection status, and the install action remain painted, contained, text-labeled, and keyboard reachable at 80 columns under production CSS.
- [x] #7 Focused pure and mounted tests cover recognized and unknown quantization, local sort and filter, selection safety, focus, and narrow-width rendering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs provider-neutral artifact truth, GGUF structural admission, provenance, compatibility claims, and explicit runtime selection. This task adds a pure, filename-derived UI projection and local candidate controls without changing provider, acquisition, storage, or runtime contracts.

1. Run the focused Remote baseline and add failing pure tests for bounded quantization recognition, unknown fallback, filtering, and stable sorting.
2. Implement the minimal provider-neutral variant-guidance projection using only filename, size, and shard count.
3. Add failing mounted RemoteView tests for local filter/sort behavior, selection safety, focus, honest copy, and the existing install lifecycle; then wire the controls and candidate rows.
4. Apply production CSS, regenerate the consolidated bundle, and verify the focused Remote, host-adoption, CSS, lint, compilation, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a provider-neutral deterministic projection from exact upstream file paths, bytes, and shard counts, with bounded quantization recognition and qualified same-model guidance. Added local filename/quantization filtering, four stable sort orders, hidden-selection clearing, sort-preserved identity, and production 80-column keyboard/paint behavior. Kept Hugging Face attribution and the existing provider, acquisition, consent, verification, completion, and runtime boundaries unchanged. Updated model browser state, RemoteView, production CSS/generated bundles, and focused pure/mounted/real-host tests. ADR check: no new ADR; ADR-025 continues to govern artifact truth and compatibility claims. Independent re-review found no remaining actionable issues. No new lessons entry was needed because existing production-hierarchy and compositor-geometry guidance covered the issues encountered.
<!-- SECTION:NOTES:END -->
