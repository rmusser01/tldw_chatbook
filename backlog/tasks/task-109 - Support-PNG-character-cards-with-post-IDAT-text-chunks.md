---
id: TASK-109
title: Support PNG character cards with post-IDAT text chunks (SillyTavern)
status: In Progress
labels:
- character-chat
- import
- bugfix
- regression
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-100 shipped, the reporter still could not import cards from their SillyTavern characters folder (`Ann1.png` failing with "no character JSON metadata found"). Root cause: SillyTavern (and some other exporters) write the `chara`/`ccv3` tEXt chunks AFTER the IDAT image-data chunk, and Pillow only surfaces trailing chunks in `Image.info` once the image data has been decoded. `extract_json_from_image_file` only inspected `.info` before any decode, so it never saw the metadata and rejected the card. This also explains the single "failure" noted in TASK-100 (`Awkward Questions FM.png`), which likewise carries a trailing `chara` chunk.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PNG cards whose `chara`/`ccv3` tEXt chunks appear after the IDAT chunk are detected and imported.
- [x] #2 Cards with pre-IDAT metadata, EXIF UserComment cards, and metadata-less plain images keep their existing behavior (no regressions; plain images are still rejected cleanly).
- [x] #3 All cards in the reporter's collection import end-to-end through the app's full import path (verified: 108/108 in E:\LLM-Models\Charcards).
- [x] #4 All cards in a real SillyTavern characters folder import (verified: 95/95, including the reported `Ann1.png`).
- [x] #5 Regression tests build a real PNG with a post-IDAT tEXt chunk and cover both metadata extraction and full card load.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Bug fix within the existing image-metadata extraction path; no storage/schema, sync policy, provider boundary, or architectural decision changes.

1. Reproduce with the reported `Ann1.png`; inspect PNG chunk layout to confirm post-IDAT placement of the `chara` chunk.
2. In `extract_json_from_image_file`, force a full decode (`img.load()`) when no metadata key is found in `.info`, then re-check for `chara`/`ccv3` before the EXIF UserComment fallback.
3. Add regression tests that construct a PNG with a trailing tEXt chunk via chunk surgery (insert before IEND) and assert extraction plus end-to-end card load.
4. Re-verify the reporter's 108-card folder and a real SillyTavern characters folder through `import_and_save_character_from_file`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
`extract_json_from_image_file` in `Character_Chat_Lib.py` now, when the initial `.info` scan finds no `chara`/`ccv3` key, calls `img_obj.load()` (exceptions logged at debug and ignored) and re-checks `.info`, logging when metadata is found in a trailing (post-IDAT) chunk. All pre-existing paths (pre-IDAT chunks, EXIF UserComment for WebP/JPEG, clean rejection of plain images) are unchanged. The full decode happens only on the previously-failing path, so there is no performance change for cards that already worked.

Modified/added files:
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (post-IDAT metadata recovery in `extract_json_from_image_file`)
- `Tests/Character_Chat/test_character_card_lenient_import.py` (PNG chunk-surgery helper + 2 regression tests)

Verification:
- `pytest Tests/Character_Chat/test_character_card_lenient_import.py` -> 29 passed.
- Full import path over reporter's folder -> IMPORTED: 108, FAILED: 0.
- Full import path over SillyTavern characters folder -> IMPORTED: 95, FAILED: 0 (includes reported `Ann1.png`).

Review hardening (Qodo): the metadata keys are now a shared module constant (`_CARD_IMAGE_METADATA_KEYS`), the forced decode is PNG-only (WebP/JPEG skip straight to EXIF) and bounded by `_MAX_CARD_DECODE_PIXELS` (50 MP, checked pre-decode via `img_obj.size`) to prevent CPU/memory spikes on oversized untrusted images, and PNG decode failures log at warning. A regression test covers the oversized-skip path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PNG character cards written by SillyTavern-style exporters (metadata chunk after IDAT) import correctly; the extractor forces a full image decode before concluding a card has no embedded metadata.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
