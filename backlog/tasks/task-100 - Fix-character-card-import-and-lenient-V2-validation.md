---
id: TASK-100
title: Fix character card import failures and loosen over-tight V2 validation
status: Done
labels:
- character-chat
- import
- bugfix
- regression
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Users reported being unable to import character cards (V2 JSON and PNG cards exported from Chub/SillyTavern). Two root causes: `load_character_card_from_file` in `Character_Chat_Lib.py` was a stub that always returned `None` (breaking the ingest preview and name resolution paths), and the V2 import pipeline hard-rejected cards that omitted spec-"required" fields, used un-namespaced extension keys, used numeric lorebook positions, or declared `chara_card_v3` — all common in real-world cards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `load_character_card_from_file` is a real implementation that parses .json/.yaml/.yml/.md/.txt text cards and .png/.webp image cards with embedded metadata.
- [x] #2 V2 cards missing description/personality/scenario/first_mes/mes_example still import; only a missing/blank `name` (or non-dict `data` node) is fatal.
- [x] #3 Explicit-V2 cards that fail structural validation no longer hard-abort; import falls back V2 → V1 → generic multi-format detection.
- [x] #4 Character book entries are preserved with defaults (numeric positions normalized, missing enabled/insertion_order defaulted) instead of failing validation or being dropped.
- [x] #5 V3 cards (`spec: chara_card_v3`, `ccv3` PNG metadata key) are recognized and imported.
- [x] #6 All user-supplied cards with embedded card data parse successfully (verified: 107/108 files in the reporter's card folder; the single failure is a PNG with no embedded metadata at all).
- [x] #7 Regression tests cover the lenient import paths and the previously-broken stub.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Bug fix plus leniency changes within the existing character card import pipeline; no new storage/schema, sync policy, provider boundary, service contract, or long-lived architectural decision is introduced.

1. Reproduce with reporter's card collection (E:\LLM-Models\Charcards) via static analysis and a verification script.
2. Replace the `load_character_card_from_file` stub with a real parse-only loader that reads bytes directly (avoids base-directory path rejection for files outside the app data dir).
3. Restructure `validate_v2_card` into fatal errors vs non-fatal warnings; loosen `parse_v2_card`/`parse_v1_card` with coercion helpers.
4. Remove the hard abort in `import_character_card_from_json_string`; add V1 and generic multi-format fallbacks.
5. Make character book validation warning-based and entry parsing default-applying; extract `ccv3` PNG metadata.
6. Add regression tests; run related suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed character card import on two fronts. First, `load_character_card_from_file` (used by the ingest preview and import name resolution) was a stub returning `None` at the bottom of `Character_Chat_Lib.py`; it is now a real parse-only loader for text and image card files. Second, the V2 pipeline was stricter than the ecosystem: `validate_v2_card` now treats only a missing/blank `name` and a missing/non-dict `data` node as fatal, reporting everything else (missing spec fields, wrong types, un-namespaced extension keys, character book quirks, unexpected spec/spec_version) as warnings. `parse_v2_card`/`parse_v1_card` coerce missing fields to `""` and lists/numbers to strings via new `_coerce_card_text`/`_coerce_card_str_list` helpers. `import_character_card_from_json_string` no longer aborts on validation failure — it parses leniently, then falls back to V1 parsing and finally to the generic `character_card_formats` detector. Character books: `validate_character_book`/`validate_character_book_entry` are warning-based, and `parse_character_book` keeps entries with defaults (enabled=True, insertion_order=list position, numeric position 0/1 → before_char/after_char) instead of dropping them. `extract_json_from_image_file` also reads the `ccv3` metadata key (V3 cards) and bytes-valued chunks.

Modified/added files:
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py` (stub replacement, lenient validation/parsing, ccv3 extraction)
- `Tests/Character_Chat/test_character_card_lenient_import.py` (new, 15 regression tests)

Verification:
- `verify_charcard_imports.py` over the reporter's 108 files -> PARSED: 107, FAILED: 1 (the failure is `Awkward Questions FM.png`, a plain image with zero embedded metadata — correct rejection).
- `pytest Tests/Character_Chat/test_character_card_lenient_import.py` -> 15 passed.
- Related existing suites (character chat, file operations, dictionaries portability, markdown/image import, world info, ingest events, world book import, expression sets, integration character book) -> 166 passed, 0 failed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Character card import works for real-world V1/V2/V3 cards: the broken `load_character_card_from_file` stub is implemented, and V2 validation/parsing is lenient where the ecosystem is loose, so Chub/SillyTavern exports import cleanly instead of being rejected.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
