---
id: TASK-621
title: 'Image-gen config silently ignores flat backend keys in [image_generation]'
status: Done
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 16:52'
labels:
  - image-generation
  - config
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: writing `openrouter_image_default_model = "..."` directly under `[image_generation]` — the exact FLAT field name the config dataclass uses — is silently ignored; only the nested `[image_generation.openrouter] default_model = "..."` shape parses. No warning is logged, so a user who guesses the flat spelling (which matches the dataclass and reads naturally) gets the shipped default with zero feedback. Cost during UAT: a full restart-and-retest cycle to discover the override never applied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either the flat spellings are accepted as aliases of the nested keys, or an unknown/unmapped key found directly under `[image_generation]` logs a clear warning naming the key and the expected nested section (choose one; document the choice in the shipped config example).
- [x] #2 The nested shape keeps working unchanged; secrets/env precedence unaffected.
- [x] #3 Tests pin the chosen behavior for at least one backend field (accepted-alias or warn-on-unknown).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Decision pinned: warn-on-unknown-key (no flat aliases -- avoids dual-spelling precedence ambiguity).
2. Image_Generation/config.py: in _load_image_generation_section(), after assembling `flat`, walk `raw`'s top-level keys and, for each key NOT in _GLOBAL_KEYS, not "styles", and not a known backend subsection name (derived from _SECRETS/_NON_SECRET backend names), log ONE loguru warning naming the key. If the key matches a flat field name used as a value in _FLAT_MAP (built by reversing _NON_SECRET/_SECRETS: flat_field_name -> (backend, toml_key)), name the exact nested [image_generation.<backend>] <toml_key> replacement; otherwise a generic "unknown key" warning.
3. Ensure this fires once per _load_image_generation_section() call (i.e. once per get_image_generation_config(reload=True) load), not per field access, and never raises (wrap in try/except like the rest of the loader's never-crash posture -- though this is pure dict/string logic so a crash is unlikely, still guard defensively).
4. Update the shipped [image_generation] config example in tldw_chatbook/config.py with a comment making the nested shape prominent (e.g. a one-line note that backend fields must live under [image_generation.<backend>], not flat).
5. TDD: test that a flat backend key (e.g. openrouter_image_default_model) directly under [image_generation] logs one warning naming the flat key and the nested [image_generation.openrouter] default_model replacement (capture loguru via caplog or a sink, matching existing test patterns); test that fully nested config produces no such warning; confirm existing Tests/Image_Generation/test_config_loader.py suite stays green.
6. Run Tests/Image_Generation/, ruff check touched files, import smoke test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the pinned warn-on-unknown-key decision (no flat aliases, to avoid two spellings of one setting needing a collision-precedence rule).

- Image_Generation/config.py: added _FLAT_MAP (flat_field_name -> (backend, toml_key)), built by reversing _NON_SECRET plus _SECRETS (whose TOML key is always "api_key") -- so the warning's suggested replacement is derived, never hand-maintained. Added _BACKEND_NAMES (union of known backend subsection names) and _warn_unknown_top_level_keys(raw), called once from _load_image_generation_section() right after reading the raw TOML section (i.e. once per get_image_generation_config(reload=True) load, not per field access -- the built dataclass is cached afterward). For each top-level raw key not in _GLOBAL_KEYS, not "styles", and not a known backend subsection name: if it matches a flat field name, logs one warning naming the key and the exact "[image_generation.<backend>] <toml_key>" replacement; otherwise logs a generic unknown-key warning. Wrapped in try/except (never crashes the loader, matching the existing never-crash posture).
- tldw_chatbook/config.py: added a prominent comment atop the shipped [image_generation] example stating backend fields must live under [image_generation.<backend>], and that a flat key here is ignored with a startup warning.

Tests added (Tests/Image_Generation/test_config_loader.py), using the project's established loguru-sink capture pattern (caplog does not intercept loguru): test_flat_backend_key_under_image_generation_warns_with_nested_replacement (openrouter_image_default_model under [image_generation] logs exactly one warning naming "[image_generation.openrouter] default_model", and the flat value never reaches the dataclass field), test_unrecognized_key_under_image_generation_warns_generically, test_nested_config_produces_no_unknown_key_warnings (fully nested config incl. [image_generation.styles] -> zero warnings), test_unknown_key_warning_fires_once_per_load_not_per_field_access (repeated attribute access on the cached config and a cache-hit reload add no further warnings).

Verification: python -m pytest Tests/Image_Generation/ -q -> 98 passed, 6 skipped (existing test_config_loader.py suite green throughout). ruff check on touched files clean (2 pre-existing unrelated F841s in tldw_chatbook/config.py, confirmed present before this change). python -c "import tldw_chatbook.app" succeeds.
<!-- SECTION:NOTES:END -->
