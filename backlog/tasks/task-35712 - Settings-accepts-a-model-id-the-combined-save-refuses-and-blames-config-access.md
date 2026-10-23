---
id: TASK-35712
title: Settings accepts a model id the combined save refuses and blames config access
status: Done
assignee:
- '@claude'
created_date: 2026-10-20 09:30
updated_date: 2026-10-22 23:41
labels:
- follow-up
- settings
- providers
- model-config
priority: low
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/UI/Settings_Modules/providers_models_card.py
- tldw_chatbook/UI/Screens/settings_screen.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35011 (2026-10-20). Settings ▸ Providers & Models takes a Default model id of any length: the `settings-model-value` Input sets no `max_length`, and a Custom ID is staged as typed. The combined provider save refuses any model id over 120 characters (`_MAX_MODEL_CHARS` in provider_setup_persistence.py; `persist_provider_settings_atomic` raises `ValueError("Provider settings model is invalid.")` before it validates anything else). Settings catches that ValueError as a failed write, so the user sees "Provider settings not fully applied: the file was not written. Draft retained; check config access and retry". Retrying fails the same way, and the message sends the user looking at file permissions instead of at the id. Discovered ids are already capped at 120 by the discovery cache, so this is reached through a typed Custom ID (or a hand-edited draft), not through Discover models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A model id the save cannot write is refused where it is entered, with a message that names the limit, before s is pressed
- [x] #2 If such an id still reaches the save, the message names the model id as the cause, not config access
- [x] #3 Settings and the combined save agree on one limit, defined in one place
<!-- AC:END -->

## Implementation Plan

Finding: the save refuses more than length. `_safe_model` rejects padding, an empty id, an id over 120 characters, and anything not `isprintable()` or in Cc/Cf/Cs. Settings' Custom ID check (`normalize_model_id`, 256 characters) accepts several of those characters: U+200B, U+00A0, U+202E, U+007F, U+2028 and U+E000 were each probed through the wrapper. So the entry check reuses the save's own predicate: one predicate, two callers.

1. provider_setup_persistence.py: make the limit public as `MAX_MODEL_ID_CHARS` (it was `_MAX_MODEL_CHARS`). Add `is_writable_model_id(model)`, which is `_safe_model(model) == model`. `persist_provider_settings_atomic` calls it.
2. Widgets/model_search_picker.py (its only host is Settings ▸ Providers & Models): a typed Custom ID must pass `normalize_model_id` and then `is_writable_model_id`, and the invalid-ID status line names `MAX_MODEL_ID_CHARS`. Catalog and saved ids keep the picker's existing check, so a hand-edited saved id still shows.
3. settings_screen.py: when the combined save's ValueError comes from a model id the predicate refuses, the result line names the model ID and the limit instead of config access. Pay for the added lines by deleting the dead `_set_provider_context_window_input_value`, and lower the module-size row to the new count.
4. saved_model_list.py: the join's "over 120 characters" copy reads the same constant.
5. Tests in Tests/UI/test_settings_default_model_picker.py (a whole-file UI-lane census entry):
   - mounted, Custom ID refused at entry with the limit named (AC#1);
   - mounted, an over-limit id that reaches s is named as the cause (AC#2);
   - unit, the Custom ID check and `persist_provider_settings_atomic` refuse exactly the same ids (AC#3).
   Then the guide (256 to 120, plus the new save message).

## Implementation Notes

Settings and the combined provider save now share one model-id limit and one check. Settings refuses an unwritable id as it is typed; if one still reaches `s`, the result line names the model ID and the limit.

**Finding.** The save refuses more than length. `_safe_model` refuses:
- an empty or padded id;
- an id over 120 characters;
- an id that is not `isprintable()`;
- an id holding a Cc, Cf or Cs character.

The Custom ID check (`normalize_model_id`, 256 characters) admitted U+200B, U+00A0, U+202E, U+007F, U+2028 and U+E000; each was probed through the wrapper. So the entry check reuses the save's predicate rather than copying its length.

**Changes:**
- **One limit, one predicate** (`Chat/provider_setup_persistence.py`): `_MAX_MODEL_CHARS` is now the public `MAX_MODEL_ID_CHARS`. New `is_writable_model_id(model)` returns `_safe_model(model) == model`, the save's own test. `persist_provider_settings_atomic` now calls it instead of repeating the expression.
- **Refused where it is entered (AC#1)** (`Widgets/model_search_picker.py`, hosted only by Settings ▸ Providers & Models): new `normalize_custom_model_id` applies `normalize_model_id` and then `is_writable_model_id`. Every read of typed Custom ID text uses it:
  - the `value` property;
  - typing;
  - Enter;
  - leaving Custom ID mode;
  - `set_custom_value`.

  The status line now reads "Invalid model ID. Use a single-line value of at most 120 characters." The existing roll-back on Esc or blur still puts the previous model back. Listed and saved ids keep the plain check, so a hand-edited saved id still shows in the field. `normalize_model_id`, which Console's Switch model shares, is unchanged.
- **Named at the save (AC#2)** (`UI/Screens/settings_screen.py`): before building the mutation, the providers save checks the form's model with `is_writable_model_id`. A refused id now reads "Provider settings not saved: the model ID is invalid. Use a single-line value of at most 120 characters. Draft retained." Before, it read one of two other messages, depending on the route:
  - a staged id hit `build_provider_setup_mutation` and read "review the endpoint and model";
  - an id already in config.toml, saved alongside another field, hit `persist_provider_settings_atomic` and read "the file was not written … check config access and retry".

  The check comes after the "No Settings changes" return, so a clean form still says that.
- **Deviation from plan step 3**: the plan classified the save's ValueError after the write was attempted. The red AC#2 run showed that a staged id never reaches that write: `build_provider_setup_mutation` refuses it first, with "review the endpoint and model". So the check runs ahead of both refusals.
- **The join's copy** (`Settings_Modules/saved_model_list.py`): the "over 120 characters" receipt reads `MAX_MODEL_ID_CHARS`. Its local import became a module alias, which keeps the file at its 518-line row.
- **Size ratchets**: settings_screen.py's +14 lines are paid for by deleting two methods nothing in `tldw_chatbook/` or `Tests/` calls: `_set_provider_context_window_input_value` and `_tool_result_display_chars_label`. That is −19 lines, and the row is lowered 35,668 → 35,663. providers_models_card.py is untouched. saved_model_list.py stays at 518.
- **Tests** (Tests/UI/test_settings_default_model_picker.py, a whole-file UI-lane census entry):
  - `test_custom_id_and_the_combined_save_refuse_the_same_ids` (AC#3, 8 ids): an id the Custom ID check holds is one `persist_provider_settings_atomic` writes; an id it refuses is one the save refuses.
  - `test_a_custom_id_the_save_cannot_write_is_refused_as_it_is_typed` (AC#1): 120 characters is staged; 121 characters and U+200B are refused, with the limit named.
  - `test_an_id_the_save_refuses_is_named_as_the_cause_not_config_access[staged-model|saved-model]` (AC#2).
- **Red** (old picker check via a temporary `normalize_custom_model_id = normalize_model_id` alias, and the old save path; receipts in the session scratchpad as a-35712-red*.log):
  - the AC#3 test failed `assert not held` for all 6 unwritable ids;
  - AC#1 failed `assert 'mmm…m' is None` (the 121-character id was held);
  - AC#2 saved-model failed on "…check config access and retry.";
  - AC#2 staged-model failed on "…review the endpoint and model.".

  Green: the 11 new cases pass.
- **Covering runs** (scratch wrapper, `-n 4`):
  - All 36 files that reference the touched symbols, plus the two Custom ID files and the module ratchet: 1569 passed and 612 unique node IDs failed. The same 612 IDs also fail with the Task 2 change reverted, and with the sources set back to dev 28beeb9ed5.
  - Of those 612: 570 fail with the local `RecoveryRequired: raw_source_selection_changed` signature (plain Tests/UI tests); 8 are size-ratchet rows for untouched files; the rest are existing Console failures.
  - After the patch was reapplied, the six core files (default-model picker, Widgets picker, provider_setup_persistence, pick-joins, the two Custom ID files): 423 passed.
  - Import-budget tests (preimport payload, app import weight, UI-ready census): 29 passed.
  - Fifteen Tests/Performance compact-setup and three-turn cases fail the same way at dev.
  - Module size ratchet for settings_screen.py, providers_models_card.py and saved_model_list.py: passed.
  - `check_markup_interpolation`: OK.
  - `./scripts/preflight.sh`: passed.
- **Docs**: Docs/User_Guide/settings.md now gives the Custom ID limit as 120, not 256. It quotes both the as-you-type refusal and the save's message.
- **Files**:
  - tldw_chatbook/Chat/provider_setup_persistence.py
  - tldw_chatbook/Widgets/model_search_picker.py
  - tldw_chatbook/UI/Screens/settings_screen.py
  - tldw_chatbook/UI/Settings_Modules/saved_model_list.py
  - Tests/UI/test_settings_default_model_picker.py
  - Tests/Architecture/test_module_size_ratchet.py
  - Docs/User_Guide/settings.md

**Review round 1 (verdict "With fixes")**

- **I3 (wrong reason).** An id of 4 characters holding U+200B or a no-break space was told "…at most 120 characters". The picker's status line and the save message now add the guide's own clause, "with no invisible or control characters". The test constant `_INVALID_ID_STATUS`, the AC#2 expectation and the User Guide quotes were updated to match.
  - Red under the old copy (temporary revert): 3 failed, namely AC#1 for both unwritable ids and both AC#2 routes.
  - Green after: 11 passed.
  - `check_markup_interpolation`: OK.
- **AC#3, the wizard copy (unticked until fixed).** `UI/Wizards/first_run_setup_state.py` kept `_MAX_MODEL_CHARS = 120`, and `validate_first_run_model_id` hand-copied `_safe_model` for the same combined save. The validator now calls `is_writable_model_id` (lazy import, as the module does elsewhere) and the private constant is gone. Its "Model is invalid." message is unchanged.
  - The AC#3 test now also asserts the wizard holds exactly the ids the Custom ID check holds. Red with the dev wizard code and its constant drifted to 121 (temporary edit): `AssertionError: mmm…`. Green after.
  - first_run_setup_state.py has no size-ratchet row.
  - Left alone on purpose: the discovery caches' and `local_server_discovery`'s 120s. They truncate listed ids, which is a different concern.
- **M1.** The test source used literal U+200B, U+00A0, U+202E and U+2028. It now uses `\u200b`, `\u00a0`, `\u202e` and `\u2028` escapes; a scan of every changed file finds no Cf, Zl, Zp, Co or non-space Zs characters.
- **M2.** One blank line removed before `normalize_custom_model_id`, leaving two.
- **Runs (`-n 4`):**
  - tools_hub, default_model_picker, both Widgets picker files, provider_setup_persistence, pick_joins, custom_id rollback and fast typing, saved_lists_follow_file, plus the four import-budget files: 504 passed.
  - Wizard tests (test_first_run_setup_state, test_first_run_setup_wizard, test_first_run_setup_integration, test_first_run_wizard_live_contract): 739 passed, 49 failed and 31 errors. The same 80 node IDs fail identically at HEAD without these fixes and at dev 28beeb9ed5 (76 are `RecoveryRequired: raw_source_selection_changed`). `test_real_discovery_result_rejects_malformed_or_unsafe_models`, which reaches `validate_first_run_model_id`, passes.
  - Module and screen size ratchets for the touched rows: passed.
  - Preflight: passed.
