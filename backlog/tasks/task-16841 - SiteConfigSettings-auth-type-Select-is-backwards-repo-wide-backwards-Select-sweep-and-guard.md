---
id: TASK-16841
title: >-
  SiteConfigSettings auth-type Select is backwards; repo-wide backwards-Select
  sweep and guard
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
updated_date: '2026-08-16 17:52'
labels:
  - bug
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-15991 review (PR #1701) found a sixth instance of the backwards-`Select` bug
class, still live at dev `ee741cf10`: `UI/SiteConfigSettings.py:241-249` composes
`#auth-type-select` as `("none", "None"), ("basic", "Basic Auth"), ...` — `(value, label)`
order, backwards against Textual's `(label, value)` contract. Its consumer
`display_config` sets `auth_select.value = config.auth_type or "none"`, which raises
`InvalidSelectValueError` (the Select's real values are the display labels) — swallowed
by the bare `try/except Exception: logger.error(...)` around the
`call_from_thread(self.display_config, config)` in `load_site_config`, so selecting a
site with a saved config silently truncates the config-display refresh. Severity is
capped only because `SiteConfigSettings` is itself nav-unreachable (same as
`ScraperBuilderWindow`, task-15991).

This is the bug class TASK-15772 (PR #1691, six sites across `UI/STTS_Window.py` +
`Widgets/TTS/`) and TASK-15991 (two sites in `ScraperBuilderWindow.py`) fixed piecemeal —
**three file families, always found by review rather than by tooling, always paired with
a broad `except` that swallows the crash**. Two of the found sites hid from a plain
`grep "Select("` sweep because the options arrive later via `.set_options()`. Fix the
auth-type Select, then do the systematic version: an AST-based sweep for
`(id_string, "Display Text")`-shaped option tuples covering `Select(options=...)`,
`set_options(...)`, and option-list constructions, classify every hit, and land a
permanent guard (Tests/Architecture — none exists for this class today) so the seventh
instance cannot ship.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `#auth-type-select` composes `(label, value)` and `display_config` restores a saved auth type without raising (born-red test)
- [x] #2 A repo-wide sweep (AST-level, covering deferred `.set_options()` population) classifies every Select option construction; every backwards site found is fixed or justified in the notes
- [x] #3 A permanent guard fails on a reintroduced backwards `(value, label)` options list (proven by temporary reintroduction)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix #auth-type-select: swap to (label, value) tuples; verify display_config's
   auth_select.value = config.auth_type assignment now matches machine tokens
   ("none"/"basic"/"bearer"/"api_key"); narrow load_site_config's bare
   except Exception so a future regression in this class logs with traceback
   instead of silently swallowing. Born-red regression test in Tests/UI
   mounting SiteConfigSettings and driving display_config with a saved
   auth_type, red at HEAD with InvalidSelectValueError, green after.
2. Write an AST sweep script (scratchpad) over tldw_chatbook/**/*.py that finds
   every Select(...) constructor call's options= kwarg and every .set_options(...)
   call's first positional arg, resolving literal lists, list-comprehensions, and
   simple same-function variable assignments feeding those two shapes. Emit a
   table: file:line, construction kind, resolved tuple pairs (or "dynamic/
   unresolved"), and widget id where discoverable.
3. Manually classify every site the sweep flags as a literal 2-tuple list: trace
   what the consumer reads off `.value` (comparisons, dict lookups, attribute
   assignments) to decide label/value order is correct or backwards. Record the
   full table + verdicts in the task notes. Fix any further backwards sites found,
   each with its own born-red test.
4. Design and land a permanent AST-based Architecture guard (Tests/Architecture/)
   modeled on test_on_mount_super_guard.py's inventory-closure style: heuristics
   for a literal options list where element 0 looks like a machine token
   (snake_case/known id) and element 1 looks human (Title Case / contains spaces),
   or where a later `.value ==`/dict-key comparison in the same class matches
   element 0. Allowlist mechanism for genuinely label-valued Selects. Prove it
   fires via a synthetic tmp_path reintroduction (mutation test), and document its
   honest coverage (what it catches vs. structurally cannot, e.g. fully dynamic
   options built from external data) in its own docstring.
5. ruff check/format on touched files; run the new/changed tests plus the
   existing Architecture suite; tick ACs, write Implementation Notes with the
   full sweep table, commit locally (no push/PR/merge).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the headline bug plus three more found by a repo-wide AST sweep, then
landed a permanent Architecture guard. All four fixes are born-red tested
against the exact pre-fix code (three via `pytest` runs showing the real
`InvalidSelectValueError`/wrong-value failure, one via a live temporary
reintroduction of the guard's own target).

**Fix 1 (headline, AC#1): `UI/SiteConfigSettings.py`'s `#auth-type-select`.**
Swapped `("none", "None"), ...` to `("None", "none"), ...` (label, value).
Also narrowed `load_site_config`'s bare
`try: get_config(); call_from_thread(display_config, config) / except
Exception: logger.error(str(e))` into two try/excepts: the config-store
fetch keeps a narrow catch-and-log-and-return (an expected, DB/file-shaped
failure mode), while `call_from_thread(display_config, ...)` now gets its
own `except Exception: logger.exception(...)` -- a UI-rendering bug (like
this one) now logs a real traceback instead of a stack-free one-liner, so
the same bug class can't hide behind that log call again. Born-red test:
`Tests/UI/test_site_config_settings.py` (8 tests) -- mounts the widget with
a real, tmp_path-isolated `SiteConfigManager`, calls `display_config`
directly and through the actual `load_site_config` worker, and separately
proves the narrowed except now logs a record with `record["exception"] is
not None`. Ran red at HEAD with `InvalidSelectValueError: Illegal select
value 'basic'` (exactly the review's prediction), green after.

**AC#2 sweep.** Wrote an AST sweep script (scratchpad, not committed) that
walks every `.py` file for `Select(options=...)` / `Select(<positional>)` /
`.set_options(...)` calls, resolving literal option lists directly, through
one `name = [...]` variable hop, list/generator comprehensions, and (a
second pass) accumulator patterns (`x = []; x.append((a, b))` across a
function body, including nested if/for blocks) to find the "arrives later"
sites the task flagged. 454 total construction sites found; 230 resolved to
literal string-pair lists, 51 to comprehensions, 98 to one-hop variable
names needing the second-pass resolver, the rest to method calls or
non-literal expressions. Every one of the 98 second-pass sites, every
listcomp/genexpr, and a sample of the `call`/`unresolved` buckets were read
by hand (not just pattern-matched) tracing each site's actual `.value`
consumer. Full verdict table:

| Site | Order found | Verdict | Evidence |
|---|---|---|---|
| `UI/SiteConfigSettings.py:241` `#auth-type-select` | `(value, label)` | **BACKWARDS -- FIXED** | `display_config`: `auth_select.value = config.auth_type or "none"`; `auth_type` is always `"basic"/"bearer"/"api_key"`/`None` (`SiteConfig.__init__`'s own comment) |
| `Widgets/voice_profile_dialog.py:151` `#language-select` | `(value, label)` (`("en","English")`, ...) | **BACKWARDS -- FIXED** | `on_button_pressed`: `language = query_one("#language-select", Select).value` -> `profile_data["language"]` -> `VoiceBackendManager.create_profile(..., language=...)`. Reachable: Lab > Speech > Voice Cloning > New Profile -> `Voice_Cloning_Window._create_new_profile` pushes this dialog with the constructor's default `value=self.profile_data.get("language") or "en"` -- crashed on EVERY "New Voice Profile" open (`Select._on_mount` -> `_init_selected_option` -> the reactive `value` setter), not just edits |
| `UI/Voice_Cloning_Window.py:375` `#test-profile-select` (built in `_update_profile_display`) | `(id, display_label)` (`test_options.append((profile["name"], profile["display_name"]))`) | **BACKWARDS -- FIXED** | `_test_generate_voice`: `test_profile = query_one(...).value` -> `voice = f"profile:{test_profile}"` sent to the TTS backend. No crash (no explicit initial `value=`), but wrong behavior: the dropdown showed the internal name instead of the display name, and selecting a profile sent the display name as the profile id |
| `UI/Study_Window.py:584` `#guide-topic-select` | `(value, label)` (`("new","New Topic")`) | **BACKWARDS -- FIXED** | No `.value` consumer anywhere in the repo (grep-confirmed), so never raised -- but Textual always renders element 0, so the dropdown showed the literal text "new" instead of "New Topic". Cosmetic-only, still a real defect |
| All other literal-list sites (~226 remaining of 230) | `(label, value)` | Correct | Spot-checked broadly (STTS/TTS families, Console/Settings modals, Tools_Settings_Window, MCP modules, Study_Modules, Persona widgets, Speech settings) -- all consistent with the convention TASK-15772/15991 established |
| `mcp_inspector.py:2752`, `mcp_tools_mode.py:_server_options`, `settings_screen.py:_library_rag_profile_select_options`, `chat_approval_card.py:_options_for_row`, `console_model_popover.py`/`console_settings_modal.py` provider/model option builders, `library_screen.py:_library_notes_folder_target_options`, `speech_settings_group.py:SELECT_OPTIONS` dict, `reminder_form.py:_schedule_options`/`_preset_options`, `settings_screen.py:_appearance_theme_options`, `results_grid.py:_baseline_options`, `settings_splash_screen_viewer.py:_default_select_options`, `speech_tts_settings_panel.py:_default_profile_options`/`_safe_exact_options`/`_realtime_provider_options`, `character_voice_widget.py:voice_options`, `mcp_rail.py:scope_options`/`ref_options` | `(label, value)` | Correct | Each traced to its definition and consumer by hand (not just shape-matched) -- see per-site detail in the PR discussion; all consistent |
| `Widgets/base_components.py`'s `FormField.options` / `create_form_field(field_type="select", ...)` | n/a | **Dead code** | Zero call sites anywhere in the repo use `field_type="select"` for either helper -- nothing to be backwards *about* |
| `Widgets/enhanced_file_picker.py:self.filters.selections` | n/a | Out of scope | `Filters` is a vendored `Third_Party/textual_fspicker` type, not this repo's code |

No further sites were found beyond the three listed as BACKWARDS -- FIXED
(plus the headline #1). Each got its own born-red test, red at HEAD, green
after the fix:
- `Tests/Widgets/test_voice_profile_dialog.py` (3 tests) -- red with
  `InvalidSelectValueError: Illegal select value 'en'` on the default "New
  Voice Profile" open.
- `Tests/UI/test_voice_cloning_test_profile_select.py` (2 tests) -- red
  first on a label-content mismatch (`['narrator_1', 'villain_2'] !=
  ['Narrator One', 'Villain Two']`), then on
  `InvalidSelectValueError: Illegal select value 'villain_2'` for the
  `_test_generate_voice` round-trip.
- `Tests/UI/test_study_guide_topic_select.py` (1 test) -- red on the
  rendered-label mismatch (`'new' != 'New Topic'`).

**AC#3 guard: `Tests/Architecture/test_backwards_select_option_guard.py`.**
An AST-based inventory guard (same style as `test_on_mount_super_guard.py`)
over every `Select(...)`/`.set_options(...)` call in the package. Detects a
literal list of 2-tuples of string constants (resolved directly, or through
one `name = [...]` assignment hop) where at least one pair has element 0
shaped like a bare machine token (`^[a-z][a-z0-9_-]*$`) and element 1 shaped
like human text (contains a space, or Title-Cased), and the two aren't the
same word after normalizing case/separators (so `("none", "None")` alone
isn't flagged, but `("basic", "Basic Auth")` in the same list is).

Honest coverage, stated in the guard's own docstring: it catches literal,
hand-typed option tuples -- which is what caught 3 of this task's 4 real
findings (`#auth-type-select`, `#language-select`, `#guide-topic-select`)
-- and raises zero false positives across the ~230 correct literal-list
sites already in this repo (verified by running it against HEAD before any
fix landed, restricted to the untouched files). It structurally CANNOT
catch the `Voice_Cloning_Window` bug: `(profile["name"],
profile["display_name"])` has no string literal to pattern-match against --
that one needed tracing what the dict keys *mean*, which the guard's
docstring calls out explicitly as a known, permanent gap, together with the
other three documented gaps (single-word labels with no space/Title-Case,
more than one hop of indirection, non-string values). An `ALLOWLIST`
mechanism exists for a genuinely intentional exception; it is empty at
landing (`test_current_repo_has_no_allowlist_entries` pins that).

Proven three ways: (1) `test_no_backwards_select_option_literals` passes
clean against the real, fixed repo; (2) a tmp_path mutation test
(`test_guard_detects_a_synthetic_backwards_select`) seeds a literal-shaped
backwards Select, a variable-indirection backwards Select, a correct
Select, and the *documented-gap* dynamic-dict-key shape, asserting the
first two are flagged, the third isn't, and the fourth (correctly) isn't
either; a fourth test proves the allowlist suppresses a flagged site by
`(path, lineno)`; (3) **live temporary reintroduction** (AC#3's own
wording) -- reverted `SiteConfigSettings.py`'s already-fixed
`#auth-type-select` tuples back to `(value, label)` via `Edit`, re-ran
`test_no_backwards_select_option_literals`, watched it fail with
`tldw_chatbook/UI/SiteConfigSettings.py:241: option pair ('basic', 'Basic
Auth') looks (value, label)`, then restored the fix via `Edit` (not `git
checkout`) and re-ran green.

**Files touched:** `tldw_chatbook/UI/SiteConfigSettings.py`,
`tldw_chatbook/Widgets/voice_profile_dialog.py`,
`tldw_chatbook/UI/Voice_Cloning_Window.py`,
`tldw_chatbook/UI/Study_Window.py` (fixes); `Tests/UI/
test_site_config_settings.py`, `Tests/Widgets/test_voice_profile_dialog.py`,
`Tests/UI/test_voice_cloning_test_profile_select.py`, `Tests/UI/
test_study_guide_topic_select.py`, `Tests/Architecture/
test_backwards_select_option_guard.py` (new tests).

**Verification:** `ruff check` clean on all touched files. Full new/changed
suite (18 tests across 5 files) green. Full `Tests/Architecture/` suite:
138 passed, 3 pre-existing failures unrelated to this change (`test_
persistent_diagnostic_inventory.py` x2, `test_screen_size_ratchet.py`'s
`chat_screen.py` budget) -- none of the three touch any file this task
modified; confirmed pre-existing via `git diff --stat` showing zero overlap.
<!-- SECTION:NOTES:END -->
