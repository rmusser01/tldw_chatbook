---
id: TASK-559
title: Image-gen P2b follow-ups
status: In Progress
assignee: []
created_date: '2026-07-24 13:30'
updated_date: '2026-07-25 10:18'
labels:
  - image-generation
  - console
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred/polish items from the image-gen P2b slice (PR #850: speak 🔊, @style presets + picker, generate-from-conversation; spec `Docs/superpowers/specs/2026-07-24-image-gen-p2b-tts-style-context-design.md` §4). Post-merge live smoke 2026-07-24 verified all wiring end-to-end in the real app (style refusals, picker draft composition, draft restore, speak's graceful no-TTS toast, context-path dispatch) — these are enhancements, not defects. Distinct from [[task-497]] (P1 polish), [[task-498]] (egress adoption), and [[task-558]] (P2a polish).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Richer conversation-context extraction for `/generate-image` with no prompt: the current `extract_context_from_messages` is keyword-shallow (mood via keyword match; `mentioned_characters`/`mentioned_settings` never populated). Design and implement a better context builder (e.g. LLM-composed prompt from the last N turns), keeping the composed-prompt-visible-in-card behavior.
- [ ] #2 Console TTS playback controls: speak is fire-and-forget today; add stop (and optionally pause/save) for Console-originated speech, reusing the legacy widgets' `TTSPlaybackEvent` actions.
- [x] #3 Style picker offers template previews (base-prompt/negative snippet) in the row or a detail pane, not just name — category — id.
- [x] #4 Per-style user-defined templates (beyond the 13 built-ins) loadable from config or a templates dir.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC4+AC3 (one unit — shared files): user-defined style templates loaded from config/templates dir merged over the 13 built-ins, then picker previews (base/negative snippet) for all templates.
2. AC2: Console TTS playback controls (stop; pause/save if the legacy TTSPlaybackEvent actions support them cleanly), reusing existing playback-event plumbing.
3. AC1: richer context extraction — LLM-composed prompt from the last N turns via the session's active provider (chat_api_call), strict timeout, config kill-switch, graceful fallback to the existing keyword extractor on any failure; composed prompt stays visible in the card.
Each unit: TDD, per-unit review before the next starts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
### Unit 1 (AC3 + AC4) -- 2026-07-25

User-defined `/generate-image` style templates + picker previews, merged
over the 13 builtins.

**AC4 -- user templates.** `Media_Creation/generation_templates.py` gained
`get_all_templates()`/`load_user_templates()`: builtins overlaid with two
sources, dir wins on id collision with the config section:
1. `[image_generation.styles.<id>]` TOML config section (documented +
   commented example added to `config.py`'s shipped `CONFIG_TOML_CONTENT`).
2. `<get_user_data_dir()>/image_generation_styles/*.toml`, one template per
   file -- mirrors the `chat_dicts`/`rag_profiles` per-item-directory
   convention. Filename stem is the authoritative id (an internal `id` field
   in the file, if present, is ignored -- closes a spoofing vector).
A user id matching a builtin overrides it; new ids extend the set.
`_coerce_generation_template` validates name/category/base_prompt (required
non-empty strings) and defaults everything else (negative_prompt falls back
to the dataclass default); malformed entries (wrong shape, bad TOML, illegal
id) are skipped with a `logger.warning`, never raise.
`get_template`/`get_templates_by_category`/`get_all_categories`/
`get_templates_by_tag`/`apply_template_to_prompt` now all resolve through
this merged set, so user templates work everywhere a builtin does:
`console_generate_image.resolve_style_token` + the unknown-style refusal
listing, `ConsoleStylePickerModal`, and (bonus, same seam) the Personas
avatar-style picker and the legacy SwarmUI sidebar widget -- fixed a
latent bug there (`Widgets/Media_Creation/swarmui_widget.py`) where its
template lookup still read raw `BUILTIN_TEMPLATES` after its own dropdown
started listing merged templates.
Cached per-process like `Image_Generation.config.get_image_generation_config`
(`reload=True`/`reset_templates_cache()` to refresh).

**AC3 -- picker previews.** `ConsoleStylePickerModal` gained a detail line
(`#console-style-picker-detail`, new `_agentic_terminal.tcss` block, bundle
rebuilt) below the results list, updated on every highlight change via the
existing `_sync_highlight()` call sites (arrow keys, click, filter
re-narrow) plus the empty-results branch. `format_style_preview()` renders
truncated (90 chars/snippet) `Prompt:`/`Negative:` lines. Rendered through
a `markup=False` `Static` -- template text is untrusted, so disabling markup
interpretation entirely is safer than escaping every field (matches this
same module's existing `EMPTY_STATIC_ID` convention).

Files: `tldw_chatbook/Media_Creation/generation_templates.py`,
`Media_Creation/__init__.py`, `Chat/console_generate_image.py`,
`Widgets/Console/console_style_picker_modal.py`,
`Widgets/Media_Creation/swarmui_widget.py`, `config.py` (documented example),
`css/components/_agentic_terminal.tcss` + rebuilt `tldw_cli_modular.tcss`.
Tests: new `Tests/Media_Creation/test_generation_templates.py` (27 cases:
config-section/dir loading, override-by-id both directions, malformed-skip
x8 parametrized cases, resolver/apply_template_to_prompt/category
integration, cache reload); `Tests/Chat/test_console_style_picker.py`
extended (+9: preview content, escaping pin via `_render_markup` +
literal-text assertions, truncation, placeholder states, merged-list
integration) and its CSS-parity test updated for the new selector. Full
suite green: 202 tests across the touched files/dirs pass; broader
`Tests/Chat -k "generate_image or style_picker or console"` sweep (1070
tests) and `Tests/UI/test_personas_expression_generate.py` (38) also green.
`ruff check` clean on touched files; `python -c "import tldw_chatbook.app"`
clean; CSS bundle re-synced (`check_bundle_sync` passes).

Deferred to units 2/3 (untouched here): AC2 (Console TTS playback
controls), AC1 (richer context extraction).
<!-- SECTION:NOTES:END -->
