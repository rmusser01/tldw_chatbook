---
id: TASK-559
title: Image-gen P2b follow-ups
status: Done
assignee: []
created_date: '2026-07-24 13:30'
updated_date: '2026-07-25 11:59'
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
- [x] #1 Richer conversation-context extraction for `/generate-image` with no prompt: the current `extract_context_from_messages` is keyword-shallow (mood via keyword match; `mentioned_characters`/`mentioned_settings` never populated). Design and implement a better context builder (e.g. LLM-composed prompt from the last N turns), keeping the composed-prompt-visible-in-card behavior.
- [x] #2 Console TTS playback controls: speak is fire-and-forget today; add stop (and optionally pause/save) for Console-originated speech, reusing the legacy widgets' `TTSPlaybackEvent` actions.
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

### Unit 2 (AC2) -- 2026-07-25

Console TTS stop control, reusing `TTSPlaybackEvent` exactly as legacy chat
does -- no new audio machinery, no new slash command.

**UI surface.** Per-message action-row toggle: while a message is the one
driving Console TTS, its 🔊 speak action swaps to a ⏹ speak-stop action in
the same slot, mirroring the generation card's browsed-index-driven "Keep"
swap (`ConsoleMessageActionService.available_actions(speaking_message_id=)`,
threaded through `ConsoleTranscript._action_row`/`_action_row_signature`
via a `_console_tts_speaking_message_id()` reader that mirrors
`_generation_browsed_index()`'s `getattr(self.screen, ...)` pattern).
Screen-side ephemeral state (`ChatScreen._console_speaking_message_id`,
never persisted) is set when speak dispatches and cleared when speak-stop
dispatches for the same id; the existing 0.2s Console sync tick (plus an
explicit `_sync_native_console_chat_ui()` call in both branches, matching
the keep/variant-nav/delete precedent) picks up the swap immediately. A
`/speak stop` slash command was considered (explicitly allowed as the
"minimal core" per the brief) but skipped: no `/speak` command exists today
(speak is button-only), so it would have been new grammar/dispatch surface,
not a re-use of anything -- the message-row toggle is both cheaper and
better UX (no command to learn), at the cost of one known limitation (see
below). Button-id parsing trap closed: `"console-message-action-speak-"` is
a literal prefix of the new `"console-message-action-speak-stop-"`, so the
more specific entry had to be added *before* the existing one in
`_parse_console_message_action_button_id`'s ordered prefix table, else a
speak-stop click would mis-route as speak with a mangled message id.

**Actions wired.** Only `stop` (`TTSPlaybackEvent(action="stop", ...)`).
Wiring it exposed a real pre-existing bug shared with legacy chat:
`TTSEventHandler.handle_tts_playback`'s "stop" branch only ever deleted the
cached audio file (`_cleanup_audio_file`) -- it never told the actual
system audio player (`TTS/audio_player.SimpleAudioPlayer`, a subprocess
wrapper) to stop, so a stop click did not silence audio already playing
(afplay/mpv etc. keep streaming a deleted-but-open file on Unix). Fixed
minimally, reusing only pre-existing player methods: added
`SimpleAudioPlayer.get_current_file()` (mirrors the existing
`get_state()`/`get_position()` accessors) and a small
`stop_audio_playback_if_current(file_path)` helper in `tts_events.py` that
calls the player's real `stop()` *only* when the message being stopped is
the one currently loaded. The "only if current" guard matters: the player
is a single global "now playing" slot, and legacy chat can have several
messages simultaneously cached-in-"ready"-state (never played) while a
different message is actively playing -- stopping a never-played message
must not silence an unrelated one that is.

**Skipped: pause, save.** Pause: the handler's existing "pause" branch is a
no-op stub (a log line and a comment, no call into the player at all); real
pause+resume would need new action semantics (the current player only
supports fresh `play()`, not "resume from where a stopped clip left off"
without new state) -- that is new plumbing, out of scope per the brief.
Save: `TTSExportEvent`/`handle_tts_export` is fully generic and would work
mechanically, but Console's "play" branch always schedules
`_cleanup_audio_file(..., delay=5.0)` right after playback starts (fire-and
-auto-play, unlike legacy's manual-play "ready" state) -- a save action
would race that fixed 5s auto-delete regardless of actual clip length,
making it unreliable to wire "cleanly" without also changing that timing,
which is itself new plumbing.

**Natural end of playback -- no event exists.** `SimpleAudioPlayer.
_monitor_playback()` runs on a background thread and only mutates its own
internal `_current.state` when the subprocess exits; nothing bridges that
back into Textual's event system (would need a thread-safe callback into
`asyncio.run_coroutine_threadsafe`+a new event -- new plumbing, skipped).
Consequence, stated honestly rather than faked with a timer: the ⏹ persists
until the user clicks it, or until a NEW speak request (on any message)
overwrites the tracked id -- which also organically self-heals the display
over time, since the underlying player is single-slot and a fresh `play()`
always stops whatever was previously loaded. This exact limitation already
exists in legacy chat's own `tts_state == "playing"` widget state (also
never auto-reverts on natural completion), so it is not a new gap.

Files: `tldw_chatbook/Chat/console_message_actions.py` (`speaking_message_id`
kwarg, `speak`/`speak-stop` swap, `speak-stop` dispatch branch),
`tldw_chatbook/Widgets/Console/console_transcript.py`
(`_console_tts_speaking_message_id`, threaded into both action-row builders,
tooltip entry), `tldw_chatbook/UI/Screens/chat_screen.py`
(`_console_speaking_message_id` state, speak/speak-stop dispatch branches,
button-id prefix-table ordering fix), `tldw_chatbook/Event_Handlers/
TTS_Events/tts_events.py` (`stop_audio_playback_if_current`, wired into the
"stop" branch), `tldw_chatbook/TTS/audio_player.py` (`get_current_file()`).
Tests: `Tests/Chat/test_console_message_actions.py` (+5: swap/no-swap/
failed-message-never-swaps/dispatch), `Tests/Chat/
test_console_generation_actions.py` (+4: screen-level speak marks state +
syncs, speak-stop posts the event + clears + syncs, safe-when-nothing-
tracked, does-not-clear-a-different-message's state), `Tests/UI/
test_console_native_transcript.py` (+3, new `SpeakActionRowHarness`: mounted
row shows 🔊 by default, swaps to ⏹ when tracked, unaffected by a different
message's id), `Tests/TTS/test_tts_improvements.py` (+4: stop actually
calls the player when current, does NOT call it for an unrelated playing
message, safe no-op when nothing cached, `get_current_file()` state
tracking). All TDD red-then-green. Full suite green: `Tests/TTS/` (747
passed, 14 skipped, pre-existing/unrelated), `Tests/Chat/test_console_
message_actions.py` + `test_console_generation_actions.py` +
`Tests/UI/test_console_native_transcript.py` (134 passed), broader
`Tests/Chat -k "generate_image or style_picker or console"` regression
sweep (1079 passed) and `Tests/UI/test_console_native_chat_flow.py`
(foundational Console flow suite) both green. `ruff check` clean on all
touched files; `python -c "import tldw_chatbook.app"` clean. No CSS bundle
change needed (the ⏹ button reuses the existing generic
`console-transcript-action-button` styling).

Deferred to unit 3 (untouched here): AC1 (richer context extraction).

### Unit 3 (AC1) -- 2026-07-25

LLM-composed `/generate-image` conversation-context prompt, with graceful
fallback to the existing keyword extractor on any failure -- the load-
bearing requirement per the brief.

**Provider resolution reused, not reinvented.** `ChatScreen._build_console_
provider_selection()` + `ConsoleProviderGateway.resolve_for_send()` (both
pre-existing, the exact seam a normal Console chat send already resolves
provider/model through) are wrapped by a new `ChatScreen._console_
generate_image_llm_context_options(cfg)`, called on the UI loop only when
the invocation has no prompt -- this matches Console-send behavior
exactly, INCLUDING llama.cpp's bounded `/health` reachability probe
(`ConsoleProviderGateway._is_reachable`, capped at
`PROBE_TIMEOUT_SECONDS`); every other provider's resolution is
config/env-only. Any resolution failure (or the `context_llm_enabled`
kill-switch being off) degrades to a not-ready `LLMContextOptions` --
never raises.

**Config** (`[image_generation]`, unit-1's flat-global-key convention):
`context_llm_enabled` (bool, default true), `context_llm_turns` (int,
default 10, clamped >=1), `context_llm_timeout_seconds` (float, default 15,
clamped >=0.1). New `Image_Generation/config.py` fields + `_coerce_bool`
helper; documented in `config.py`'s shipped `CONFIG_TOML_CONTENT`.

**Composition + pipeline reuse.** New `Chat/console_generate_image.py`
pieces: `LLMContextOptions` (resolved config + provider identity, plus a
`chat_call` test-injection seam defaulting to the real `chat_api_call`),
`compose_llm_context_prompt` (the actual `chat_api_call(streaming=False,
system_message=<concise-image-prompt instruction>)` call, response cleaned
via `_clean_llm_context_response` -- whitespace/newline collapse to one
paragraph, wrapping-quote strip, 500-char cap, refuse-empty -- never
raises, returns `None` on ANY failure), and `build_context_prompt_with_llm`
(tries the LLM path first, falls back to the untouched `build_context_
prompt` on `None`). Rather than bypassing the existing template machinery,
the LLM-composed text stands in for the keyword extractor's `last_message`
inside the identical context-dict shape `apply_template_to_prompt` already
consumes -- extracted the shared render+anchor-append logic into
`_apply_template_with_anchor` so both paths use it identically. This keeps
negative_prompt/params/style-label sourced from the resolved template
exactly as before (unchanged), and keeps the composed-prompt-visible-in-
card behavior structurally unchanged -- only the anchor text's source gets
richer. `prepare_generation_request` gained an optional `llm_context`
param (default `None` preserves every prior call site byte-for-byte).

**Threading.** The handler (`_console_command_generate_image`) previously
called `prepare_generation_request` directly on the UI loop. Since the
no-prompt path can now perform blocking network I/O, the whole decision
function now runs via `await asyncio.to_thread(prepare_generation_request,
...)` -- the same offload idiom `run_generation_batch` already used below
it -- resolved-provider-identity is fetched on the loop first, via the
same resolution a normal Console send does at that point (config/env for
most providers; llama.cpp additionally runs its own bounded `/health`
probe there too). Inside the threaded call, `compose_llm_context_prompt`
additionally bounds the network call itself to `context_llm_timeout_
seconds` via a dedicated single-worker `ThreadPoolExecutor` +
`future.result(timeout=...)`, `shutdown(wait=False)` on timeout (not the
`with:` form, which would otherwise block on exit waiting for the hung
call anyway).

**Fallback matrix** (kill-switch off / not-ready provider / resolution
exception / `chat_call` raises / timeout / empty response / garbage
response shape / executor failure) all proven with tests exercising the
REAL timeout machinery (a mocked 0.3s-sleep call against a 0.02s timeout),
not mocked abstractly -- every scenario falls back to the keyword-
extractor result and generation still dispatches; no exception ever
escapes. Never logs conversation content above debug level (turn count +
`repr(exception)` only). No new persistent state or caching -- each
invocation composes fresh.

Files: `tldw_chatbook/Chat/console_generate_image.py`,
`tldw_chatbook/UI/Screens/chat_screen.py`,
`tldw_chatbook/Image_Generation/config.py`, `tldw_chatbook/config.py`.
Tests: `Tests/Chat/test_console_generate_image.py` (+26: payload shaping,
response cleaning, the full fallback matrix on `compose_llm_context_
prompt`, `build_context_prompt_with_llm`, `prepare_generation_request`
threading), `Tests/Chat/test_console_generation_actions.py` (+10: 4
isolated `_console_generate_image_llm_context_options` tests + 6
handler-level end-to-end wiring tests incl. the fallback matrix and a
"prompt present -> LLM path never resolved" regression pin),
`Tests/Image_Generation/test_config_loader.py` (+5: defaults, overrides,
string-bool coercion, min-clamping). All TDD red-then-green (one genuine
red caught mid-authoring: a truncated `Edit` read left a stray dangling
assertion line, surfaced immediately as a `NameError`, diagnosed against
`git show HEAD:...` and fixed). Full suite green: 168 tests across the
three touched test files/dirs; broader `Tests/Chat -k "generate_image or
style_picker or console"` sweep (1115 tests) and `Tests/UI/test_personas_
expression_generate.py` (38) both green. `Tests/UI/test_console_native_
chat_flow.py` (199 tests) green on repeated full-file runs both before and
after this diff; one single flaky failure seen in an earlier combined run
was bisected via stash/unstash to be a pre-existing, order/timing-
dependent flake unrelated to this diff (passed identically on a clean
stashed checkout). `ruff check` clean on touched files (one pre-existing,
unrelated `config.py` F841 pair confirmed via `git blame` to predate this
branch, left untouched); `python -c "import tldw_chatbook.app"` clean.

Full report: `.superpowers/sdd/task-559-unit3-report.md`.

All four ACs now complete -- program done.
<!-- SECTION:NOTES:END -->
