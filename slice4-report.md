# Voice profiles — slice 4 (cross-links + docs) — implementation report

Branch: `feat/voice-profiles-slice4`, off dev `7f23e0263` (PR #1397 / slice 2
merged). Spec: `Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md`
§4.4.

## What was already documented vs. what I added

- **`Docs/User_Guide/openai-compatible-tts.md`** already had an "App-wide
  default voice profile" section (written during slice 3, verified against
  `0c24f50d9`). It did **not** yet cover per-character voices, so I added a
  new `## Per-character voices` section directly after it (before "Quirks &
  troubleshooting"), plus a "Related settings & docs" link and an updated
  "Verified against" stamp.
- **`Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`**
  already documents the Voice & Speech block in detail (Preview / Create /
  Edit-Repair / Remove, loading/failure copy) — verified against `8b7fa5eb6`
  (2026-07-31), predating slices 1-4. I did **not** rewrite that page (out of
  scope for this slice and it would need its own live-driving pass); I only
  added one cross-link back to the new "Per-character voices" section in
  `openai-compatible-tts.md`, since the two pages now describe the same
  feature from two angles (TTS-config side vs. character-editor side) and
  neither pointed at the other.
- **`Docs/User_Guide/index.md`** — no Speech & TTS-adjacent page exists yet
  from any other session's work (`lab.md` is still a 🚧 stub, no dedicated
  Voice Profiles or Speech Lab guide page). `openai-compatible-tts.md`
  remains the correct target; I extended its one-line index description to
  mention per-character voices, no new page created.
- **TASK-2451** (sample persona) was already filed, confirmed at
  `backlog/tasks/task-2451 - Make-the-default-assistant-an-editable-sample-persona.md`,
  status "To Do", dependencies TASK-2450/TASK-951, citing ADR-037. Not
  re-filed. Referenced by id in the spec's Status block and §4.4 correction
  (see below).

## Deliverable 1 — Settings pointer card

`tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`,
inside the existing `#settings-speech-scope-banner` card (the one holding
the "You are editing application-wide Speech & TTS defaults..." note and
the **Open Speech Lab** button), a new `Static` was added directly after
that button:

> "Voice profiles are managed in Lab > Speech > Voice Profiles — open
> Speech Lab, above, to get there. Per-character voices are assigned in the
> Roleplay character editor's Voice & Speech section, not here."

`id="settings-speech-profile-surfaces-note"`, `classes="settings-detail-row"`
— matches the panel's existing note convention (same class used by the
credential-editor and audio.cpp advisory notes elsewhere in this file).
No new button: it explicitly references the *existing* "Open Speech Lab"
button ("above") rather than adding a second affordance, per the task's
constraint. In-app copy uses plain `>` for the breadcrumb, matching this
codebase's existing convention for in-widget cross-references (e.g.
`personas_preview_pane.py`'s "Open Settings > Providers & Models..."
tooltip) — Markdown docs use `▸` instead, per that convention's own
established use (`index.md`, `settings.md`).

### TDD evidence

1. **RED**: added `test_scope_banner_points_to_the_two_profile_surfaces` to
   `Tests/UI/test_settings_speech_tts_panel.py` before touching the widget;
   ran it alone — failed with `NoMatches: No nodes match
   '#settings-speech-profile-surfaces-note'`.
2. **GREEN**: added the `Static` to the panel; re-ran the same test alone —
   passed.
3. **Full file**: `Tests/UI/test_settings_speech_tts_panel.py` — **89
   passed, 0 failed** (was 88 tests before my addition; +1 new test).
   `test_production_settings_actions_cross_the_pushed_screen_boundary` — the
   test flagged in my brief as having one pre-existing unrelated failure —
   **passed** in this run and in 3 repeated isolated re-runs. I did not
   touch anything on its path; it is presumably flaky/timing-sensitive or
   was fixed incidentally by something already on dev between when that
   note was written and `7f23e0263`. Net result either way: no regression,
   verified by direct observation rather than assumption.
4. **Real-CSS smoke check** (ad hoc, not committed): rendered the panel
   through `_StyledDestinationHarness` (the real `tldw_cli_modular.tcss`
   bundle, same pattern as
   `test_real_stylesheet_keeps_fields_usable_and_outer_detail_scrollable`)
   and queried the new note. It renders at 3 lines, region
   `Region(x=41, y=14, width=102, height=3)`, directly below the Open
   Speech Lab button (`Region(x=41, y=13, width=17, height=1)`) — visible,
   not clipped, not zero-size.

### Label verification (read from source, not assumed)

- "🗣️ Voice Profiles" — exact tab label, `UI/Screens/stts_screen.py:51`
  (`SPEECH_RAIL_SECTIONS`), reached via Lab (F7) ▸ Speech.
- "Save result as profile" — exact button label,
  `UI/Speech/speech_playground_pane.py:1397` and `UI/STTS_Window.py:1673`.
- **"Voice & Speech" is the real, literal section title** — confirmed at
  `Widgets/Persona_Widgets/personas_character_tts_widget.py:180`
  (`Static("Voice & Speech", classes="destination-section")`), used by both
  `personas_character_editor_widget.py:482` (the character editor) and
  `personas_character_card_widget.py:111` (the character card). No
  divergence from what the spec assumed — flagged as a thing to check per
  my instructions, and it checked out.
- "No catalog check" (title case) — `UI/stts_profile_library.py:150`
  (`_PROFILE_NO_CATALOG_CHECK_COPY`), used in the profile library's
  Availability column and in `personas_screen.py:1839`'s character-widget
  status line ("{name} · No catalog check · {count}...").
- "no catalog check" (lowercase) — the character voice picker's own option
  suffix, `personas_character_tts_widget.py:116`
  (`_character_tts_option_suffix`), rendered as `"{display_name} · no
  catalog check"`.
- The guide's "Per-character voices" section quotes both cases correctly,
  attributed to the surface each appears on.

## Deliverable 2 — User Guide

`Docs/User_Guide/openai-compatible-tts.md`: new `## Per-character voices`
section (between "App-wide default voice profile" and "Quirks &
troubleshooting") covering: where profiles are created (Speech Lab
Playground → "Save result as profile", works for all seven providers, not
audio.cpp-only); where they're assigned (Roleplay & Chat Dictionaries ▸
Characters ▸ Voice & Speech, with a link to the existing full walkthrough
in `characters-and-personas.md`); the restated precedence (character voice
> app default profile > axes > provider fallback); and what "No catalog
check" means and why it's not an error. Also added a "Related settings &
docs" link and a second "Verified against dev @ 7f23e0263" stamp entry
disclosing that this pass verified labels from source, not from a live
driving session (the existing default-profile precedence bullet above my
addition was not re-verified live in this pass — only what I wrote was).

`Docs/User_Guide/index.md`: one-line How-to-guides description extended to
mention per-character voices alongside the existing default-profile
mention.

`Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`:
added one cross-link in "Related settings & docs" back to the new section.

## Deliverable 3 — Spec status close-out

`Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md`:

- Status block rewritten as a per-slice bulleted list: all four slices
  shipped, with PR numbers and dev shas — slice 1 #1368 → `e4f7aa24e`,
  slice 2 #1397 → `7f23e0263` (was "pending merge" in the doc; confirmed
  merged via `git log --merges` on this branch's own base), slice 3 #1375
  → `e7b9ebabd`, slice 4 (this branch, off `7f23e0263`).
- §4.4 got a "Landed, 2026-08-07" addendum correcting one thing against
  what actually happened: the sample-persona follow-up task (§4.4's third
  bullet) was filed during **slice 1's** scoping, not slice 4's, as
  **TASK-2451** — the id wasn't known when §4.4 was originally written, so
  I added it there rather than silently editing the original bullet.
  Everything else in §4.4 landed as written (confirmed the pointer-card
  placement and the Roleplay editor's literal "Voice & Speech" title, both
  above).

## Gates

- `ruff check` + `ruff format --check` on both touched Python files: clean.
- `Tests/UI/test_settings_speech_tts_panel.py`: 89 passed, 0 failed (see
  TDD evidence above for the one flagged test's actual status this run).
- `Tests/UI/test_speech_tts_settings_ownership_closeout.py`: 8 passed, 1
  failed (`test_first_time_audio_cpp_setup_lab_generation_and_console_handoff`,
  an `AttributeError` inside `_generate_tts_worker` — a live TTS-generation
  smoke test, nothing to do with this branch's changes). Confirmed
  pre-existing by checking out the unmodified base commit `7f23e0263` into
  a disposable worktree (`git worktree add --detach ... 7f23e0263`) and
  running the same test there: **identical failure**, byte-for-byte same
  traceback shape. Worktree removed afterward
  (`git worktree remove --force`); nothing left behind.
- Repo-wide `pytest --collect-only -q`: **31734 tests collected**, zero
  collection errors (only pre-existing, unrelated deprecation warnings).
- No PR opened, no merge performed, per instructions.

## Files touched

- `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- `Tests/UI/test_settings_speech_tts_panel.py`
- `Docs/User_Guide/openai-compatible-tts.md`
- `Docs/User_Guide/index.md`
- `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`
- `Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md`
