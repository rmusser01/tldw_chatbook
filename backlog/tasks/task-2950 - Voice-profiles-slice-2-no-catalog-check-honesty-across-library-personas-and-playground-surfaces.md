---
id: TASK-2950
title: >-
  Voice profiles slice 2: no-catalog-check honesty across library, personas, and
  playground surfaces
status: Done
assignee: []
created_date: '2026-08-07 02:08'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Legacy-provider voice profiles (openai, elevenlabs, kokoro, chatterbox, higgs, alltalk) were presented with the raw word 'Unverified' or refresh-promising copy on several surfaces, even though these providers have no catalog to preflight -- their unverified state is permanent, not a transient glitch Refresh could resolve. This falsely implied a recovery path that can never succeed (ADR-031) and did not distinguish them from audio.cpp's genuinely transient unverified state. This slice makes every such surface tell the honest 'No catalog check' story instead.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Profile library DataTable Availability cell renders 'No catalog check' (not raw 'Unverified') for legacy-provider profiles; audio.cpp's transient Unverified is unchanged
- [x] #2 Profile library detail-pane status line reads 'No catalog check -- the exact selection is used as-is.' for legacy profiles
- [x] #3 Profile library header purpose string no longer claims audio.cpp exclusivity
- [x] #4 Personas Voice & Speech Select option label and status line say 'no catalog check' (not 'unverified') for legacy-provider profile assignments without laundering them as available
- [x] #5 Playground adoption-status copy (preview banner and provider-status line, across both catalog-load-failure branches) distinguishes a no-catalog-check legacy provider from audio.cpp's transient unverified state through one shared helper
- [x] #6 A repo-wide grep for the four replaced OLD phrases shows every remaining hit is either audio_cpp-scoped-and-true or inside the documented-dead TTSPlaygroundWidget
- [x] #7 Targeted test suite, repo-wide --collect-only, and ruff check/format pass on every touched file (excluding known pre-existing dead-widget drift)
- [x] #8 Live-verified in the real TUI: a legacy profile's library row, detail line, and Playground adoption banner all show the new copy with zero 'unverified' occurrences
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Profile library surfaces: DataTable cell + detail status line + header P4 string (Docs/superpowers/plans/2026-08-06-voice-profiles-slice2-no-catalog-honesty.md Task 1). 2. Personas Select option + status line (Task 2). 3. Playground adoption copy across preview banner and provider-status sites via one shared helper (Task 3). 4. Gates, OLD-phrase sweep, live check, backlog closeout (Task 4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

Executed via `Docs/superpowers/plans/2026-08-06-voice-profiles-slice2-no-catalog-honesty.md`
(commits `7867416eb..8966edb46`, branch `feat/voice-profiles-slice2`). Full
detail in `.superpowers/sdd/2026-08-06-voice-profiles-slice2-no-catalog-honesty/`
(`progress.md`, `task-1..3-report.md`, `task-4-report.md`).

**Enum-vs-presentation interpretation (recorded, not re-litigated).** Spec
§4.2 asks for an "explicit, permanent 'No catalog check for this provider'
state." This was implemented at the PRESENTATION layer, not as a fourth
`ProfileAvailabilityState` value. The machine-readable distinction already
exists: `recovery_action == "none"` on an `"unverified"` availability means
permanent-no-catalog (slice-3's I3 fix established this, with
`_ALLOWED_RECOVERY_ACTIONS` documenting it) vs. `recovery_action == "refresh"`
for audio.cpp's genuinely transient unverified state. A new enum value would
have had to be taught to ~15 gates, validators, CSS state loops, and dozens
of tests, all of which would treat it exactly like `"unverified"` -- and "a
new vocabulary value not taught to every surface" is this program's
most-repeated shipped defect (6+ prior instances, see
`backlog/docs/lessons-live-verification.md`'s "A backend fix... can still be
dead code" entry and its three addenda). Grounding availability never
crosses a serialization boundary either way; only its presentation changed.

**What shipped (tasks 1-3).** Legacy-provider voice profiles (openai,
elevenlabs, kokoro, chatterbox, higgs, alltalk) are never shown as raw
"Unverified" or given a refresh/retry promise they cannot fulfill:
- Profile library: DataTable Availability cell + detail-pane status line +
  header purpose string (P4), one shared cell-fill helper so the two
  DataTable update sites cannot diverge.
- Personas "Voice & Speech": Select option label + status line, keyed off a
  `recovery_action` field added to `CharacterTTSProfileOption`.
- Playground: 4 adoption-status sites (preview banner, provider-status line,
  two catalog-load-failure branches) via one shared `preset_has_no_catalog_check()`
  helper in `stts_playground_catalog.py`, since the mixins there only have
  the preset's provider class in hand (no `recovery_action`-bearing object),
  which the plan's Global Constraints explicitly permits at that layer.
- All keyed off `recovery_action` (or provider class where availability
  objects aren't in hand) per the plan's Global Constraints -- never a raw
  `provider_id != "audio_cpp"` comparison in a UI copy branch when a
  recovery_action was available.
- audio.cpp's transient "Unverified -- Refresh and retry" story is
  byte-for-byte unchanged everywhere.

**Gates (task 4, re-run at HEAD `8966edb46`).** Targeted suite (the three
tasks' test files + `Tests/TTS/test_profile_service.py` +
`Tests/UI/test_stts_playground_catalog.py` + `Tests/UI/test_stts_playground_audio_cpp.py`):
**734 passed, 14 failed** in `Tests/UI/test_stts_playground_audio_cpp.py`
only -- confirmed pre-existing dead-widget drift (see task-2951), same 14
test IDs task 3 already isolated as unrelated to this slice's changes.
Repo-wide `pytest --collect-only`: **31539 tests collected, no import
errors.** `ruff check` on all 11 touched files: clean. `ruff format --check`:
2 files (`tldw_chatbook/UI/Screens/personas_screen.py`,
`Tests/UI/test_personas_workbench.py`) flagged, both confirmed as
PRE-EXISTING drift unrelated to this branch's diff hunks (identical `ruff
format --check` result reproduced against each file's merge-base copy).

**OLD-phrase sweep.** Grepped all four literal phrases the plan named
("Unverified — this provider has no catalog check", "· Unverified ·",
"Profile availability is unverified", "Profile preview unverified") across
`tldw_chatbook/`. Every remaining hit is either (a) the audio.cpp `else`
branch of a new `if <no-catalog-check>: ... else: ...` split this slice
added, confirmed correct by reading each in context, or (b) inside the
documented-dead `TTSPlaygroundWidget` (`UI/STTS_Window.py`), exempt per the
plan's Global Constraints. Zero unaccounted-for hits.

**Live verification** -- see the LIVE VERIFICATION section of
`task-4-report.md` for the full transcript; summary: a legacy `openai`
profile was seeded directly via `TTSProfileRepository` into a scratch
profile (`TLDW_CONFIG_PATH`/`HOME`/`XDG_*` isolated), the real TUI was
driven via tmux (F7 → `]` → Enter to Lab ▸ Speech, mouse-clicked to Voice
Profiles), and the library row/detail pane, the Playground preview banner,
provider-status line, and the Studio-preferences adoption screen all showed
the new "No catalog check" / "no catalog check" copy with **zero**
"unverified" occurrences anywhere in the captured panes.

**Carry-forward findings filed, not fixed here** (both out of this plan's
named scope, found during task 3's review):
- **task-2951**: task-1266 is marked Done but its AC#4 ("TTSPlaygroundWidget
  is deleted") is false on `dev` -- the class was restored by reconciliation
  commit `f9d7e6269` and never re-deleted.
- **task-2952**: `_profile_preview_blocked_presentation`
  (`speech_profile_mixin.py:178-218`) can return "refresh or retry" copy for
  an unverified preset without checking provider class, running ahead of
  this slice's honest branch in `_sync_profile_preview_status`; legacy
  reachability of its three `"unverified"`-returning branches was not traced
  and remains unresolved.

**Deliberately untouched** (per the plan's Global Constraints, verified
distinct concerns): adapter-level `VoiceDiscoveryState`/`CapabilitySnapshotState`
"unverified" (audio.cpp catalog freshness); the audio.cpp voice-pin
"(Unverified)" suffix; Settings axes "Model: Unverified/Voice: Unverified";
assignment/import gates (already correctly treat unverified as assignable,
only presentation changes); the edit-dialog free-text model/voice inputs for
legacy providers (already shipped in slice 1, `cfbe73854`/`a629246bf`).
