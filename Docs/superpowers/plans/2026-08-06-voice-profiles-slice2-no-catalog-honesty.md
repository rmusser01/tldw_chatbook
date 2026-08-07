# Voice Profiles Slice 2 — "No catalog check" honesty: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A legacy-provider voice profile is never presented to the user with the raw word "Unverified" or refresh-promising copy — every surface tells the honest story: this provider has no catalog check, its exact selection is used as-is.

**Architecture — deliberate spec interpretation (record, do not re-litigate):** Spec §4.2 asks for an "explicit, permanent 'No catalog check for this provider' state." That state is implemented at the PRESENTATION layer, not as a fourth `ProfileAvailabilityState` value. Grounding verified availability never crosses a serialization boundary, but a new enum value would have to be taught to ~15 gates, validators, CSS state loops, and dozens of tests — all of which would treat it exactly like "unverified" — and "a new vocabulary value not taught to every surface" is this program's most-repeated shipped defect (6+ instances). The machine-readable distinction ALREADY exists: `recovery_action == "none"` on an `"unverified"` availability means permanent-no-catalog (slice-3's I3 fix established this, with `_ALLOWED_RECOVERY_ACTIONS` documenting it). Slice 2 finishes carrying that distinction to every user-visible string.

**Grounding note:** file:line refs below were verified at dev `75bc25db3`; dev has since merged a Console decomposition wave. Treat line numbers as approximate — re-locate by symbol/string, and if a named string is ABSENT, stop and check whether another session already changed it before assuming the brief is wrong.

## Global Constraints

- The distinguishing key is `recovery_action == "none"` (or provider class where availability objects aren't in hand) — never a new availability value, never `provider_id != "audio_cpp"` string comparisons in UI copy branches when a recovery_action is available.
- audio.cpp's transient "Unverified — Refresh and retry" story is UNCHANGED everywhere.
- Out of scope (verified distinct concerns): the adapter-level `VoiceDiscoveryState`/`CapabilitySnapshotState` "unverified" (audio.cpp catalog freshness); the audio.cpp voice-pin "(Unverified)" suffix in `_pinned_audio_cpp_option`; Settings axes "Model: Unverified/Voice: Unverified"; the retired `TTSPlaygroundWidget` in `UI/STTS_Window.py` (dead code — zero instantiation sites; do not edit it).
- Copy vocabulary, exactly: the short form is **"No catalog check"**; the sentence form is **"This provider has no catalog check; the exact selection is used as-is."** Keep I3's existing tails where they already say the assignment/selection is preserved.
- TDD with mutation checks; venv pytest only (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`); targeted files + blast-radius greps; never `git stash`; never `git checkout <file>`; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- After changing ANY copy: grep the repo for the OLD literal phrases you replaced (the free-floating-string lesson — an identifier grep cannot see prose restating stale assumptions).

---

### Task 1: Profile library surfaces

**Files:**
- Modify: `tldw_chatbook/UI/stts_profile_library.py` (DataTable availability cell ~:1199 and ~:1227; detail status line ~:1339-1350; `_PROFILE_UNVERIFIED_COPY` ~:121-124; header purpose string ~:791)
- Test: `Tests/UI/test_stts_profile_library.py`

**Requirements:**
1. **DataTable "Availability" cell** (currently `Text(item.state.title())` — renders raw "Unverified" for legacy, and is UNPINNED by any test): render **"No catalog check"** when the availability's `recovery_action == "none"` and state is `"unverified"`; keep "Available"/"Unavailable"/"Unverified"(audio_cpp transient)/"Checking" exactly as today. Both cell-fill sites (`_publish_page` and `_publish_availability`) — extract one helper so they cannot diverge.
2. **Detail status line** (I3-shipped): legacy branch currently reads "Unverified — this provider has no catalog check." Change to **"No catalog check — the exact selection is used as-is."** The audio_cpp branch ("Unverified — Refresh and retry.") is untouched. Update the existing pins (`test_profile_recovery_copy_is_visible_at_80x24`, `test_unverified_legacy_profile_never_offers_a_refresh_it_cannot_perform`) — the latter's protective intent (no refresh/retry promise) must be preserved, extended to also assert the word "unverified" is absent for legacy.
3. **`_PROFILE_UNVERIFIED_COPY` toast** ("The exact profile selection could not be verified. Refresh and retry…"): FIRST prove reachability — grounding says `_require_authoritative_capability` returns early for legacy providers (slice 1), so the `profile_unverified` service error should be audio_cpp-only in practice. If you confirm legacy CANNOT trigger it (trace every raise of `profile_unverified` and `stale_configuration`), leave the copy and add a code comment stating the audio_cpp-only reachability with the trace result. If legacy CAN trigger it, make the copy honest for both classes. Report which you found with evidence.
4. **Header purpose string** (P4): "Manage exact native audio.cpp model and voice selections." → **"Manage exact model and voice selections for every speech provider."**
5. New test pinning the DataTable cell for all four rendered forms (legacy no-catalog, audio_cpp unverified, available, unavailable); mutation-check by reverting the cell helper to `.state.title()`.

- [ ] TDD steps as per global constraints; commit `feat(tts): profile library tells the no-catalog-check story for legacy profiles (slice 2, task 1)`

---

### Task 2: Personas surfaces

**Files:**
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py` (Select option label ~:195), `tldw_chatbook/UI/Screens/personas_screen.py` (status line ~:1772-1791)
- Test: `Tests/UI/test_personas_workbench.py`

**Requirements:**
1. **Select option label** (currently `f"{profile.display_name} · {profile.availability}"` — renders "· unverified"): for an option whose availability is `"unverified"`, render **"· no catalog check"** when its recovery action is inert, "· unverified" otherwise (audio_cpp transient). NOTE: check what the option object carries — `CharacterTTSProfileOption` may hold only the state string; if it lacks recovery_action, extend the option dataclass (its `__post_init__` validates state membership — extend carefully) and thread the value from where options are built. Do NOT branch on provider_id in the widget.
2. **Status line** (I3-shipped): legacy branch currently "· Unverified · {count}. This provider has no catalog check; the assignment is preserved." → **"· No catalog check · {count}. The exact selection is used as-is; the assignment is preserved."** audio_cpp branch untouched.
3. Update the pinned tests, preserving each one's protective intent: `test_character_tts_widget_accepts_unverified_profile_assignment_without_laundering_it` currently asserts `"unverified" in label` — its intent is NO LAUNDERING (the label must visibly differ from a plain/available profile); rewrite it to assert the label contains "no catalog check" and does NOT read as available. `test_character_tts_unverified_status_never_promises_an_impossible_refresh` keeps its no-refresh-promise assertions with the new copy.
4. Mutation-check: revert the label branch → the no-laundering test fails.

- [ ] TDD; commit `feat(tts): personas voice surfaces say no-catalog-check instead of unverified for legacy (slice 2, task 2)`

---

### Task 3: Playground adoption copy

**Files:**
- Modify: `tldw_chatbook/UI/Speech/speech_profile_mixin.py` (~:335-339), `tldw_chatbook/UI/Speech/speech_catalog_mixin.py` (~:559-561, ~:608-610, ~:613-616, ~:663-666, ~:958-961)
- Test: the files covering these mixins (locate via grep — likely `Tests/UI/test_speech_playground_pane.py` and `Tests/UI/test_stts_playground_catalog.py`)

**Requirements:**
1. Five adoption-status strings currently fire identically for audio_cpp-transient and legacy-permanent unverified (e.g. "Profile availability is unverified. Generate makes one exact attempt without fallback and shows a warning."). For a LEGACY adopted preset, each must instead say the no-catalog story, e.g. **"This provider has no catalog check. Generate makes one exact attempt without fallback."** — keep each site's existing behavioral tail (the one-exact-attempt/no-fallback fact is true for both classes and stays).
2. The mixins hold the adopted `TTSPlaygroundSelectionPreset`, which carries `provider_id` but availability presets may not carry recovery_action — investigate what's in hand at each site; if only provider class is available there, branching on the preset's provider class is acceptable HERE (unlike UI widgets fed availability objects) — say so explicitly in the report if you take that path, and centralize the class test in ONE helper shared by all five sites so a sixth site cannot diverge.
3. Do NOT touch the audio_cpp voice-pin "(Unverified)" suffix or the retired `TTSPlaygroundWidget` duplicates.
4. Tests: for at least the two highest-traffic sites (preview banner `_sync_profile_preview_status`, provider-status line `_apply_controls`), pin legacy vs audio_cpp copy divergence; mutation-check the shared helper.
5. After the change, grep the repo for each OLD literal phrase you replaced and confirm zero remaining hits outside the retired `TTSPlaygroundWidget` (which is exempt, documented dead).

- [ ] TDD; commit `feat(tts): playground adoption distinguishes no-catalog providers from transient unverified (slice 2, task 3)`

---

### Task 4: Gates, live check, ship prep

- [ ] Full targeted sweep: the three tasks' test files + `Tests/TTS/test_profile_service.py` + `Tests/UI/test_stts_playground_catalog.py`; repo-wide `--collect-only`; ruff check + format on all touched files.
- [ ] OLD-PHRASE sweep (the free-floating-string lesson, run as its own numbered step): grep for "Unverified — this provider has no catalog check", "· Unverified ·" (legacy paths), "Profile availability is unverified", "Profile preview unverified" — every remaining hit must be either audio_cpp-scoped-and-true or inside the documented-dead `TTSPlaygroundWidget`.
- [ ] Live check (tmux, scratch config): Lab ▸ Speech ▸ Voice Profiles — with at least one legacy profile present (create via Playground save if the store is empty), observe the library row showing "No catalog check" and the detail line; adopt a legacy profile in the Playground and observe the adoption banner. If creating a profile live is impractical, drive the surfaces via the Textual pilot harness and say plainly what was and was not observed live.
- [ ] Backlog: scan ALL worktrees for max task id, leapfrog with headroom; file the slice-2 task Done with Implementation Notes (including the enum-vs-presentation interpretation and its rationale). Update the spec's Status line for slice 2 in `Docs/superpowers/specs/2026-08-04-voice-profiles-expansion-design.md`.
- [ ] Do NOT open a PR and do NOT merge — the controller owns that.

## Self-review notes

- §4.2 coverage: no-catalog presentation (T1-T3), library header P4 (T1), availability semantics already machine-encoded by I3 (no enum change — recorded interpretation). The `_PROFILE_UNVERIFIED_COPY` item is investigate-then-act with the decision rule stated.
- Deliberately untouched: assignment/import gates (they already treat unverified as assignable — behavior is correct; only presentation changes), adapter-level enums, Settings picker (verified clean), dead widget.
