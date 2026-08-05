# Character TTS Assignment and Roleplay Speech Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let a user assign one existing native audio.cpp TTS profile to an exact local or authority-scoped server character, then have manual Console Speak honor that immutable assignment while retaining the current global path for unassigned speech.

**Architecture:** Extend the existing app-owned profile service with one joined assignment read, then place a pure `CharacterTTSRequestResolver` between trusted Console snapshot validation and TTS request admission. Personas owns character authority proof and Textual freshness; focused Voice & Speech widgets only render immutable screen-owned state and emit intents. Assigned speech uses `TTSService.synthesize_exact`; unassigned and explicitly overridden speech reuse `synthesize_default` and the existing complete-audio artifact lifecycle.

**Tech Stack:** Python 3.11+, asyncio, Textual, immutable dataclasses, existing SQLite-backed `TTSProfileRepository`, existing `TTSProfileService`, pytest/pytest-asyncio.

**ADR required:** No.

**ADR path:** `backlog/decisions/028-character-tts-generation-profile-ownership.md` and `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`.

**Reason:** ADR-028 already fixes profile/assignment ownership and native request behavior. ADR-037 already fixes trusted character authorship and explicitly assigns visible controls plus runtime resolution to Slice 3B. No new storage, ownership, provider, or long-lived UX decision is introduced.

**Deliberate boundary:** Personas currently exposes only character soft delete. This plan verifies that soft delete/restore do not detach assignments and does not invent a permanent-delete or target-garbage-collection subsystem.

---

### Task 1: Add the exact assignment service read

**Files:**

- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Test: `Tests/TTS/test_profile_service.py`

**Step 1: Write failing service tests**

Add tests proving that the service:

- performs exactly one repository `get_assigned_profile(CharacterRef)` read;
- returns an immutable loaded result paired with the repository generation;
- admits exact canonical unassigned and assigned results;
- rejects the wrong character, profile/assignment mismatch, malformed joined values, wrong generations, and unexpected collaborator failures with bounded service errors;
- preserves the exact profile revision returned by the joined read.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_profile_service.py -q
```

Expected: the new tests fail because no service read exists.

**Step 2: Implement the minimal service extension**

Add `get_assigned_profile` to the repository protocol, introduce a frozen `LoadedCharacterTTSAssignment` value containing the repository generation and optional `AssignedTTSProfileSnapshot`, validate/canonicalize all collaborator output, and export the value through `tldw_chatbook.TTS`.

**Step 3: Run the focused tests**

Run the command from Step 1.

Expected: all profile-service tests pass.

**Step 4: Commit**

```bash
git add tldw_chatbook/TTS/profile_service.py tldw_chatbook/TTS/__init__.py Tests/TTS/test_profile_service.py
git commit -m "feat(tts): expose exact character assignment reads"
```

### Task 2: Resolve trusted character speech without fallback

**Files:**

- Create: `tldw_chatbook/TTS/character_request_resolver.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Test: `Tests/TTS/test_character_request_resolver.py`

**Step 1: Write failing resolver tests**

Cover:

- generic assistant message → global selection;
- exact unassigned `CharacterRef` → global selection after one joined read;
- exact assigned structurally valid profile → exact `TTSRequest` copying provider, model, voice, WAV format, speed, options, and current message text;
- malformed/corrupt joined assignment state → bounded fail-closed error with global override allowed;
- character-authored message without a `CharacterRef` → missing-authority failure with global override allowed;
- repository/service failures → bounded fail-closed result and no adapter call;
- local/server authority and same-ID character separation;
- explicit global override → global selection without reading or mutating assignment state.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_character_request_resolver.py -q
```

Expected: import/behavior failures.

**Step 2: Implement the resolver**

Create immutable resolution values for `global`, `assigned`, and `explicit_override`, plus a bounded `CharacterTTSResolutionError`. The resolver accepts only validated authorship facts and text supplied by the trusted snapshot path. For assigned state, copy the joined immutable profile directly into one exact request. Do not perform a catalog, voice-discovery, health, or availability preflight: native request admission remains the request-time authority for lazily exposed audio.cpp models and voices. Never inspect a concrete adapter, mutate an assignment, or fall back.

**Step 3: Run focused tests**

Run the command from Step 1.

Expected: all resolver tests pass.

**Step 4: Commit**

```bash
git add tldw_chatbook/TTS/character_request_resolver.py tldw_chatbook/TTS/__init__.py Tests/TTS/test_character_request_resolver.py
git commit -m "feat(tts): resolve character speech assignments"
```

### Task 3: Route Console Speak through exact assignment admission

**Files:**

- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/TTS/test_console_speech_snapshot_admission.py`
- Modify: `Tests/TTS/test_console_audio_cpp_native.py`
- Modify: `Tests/TTS/test_console_speak_autoplay.py`
- Modify: `Tests/TTS/test_tts_logging_privacy.py`

**Step 1: Write failing event/runtime tests**

Add tests proving:

- trusted snapshot validation still happens before profile resolution and cooldown;
- assigned resolution calls `synthesize_exact` once and never calls `synthesize_default`;
- unassigned and generic speech retain `synthesize_default`;
- any assigned resolution failure publishes fixed copy, consumes no synthesis admission, and offers exactly one explicit global override;
- an exact native synthesis/admission rejection fails safely without calling `synthesize_default` or selecting another model/voice;
- accepting the override revalidates the same trusted snapshot, calls only `synthesize_default`, and never mutates assignment state;
- declining the override performs no work;
- an externally constructed, unknown, expired, or replayed override decision is rejected without validation, cooldown, profile, or synthesis work;
- stale override snapshots remain rejected;
- assigned success follows the existing complete WAV artifact, autoplay, cleanup, cancellation, and privacy-safe metric path.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py Tests/TTS/test_console_speak_autoplay.py Tests/TTS/test_tts_logging_privacy.py -q
```

Expected: new exact-resolution assertions fail.

**Step 2: Extend the trusted speech event contract**

Keep `TTSMessageSpeechRequestEvent` assignment-aware by default and do not expose a caller-selectable override mode. On an eligible resolution failure, retain the original snapshot plus validator behind one bounded, expiring, single-use handler-issued opaque capability. Completion metadata may carry only that capability token. The app returns the token through a dedicated accept/decline decision event; the handler atomically consumes it before any validation or work. Unknown, expired, or replayed tokens fail closed. Do not expose raw caller-selected text or `CharacterRef`.

**Step 3: Resolve before cooldown and synthesize by mode**

Inject the app’s lazy `_ensure_tts_profile_service` loader into `TTSEventHandler`. After snapshot validation and text admission, invoke the resolver before writing cooldown state. Pass the immutable resolution into the existing generation task. Branch only at synthesis:

- assigned → call `synthesize_exact` and unpack its existing `(response, requested_selection)` result;
- global/unassigned/explicit override → `synthesize_default`.

Keep response closing, complete artifact writing, autoplay, cleanup, cancellation, and metric publication shared.

**Step 4: Add bounded recovery UI**

When an eligible resolution failure completes, launch the existing `ConfirmationDialog` from a dedicated Textual worker and offer “Use global for this message.” Always return an accept/decline decision for the exact opaque capability so the handler can consume it. Acceptance revalidates the retained snapshot before cooldown and global synthesis. Cancel, dialog failure, unknown token, expiry, and replay perform no speech work.

**Step 5: Run focused tests**

Run the command from Step 1.

Expected: all focused Console/TTS tests pass.

**Step 6: Commit**

```bash
git add tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py tldw_chatbook/app.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py Tests/TTS/test_console_speak_autoplay.py Tests/TTS/test_tts_logging_privacy.py
git commit -m "feat(tts): apply character profiles to console speech"
```

### Task 4: Add focused Voice & Speech controls

**Files:**

- Create: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`

**Step 1: Write failing widget tests**

Cover the compact control state:

- new/authority-missing character → disabled “Save/reopen before assigning” state;
- unassigned → “Use global default” selected;
- assigned → exact display name, availability, and assignment count;
- unavailable/unverified assigned profile remains visible with repair guidance and detach enabled;
- only available profiles can emit assign/replace;
- stale population cannot overwrite a newer character;
- button/select intents contain profile IDs/actions only, never guessed authority;
- the joined current-assignment read, profile page, and availability result must share one repository generation before publication;
- an assigned profile outside the first 50-row page remains selected and gets a separate one-profile availability observation whose repository, provider-configuration, and catalog revisions match the page observation;
- assign/replace passes the exact selected loaded profile plus caller-observed current assignment from that same generation;
- choosing “Use global default” and Remove both detach only the exact observed assignment;
- compare-and-set conflicts refresh instead of overwriting;
- local/server authority failure disables mutation, while server target or authenticated-principal changes reject late results;
- controlled repository restore interleavings reject mixed-generation population, assign/replace, and detach;
- soft delete and restore never call assignment detach, while unavailable/unverified assigned profiles remain visible and attached.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_workbench.py -q
```

Expected: widget imports and assertions fail.

**Step 2: Implement the presentation widget**

Build a small `PersonasCharacterTTSWidget` containing:

- a “Voice & Speech” label;
- profile selector with “Use global default”;
- availability/count/status copy;
- Preview, Create, Edit/Repair, and Remove assignment actions.

Mount one instance inside the existing character card scroll area and one inside the character editor scroll area. Only one is visible at a time; both receive the same immutable screen-owned state.

**Step 3: Prove exact character authority in the screen**

For local characters, read `get_local_authority_id` off-loop. For server characters, use the existing server-context capture/resolver/currentness fences and refuse assignment when stable authenticated authority cannot be proved. Construct `CharacterRef` only after those checks.

**Step 4: Load and mutate through screen-owned workers**

Use the lazy app-owned profile service to:

- load the exact assignment plus the bounded first profile page;
- observe availability off the Textual event path;
- set/replace using caller-held loaded profile, repository generation, and observed current assignment;
- detach the exact observed assignment;
- refresh after conflicts or profile edits.

The joined assignment result, profile page, and availability result must carry one identical repository generation. If the assigned profile is outside the bounded selector page, build a separate one-profile page from the joined immutable profile and observe only its UI availability. Require that observation and the selector-page observation to share repository generation, provider-configuration revision, and catalog revision before publishing either; any mismatch discards the entire screen snapshot and reloads. Mutations never combine assignment state from one generation with a profile from another. Each worker captures that shared generation, a UI request generation, exact `CharacterRef`, runtime source/authority context, selected entity, and mounted state. It publishes or mutates only while all remain current.

**Step 5: Reuse existing profile surfaces**

- Preview: create the existing `TTSPlaygroundSelectionPreset` and navigate to Speech Playground.
- Create: navigate to Speech Playground, where successful generated audio already supports “Save as profile.”
- Edit/Repair: open the existing `TTSProfileEditorModal`, pass the assignment count, call the existing service update, and refresh.

Do not add a second profile editor or direct adapter access.

**Step 6: Implement lifecycle preservation**

Preserve the existing soft-delete/restore behavior proven by Step 1: neither path calls assignment detach, and temporary unavailable/unverified status remains visible without mutation. Do not add a permanent-delete path.

**Step 7: Run focused tests**

Run the command from Step 1.

Expected: all Personas workbench tests pass.

**Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(personas): add character voice profile controls"
```

### Task 5: Accept Speech navigation context

**Files:**

- Modify: `tldw_chatbook/UI/Screens/stts_screen.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Create: `Tests/UI/test_speech_profile_navigation.py`

**Step 1: Write failing navigation tests**

Prove that process-local navigation context can:

- open the profile library;
- open Playground with one exact `TTSPlaygroundSelectionPreset`;
- reject malformed view/preset values;
- defer safely until the `STTSWindow` body mounts;
- consume and render a preset even when Playground is already the mounted view.

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_speech_profile_navigation.py -q
```

Expected: context behavior is absent.

**Step 2: Implement the minimal context bridge**

Add `STTSScreen.apply_navigation_context` that validates and retains pending context because app navigation invokes it before the deferred Lab body mounts. Apply it from `on_lab_body_ready`. Add a small `STTSWindow` method that selects an existing view and, when Playground is already mounted, explicitly reapplies/remounts it with the exact preset instead of relying on a same-value reactive assignment that cannot fire. No persistence or new route is added.

**Step 3: Run focused tests**

Run the command from Step 1.

Expected: all Speech screen tests pass.

**Step 4: Commit**

```bash
git add tldw_chatbook/UI/Screens/stts_screen.py tldw_chatbook/UI/STTS_Window.py Tests/UI/test_speech_profile_navigation.py
git commit -m "feat(speech): accept profile navigation context"
```

### Task 6: Cumulative verification and task closure

**Files:**

- Modify: `backlog/tasks/task-617.5 - Add-character-TTS-assignment-UI-and-roleplay-speech-runtime.md`

**Step 1: Run lint, formatting, and static checks**

```bash
git diff --check
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/TTS tldw_chatbook/Event_Handlers/TTS_Events tldw_chatbook/Widgets/Persona_Widgets tldw_chatbook/UI/Screens
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/TTS/profile_service.py tldw_chatbook/TTS/character_request_resolver.py tldw_chatbook/TTS/__init__.py tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Screens/stts_screen.py Tests/TTS/test_profile_service.py Tests/TTS/test_character_request_resolver.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py Tests/TTS/test_console_speak_autoplay.py Tests/TTS/test_tts_logging_privacy.py Tests/UI/test_personas_workbench.py Tests/UI/test_speech_profile_navigation.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --ignore F401 tldw_chatbook/UI/STTS_Window.py tldw_chatbook/app.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/TTS/profile_service.py tldw_chatbook/TTS/character_request_resolver.py tldw_chatbook/TTS/__init__.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py Tests/TTS/test_profile_service.py Tests/TTS/test_character_request_resolver.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py Tests/TTS/test_console_speak_autoplay.py Tests/TTS/test_tts_logging_privacy.py Tests/UI/test_speech_profile_navigation.py
```

Expected: every command exits 0. `tts_events.py`, `personas_screen.py`, `personas_character_editor_widget.py`, `stts_screen.py`, `STTS_Window.py`, `app.py`, and `test_personas_workbench.py` have known pre-task whole-file Ruff-format drift, so this task must keep their diffs conventionally formatted and `git diff --check` clean without mass-formatting unrelated baseline code. `STTS_Window.py` also has known pre-task unused-import findings; its focused lint command ignores only `F401`, while all new files and the remaining touched files use the ordinary lint gate.

**Step 2: Run the cumulative feature suite**

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS/test_profile_types.py Tests/TTS/test_profile_repository.py Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_service.py Tests/TTS/test_character_request_resolver.py Tests/TTS/test_tts_profile_capabilities.py Tests/TTS/test_tts_app_ownership.py Tests/TTS/test_console_speech_snapshot_admission.py Tests/TTS/test_console_audio_cpp_native.py Tests/TTS/test_console_speak_autoplay.py Tests/TTS/test_tts_logging_privacy.py Tests/UI/test_personas_workbench.py Tests/UI/test_stts_profile_library.py Tests/UI/test_speech_profile_navigation.py -q
```

Expected: all tests pass.

**Step 3: Run targeted full-module regressions**

```bash
PYTHONPYCACHEPREFIX=/tmp/tts-slice3b-pycache /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/TTS Tests/UI/test_console_native_chat_flow.py -q
```

Expected: all tests pass.

**Step 4: Self-review and request code review**

Inspect the branch diff for:

- any raw text, authority IDs, profile IDs, model IDs, or voice IDs added to logs;
- any fallback after assigned resolution begins;
- event-loop database/capability work;
- stale Textual publications;
- duplicate profile-management behavior;
- scope creep into automatic speech, managed audio.cpp, Persona inheritance, Sync, or portability.

Then use `superpowers:requesting-code-review` and address every verified issue.

**Step 5: Close the Backlog task**

Check every acceptance criterion, add concise Implementation Notes including the no-permanent-delete finding, and set TASK-617.5 to Done only after all verification and review gates pass.

**Step 6: Commit documentation**

```bash
git add "backlog/tasks/task-617.5 - Add-character-TTS-assignment-UI-and-roleplay-speech-runtime.md" Docs/superpowers/plans/2026-07-31-tts-character-assignment-runtime.md
git commit -m "docs(tts): record slice 3b implementation"
```
