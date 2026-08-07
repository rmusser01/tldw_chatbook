# Voice profiles beyond audio.cpp + app-wide default voice profile — design

Date: 2026-08-04
Status: **Approved. All four slices shipped to dev — program complete 2026-08-07.
**

- **Slice 1** (§4.1, all seven providers) — shipped 2026-08-06, PR #1368 →
  dev `e4f7aa24e`, TASK-2450.
- **Slice 2** (§4.2's availability-honesty + P4 copy items) — shipped
  2026-08-07, PR #1397 → dev `7f23e0263`, TASK-2950 — legacy-provider
  profiles are never presented with the raw word "Unverified" or a refresh
  promise they cannot fulfill, across the profile library, Personas, and
  Playground surfaces (implemented at the presentation layer per
  `recovery_action`, not as a new `ProfileAvailabilityState`; see
  TASK-2950's Implementation Notes for the rationale). §4.2's edit-dialog
  provider-set expansion was already covered by slice 1. Two follow-ups
  filed, not blocking: TASK-2951 (a stale AC on the unrelated task-1266
  board entry), TASK-2952 (one untraced legacy-reachability question in the
  Playground preview's blocked-state copy).
- **Slice 3** (§4.3, default voice profile end-to-end) — shipped
  2026-08-06, PR #1375 → dev `e7b9ebabd`.
- **Slice 4** (§4.4, cross-links + docs) — **authored on branch
  `feat/voice-profiles-slice4` (off dev `7f23e0263`), PR pending, not yet
  merged.** Adds the Settings ▸ Speech & TTS pointer card, a new
  "Per-character voices" section in `Docs/User_Guide/openai-compatible-tts.md`,
  and this status close-out. §4.4's text said "file the follow-up task for
  the sample-persona idea (ruling 1)"; that filing happened during slice-1
  scoping, ahead of this slice, as **TASK-2451** ("Make the default
  assistant an editable sample persona") — noted here since §4.4 did not
  carry the id when it was written. This bullet — and the "complete" framing
  below — get corrected to shipped/merged in the wake of slice 4's merge
  commit; do not treat this document as evidence the branch has landed.

Slice 4 was deliberately left unplanned in this document until slice 3
landed — each slice was planned only after the prior one shipped, because
slice 1 demonstrated how much implementation reshapes a plan (see the §3
correction). That held for all four: this status block is the only design
artifact for slice 4, and no separate plan document exists for it.

Owner decisions recorded: 2026-08-04 (five rulings, §2)
Extends: ADR-023, ADR-028, ADR-037, ADR-039

## 1. Problem

Two user asks, grounded against dev `265dbd687`:

1. **Per-character/persona voice settings that apply.** This *exists* — the Roleplay
   character editor's "Voice & Speech" section assigns a Voice Profile per character
   (`Widgets/Persona_Widgets/personas_character_tts_widget.py`), and Console's 🔊 speak
   resolves it through the character-aware path (`TTSMessageSpeechRequestEvent` →
   `CharacterTTSRequestResolver` → `character_profile` precedence source). But it is
   **audio.cpp-only**: every other provider is refused, so for users on OpenAI,
   ElevenLabs, Kokoro, Chatterbox, Higgs, AllTalk — or an OpenAI-compatible local
   server (pocket-tts, TASK-2260) — the entire feature is inert.
2. **An app-wide default "assistant" voice.** Today the no-character voice is the raw
   Global-defaults axes in Settings ▸ Speech & TTS. There is no way to point the app
   default at a *named voice profile*, so the "assistant voice" cannot be managed as
   one reusable object alongside character voices.

## 2. Owner rulings (2026-08-04)

1. **Scope:** voice settings only. The "default assistant becomes a sample persona"
   idea is a separate follow-up task, not this feature.
2. **Item-1 shape:** extend provider coverage and cross-link from Settings.
   Per-character assignment **stays in the Roleplay editor** (ADR-039 scope
   separation); no per-character management surface inside Settings.
3. **Default shape:** "Default voice profile" selector *above* the Global-defaults
   axes — live-linked named profile, or "None — use the fields below". The axes remain
   as fallback. Non-breaking.
4. **Rollout:** all six legacy-bridge providers in one expansion (not OpenAI-first).
5. **Failure mode (defaults):** when the default profile cannot be used at speak
   time, **refuse + one-tap override** ("Speak with global defaults"), matching the
   existing character-voice failure pattern. Never silently substitute a voice.

## 3. Current state — the audio.cpp pins

> **Correction, recorded 2026-08-06 after slice 1 shipped.** This section named four pins,
> derived by reading. Implementation found **eight-plus**: the four below, plus the
> `TTSRequestedSelectionSnapshot` construction pin (playground provenance), the
> `PortableTTSProfile` construction pin (portable/chatbook import), the
> `commit_portable_profile_import` auto-assign gate, and the Roleplay assignment path's
> *two* independent availability gates (widget handler + assignment worker). Two further
> defects were emergent rather than pins: `observe_availability` coupled all-legacy pages
> to audio.cpp health, and playground adoption forced legacy presets to "unavailable".
>
> The generalisable lesson, which cost three fix rounds: **a pin count derived from reading
> is a lower bound, and a new state must be taught to every surface that reads it.** Only
> driving the real TUI plus a classified grep of every availability comparison closed it.

The profile **store is already provider-agnostic**: `TTS/profile_repository.py` has no
provider pin, and `TTS/profile_types.py` applies the WAV / speed-1.0 / empty-options
constraints only when `provider_id == "audio_cpp"` (`_validate_audio_cpp`). The pin
lives in four gates, each a distinct work item:

| # | Site | Semantics |
|---|------|-----------|
| P1 | `TTS/character_request_resolver.py:91` | Character speech resolver refuses non-audio.cpp profiles (feeds the Console error+override flow) — lifting it is a behavior change |
| P2 | `TTS/profile_service.py:250` `_selection_is_profile_safe` | "Save result as profile" playground eligibility: audio.cpp + WAV + 1.0 + empty options only |
| P3 | `TTS/profile_service.py:533` + `UI/stts_profile_library.py` availability phases | Native-capability catalog probing; availability column and Refresh flow exist only for the native adapter |
| P4 | UI copy | Library header: "Manage exact native audio.cpp model and voice selections." |

Also verified: the admission layer is provider-agnostic — `TTS/request_admission.py`
`_build_request` routes `provider_id == "audio_cpp"` natively and everything else via
`_legacy_request` → `resolve_legacy_route` (`:439-450`), constructing an
`OpenAISpeechRequest` for the six legacy providers. `synthesize_effective` accepts a
`character_profile` of any provider at the type level.

## 4. Design

### 4.1 Slice 1 — lift the gates, per-provider validation

- Replace the audio.cpp-only checks in profile creation/edit with a **single
  per-provider validation table**: allowed response formats and speed bounds per
  provider, sourced from the constraints the Settings model already encodes
  (`UI/Screens/settings_speech_tts.py`). audio.cpp keeps its exact WAV / 1.0 /
  empty-options contract unchanged.
- **Options stay empty for all legacy providers in this slice** (mirrors the
  audio.cpp first-release contract; per-provider options are a later, separately
  validated addition). Model and voice are free-text exact IDs — same policy as the
  Global-defaults "Model value"/"Voice value" fields, which is what makes
  OpenAI-compatible custom endpoints (arbitrary model/voice names) work.
- Lift P1: `CharacterTTSRequestResolver` accepts any provider the validation table
  covers; its existing refusal/override flow remains for genuinely unresolvable
  profiles.
- Lift P2: "Save result as profile" becomes eligible for legacy-provider playground
  results (captures provider/model/voice/format/speed from the generation) — this is
  the primary creation UX for legacy profiles; no new editor form is required.
- **Store schema version bump** (ADR-028's versioned store): pre-expansion builds must
  refuse a store containing non-audio.cpp profiles cleanly rather than half-load it.
- ADR-028 amendment recording the expansion and the empty-options rule.

### 4.2 Slice 2 — availability honesty + library UI

- Availability is a **per-provider capability**: native (audio.cpp) keeps catalog
  probing; legacy providers get an explicit, permanent **"No catalog check for this
  provider"** state. The Refresh flow never spins for a provider it cannot verify;
  the availability column renders the distinction honestly.
- P4 copy updates (library header, any "audio.cpp" phrasing in profile surfaces).
- The profile **edit dialog** accepts the expanded provider set: provider select plus
  free-text model/voice for legacy providers (native audio.cpp keeps its
  catalog-backed selects).
- Persona widget ("Voice & Speech") requires no structural change — it lists
  profiles by UUID; it inherits multi-provider profiles automatically. Verify its
  Preview path for legacy providers.

### 4.3 Slice 3 — default voice profile end-to-end

- New nullable `[app_tts] default_profile_id` (UUID string). Malformed or dangling
  values are a **defined state**: Settings shows the raw value with an explanatory
  notice; speak-time treats it as resolution failure (ruling 5). Hand-edited TOML must
  never crash the panel.
- Resolver: one new source `DEFAULT_PROFILE` between `CHARACTER_PROFILE` and `GLOBAL`
  in `TTS/effective_settings.py`. Non-studio precedence becomes: explicit → character
  profile → **default profile** → global axes → provider fallback. Studio surfaces
  keep studio draft/saved above it.
- Speak-time failure (profile deleted, store unavailable, provider invalid):
  **refuse + one-tap "Speak with global defaults" override**, reusing the character
  path's override-token machinery.
- Settings UI: "Default voice profile" `Select` at the top of Global defaults —
  "None — use the fields below" plus named profiles. **Purity boundary preserved**:
  the pure model (`settings_speech_tts.py`) receives the profile list as static
  choices loaded by the impure screen; if the store is unavailable, the panel renders
  the saved ID + copy ("Voice profile store unavailable — the saved default is kept")
  and never silently drops or clears the setting.
- Reconfiguration: `default_profile_id` flows through the existing settings-save →
  service-reconfiguration path so changes apply without restart (extend the
  reconfiguration tests).
- Deletion integrity: the library's Delete warns when the profile is the app default
  (extend the existing `assignment_count` machinery to count "app default" as a use).

### 4.4 Slice 4 — cross-links + docs

- Settings ▸ Speech & TTS pointer card: profiles are managed in Lab ▸ Speech ▸ Voice
  Profiles; per-character voices in the Roleplay character editor ▸ Voice & Speech.
  Reuse the existing "Open Speech Lab" affordance pattern.
- User Guide: extend `Docs/User_Guide/openai-compatible-tts.md` + index; document the
  default-profile concept and per-character voices.
- File the follow-up task for the sample-persona idea (ruling 1), noting ADR-037's
  constraint (personas do not inherit character TTS assignments; persona runtime
  parity is TASK-617).

**Implemented on `feat/voice-profiles-slice4`, 2026-08-07, PR pending
(correcting the above against what was actually built):** the pointer card
and User Guide items were implemented as written — the card sits in the
panel's existing scope-banner card, right below the "Open Speech Lab"
button it references, and the Roleplay editor's section is indeed
literally titled "Voice & Speech" (`personas_character_tts_widget.py`),
confirmed by reading the widget rather than assumed. The one correction:
**the sample-persona follow-up task was already filed before this slice
began** — as part of slice 1's scoping, not slice 4 — as **TASK-2451**
("Make the default assistant an editable sample persona"), which does cite
ADR-037's constraint. This bullet is a description of intent from the
2026-08-04 design, not a task queued for slice 4 to execute.

## 5. Out of scope

- Persona-*entity* voice assignment (ADR-037; characters remain the voice-bearing
  entity).
- Per-provider profile options beyond empty (later slice, separately validated).
- New speak affordances in other screens (the default layer makes them cheap later).
- Sample persona (follow-up task).

## 6. Testing

- Validation-table accept/reject matrices per provider (formats, speed bounds,
  non-empty options rejected, audio.cpp contract unchanged).
- Resolver precedence: character > default profile > axes; each failure class of the
  default profile refuses with override (mutation-test the refusal).
- Settings binding: persist/delete/reconfigure `default_profile_id`; dangling and
  malformed values; store-unavailable rendering.
- Store version bump: pre-expansion reader refuses cleanly (fixture DB).
- Live verification: character-assigned OpenAI profile speaks through Console 🔊
  against a real provider (repo-root API keys), plus a keyless OpenAI-compatible
  local server (real-socket pattern from `Tests/TTS/test_openai_compatible_endpoint.py`).

## 7. Delivery & coordination

Four PRs matching §4's slices, each with targeted tests + `--collect-only` sweep.
An active speech program (another session) owns adjacent surfaces (V4 realtime engine
pending PR; `feat/speech-console-redesign` branch); **check open speech PRs/branches
before starting each slice** — `TTS/profile_service.py` is the hottest shared file.
Backlog task IDs assigned via the all-worktrees scan with headroom, re-verified at
each merge.
