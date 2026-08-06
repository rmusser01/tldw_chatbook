# Tamagotchi as Console Avatar — Brainstorm & Design Proposal

Date: 2026-08-05 (revised after spec review)
Status: Proposal for discussion (not yet approved for implementation)
Related: `tldw_chatbook/Widgets/Tamagotchi/` (existing, currently unmounted widget library)

## Context

The repo already contains a complete Tamagotchi widget library (`Widgets/Tamagotchi/`) that is **not mounted anywhere in the live app**: reactives for state (happiness/hunger/energy/health/age), sprite/mood rendering, personalities, three display sizes (normal/compact/minimal), keyboard interactions (feed/play/sleep/clean/pet), storage adapters, and rate-limited interactions. Only its tests and a demo example reference it.

Accuracy notes (verified against the repo during review):
- `sprite_theme` currently accepts only `emoji|ascii|custom` (`validators.py` `VALID_THEMES`). The `retro|kawaii|minimal` dicts in `tamagotchi_sprites.py` (`ThemePresets`) are orphans referenced by nothing — using them would raise `ValidationError`.
- The mood vocabulary is `happy/neutral/sad/very_sad/hungry/sleepy/sick/dead`, with additional sprite sets for `excited/angry/love`.
- The widget's `on_mount` unconditionally starts a decay `_update_timer`, and `_periodic_update` recomputes mood/sprite from decayed stats every tick — so a render-only avatar cannot be achieved by configuration alone (see "Render-only subclass" below).
- The Console's run state is `ConsoleRunStatus`: `IDLE/VALIDATING/STREAMING/COMPLETED/BLOCKED/STOPPED/FAILED/RETRYING` (`console_chat_models.py`). Tool calls happen *inside* STREAMING; approvals live separately in `task_resume_state.pending_approval`, not in the run status.

The Console (`UI/Screens/chat_screen.py` + `Widgets/Console/`) is the app's primary surface. This proposal explores giving it an avatar-like companion — comparable to the "buddy" in Codex: a small, ambient presence that reflects what the app is doing, rather than a game you play.

## Goal

A lightweight, personality-ful avatar in the Console (and optionally other surfaces) that makes run state glanceable and the app feel alive — without adding gameplay obligations, layout noise, or CPU cost.

Non-goals for phase 1: a full virtual-pet game, persistence obligations (feeding or it "dies"), sounds, any new settings screens.

## Approaches considered

### A. Ambient status avatar (recommended)

A `compact`-size Tamagotchi mounted in the Console's right rail as a collapsible rail section (`console_rail_section.py` already provides the pattern). Its sprite/mood is **driven by app events, not by the pet simulation** — see "Render-only subclass" for the mechanism.

- Trade-offs: smallest scope; zero gameplay burden; reuses sprites/personalities; easy to disable. Gives up "pet" interactivity (deliberately, for phase 1).
- Why recommended: it captures the Codex-buddy feel (ambient, reflective) with the least new machinery.

### B. Full interactive pet companion

Mount the widget as-is (feed/play/pet bindings, decay simulation, persisted state via a storage adapter) in a rail section or modal.

- Trade-offs: the complete existing feature set, but persistent obligations (neglect → sick/dead) distract in a productivity tool, the simulation timers add background work, and it needs persistence design (per-user? per-session?). Possible phase 2 behind a config flag if phase 1 lands well.

### C. Sprite-only cameos

Use the sprite sets purely as loading/thinking art (replace the "Generating…" placeholder, splash screen, run-status glyphs). No persistent avatar.

- Trade-offs: cheapest, but loses the "buddy" continuity. Could be folded into A later.

**Recommendation: A for phase 1**, designed so B and C remain possible without rework.

## Proposed design (Approach A)

### Render-only subclass (required new component)

The existing widget cannot be reused unmodified: `on_mount` starts the decay timer, and `_periodic_update` would clobber externally-assigned moods. Phase 1 therefore adds one small subclass, e.g. `ConsoleBuddyAvatar(BaseTamagotchi)` (in a new `Widgets/Console/console_buddy.py` adapter module), which:

- Overrides `on_mount` so the decay `_update_timer` is **not** started (simulation disabled). With no simulation there is no continuous timer at all; the only remaining timer is `_play_animation`, which is transient (fixed frame sequence, self-stops) and already follows the repo's timer conventions on current dev.
- Sets `BINDINGS = []` and `can_focus = False`, so the pet-game bindings (`f/p/s/c/space`) cannot fire and the avatar never enters the focus chain. This is what keeps it compliant with ADR-031: not because single-letter bindings are forbidden (they aren't), but because advertised footer hints must match live bindings and a focusable widget with hidden active bindings would violate that and could shadow Console bindings.
- Exposes `sync_run_state(status: ConsoleRunStatus, *, pending_approval: bool = False)` — the single public method the adapter/screen calls. Its `on_mount` override must still perform the non-timer parts of the base mount (`_load_state`, initial `_update_sprite`) so the avatar renders immediately — it skips only the decay timer, not the whole base mount.

### Observation mechanism (no new event system)

`_set_run_state` on the controller only assigns `controller.run_state` and appends history; the screen already reacts to run-state changes in its sync methods (mode bar, composer action state). The adapter hooks in there: **one call site** — wherever `ChatScreen` already syncs on run-state transitions — invokes `buddy.sync_run_state(controller.run_state, pending_approval=...)` with the current `ConsoleRunStatus` enum value. No signals, no messages, no polling.

### State mapping (run state → mood, using the real vocabularies)

| App state | Avatar mood |
|---|---|
| `IDLE` (no active run) | `sleepy` or `neutral` (personality-dependent) |
| `VALIDATING` | `neutral` (attentive sprite variant) |
| `STREAMING` | `excited` |
| `BLOCKED` | `sad` |
| `RETRYING` | `neutral` |
| `COMPLETED` | `happy` (brief, then settles to idle mood) |
| `STOPPED` / `FAILED` | `sick` (brief, then settles) |
| `pending_approval` (from `task_resume_state`, any run state) | `hungry` — "wants something from you" (phase 1.5 if the second source proves awkward) |

"Tool call executing" is not a distinct state (it occurs inside STREAMING); no separate row. The mapping table lives in `console_buddy.py`, which owns the avatar instance and translates `sync_run_state` calls into `mood`/`sprite` reactive assignments. "Brief, then settles" is a timed revert via one-shot `set_timer`, not a loop.

### Configuration

- `[console] buddy = true|false` — **default `false`** (see open question 1).
- `[console] buddy_theme = emoji|ascii` (default `emoji` — the only wired themes today; `custom` is deferred).
- `[console] buddy_personality = balanced|...` (selects sprite variants per mood).
- **Adapter-owned fallback** (not validator behavior): invalid theme/personality values log a warning and fall back to defaults; the validator's `ValidationError` is caught by the adapter.

### Error handling & edge cases

- Widget construction validates via `TamagotchiValidator`; the adapter catches `ValidationError`, logs, falls back to defaults — never crashes the Console.
- If the avatar section fails to mount, the Console proceeds without it (mount wrapped, logged).
- Session switches reset the avatar to the idle mood; no state persists across restarts in phase 1.

### Performance

- One small widget, no continuous timer (simulation disabled); the transient sprite animation is a short self-stopping sequence. Mood changes only on run-state transitions — no per-chunk or per-tick work.
- A collapsed-by-default rail section does not mount the avatar until first expanded.

### Testing

- Unit: `sync_run_state` mapping — every `ConsoleRunStatus` value maps to the documented mood; unknown/future statuses fall back to `neutral`; the pending-approval override wins; brief moods revert after the one-shot timer.
- Widget: rail section mounts/unmounts cleanly; `BINDINGS` empty and `can_focus` false; decay timer not started (assert no `_update_timer` after mount).
- Config: `buddy = false` mounts nothing; invalid theme/personality falls back with a warning.
- Existing Console suites must stay green (rail, session surface, run state).

## Open questions (for the user)

1. Default on or off? (The doc currently says default `false` — change to `true` if you want it showcased.)
2. Should it have a name/personality choice exposed anywhere, or stay a fixed "Buddy"?
3. Click-to-pet micro-interaction (pure delight, no gameplay) — in or out for phase 1?
4. Other surfaces wanted now or later: splash screen, home screen, notes? (The adapter is Console-agnostic so this is additive later.)
5. Any mood mappings you want changed (e.g. approvals as `hungry` vs something else)?

## Out of scope (phase 1)

Feed/play/clean gameplay, persistence, sounds, settings-screen UI, per-agent avatars, wiring the orphan `retro|kawaii|minimal` ThemePresets into the validator, animations beyond the existing sprite sets.
