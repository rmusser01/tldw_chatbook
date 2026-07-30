# Speech Playground: preset ↔ axis-row ownership ruling

**Date:** 2026-07-30
**Decision:** Option A — the pane's model owns axis state; presets are data.
**Decided by:** user (explicit selection), refined against the tree at `96d669d06`.
**Unblocks:** the takeover gate in `test_the_playground_view_mounts_a_playground`
("the pane takes the view over once its axis row and dev's profile presets are
reconciled").

## The question

dev shipped a TTS profile library (save/preview exact presets) onto the legacy
`TTSPlaygroundWidget` while the Speech Console rebuild was in flight. The rebuilt
`SpeechPlaygroundPane` carries a `SpeechAxisRow` whose whole purpose is override
visibility: each comparison axis shows a `*` marker and tooltip when its session
value differs from the persisted default. Both features want to drive the same six
controls. Who owns the value?

## What the tree already settled

The rebuild already made three ownership calls this ruling keeps, not revisits:

1. **The control ids are the contract.** `SpeechAxisRow` mounts the legacy ids
   (`#tts-provider-select` …), so the shared synthesis, catalog, and profile
   mixins read the axes unchanged (`speech_axis_row.py` header docstring).
2. **Presets are already pure data on the way in.**
   `controls_from_profile_preset(...)` → `PlaygroundControls` →
   `_apply_controls(...)` (`speech_catalog_mixin.py:524`) is the single
   translation-then-application path; `_prime_profile_preset_controls`
   (`speech_profile_mixin.py:120`) is the only other writer, for the pre-discovery
   provider chip.
3. **Defaults are written in exactly one place.** `_save_axes_as_default`
   (playground pane) posts `STTSSettingsSaveEvent(TTSPreferencesSnapshot)`; the
   pane "never writes the defaults — overrides are session-scoped by design".

## The defect the ruling resolves

Axis **values** currently live in three places that agree only at construction:

- the Select/Input widgets — authoritative for synthesis and for
  `_save_axes_as_default`;
- `SpeechPlaygroundPane.axis_values` — read once, at `compose()`, to build the row;
- `SpeechAxisRow.values` — read once, in the row's own `compose()`, to paint
  markers and tooltips.

`_apply_controls` and `_prime_profile_preset_controls` write the widgets and
nothing else. User edits (`Select.Changed` → catalog mixin) also touch nothing
else. Consequences, all reproducible on `96d669d06`:

- After the first catalog load, markers/tooltips describe the constructor-time
  values, not the screen.
- A pane opened on a profile preset receives no `axis_values`/`axis_defaults`
  at either prospective mount site (`STTS_Window` constructs the pane nowhere
  yet; the takeover swap would pass only `profile_preset`) — so preset mode
  would show **no override markers at all**, which is the axis row's entire
  reason to exist.
- `defaults` and `values` duplicated across pane and row can drift.

## The ruling

**One owner:** `SpeechPlaygroundPane.axis_values` / `axis_defaults` are the model
of record for axis *presentation* state. The widgets remain the input surface and
the synthesis read-path (unchanged); the row renders the model and owns no state
of its own beyond what it was last given.

**Contracts:**

1. **Single write chokepoint, extended.** Every path that changes an axis value
   updates the model and refreshes the row's markers:
   - `_apply_controls` — after writing the Selects, writes
     `axis_values` for model/voice/format/speed (and provider, from
     `controls.provider_id`), then calls the row's marker refresh.
   - `_prime_profile_preset_controls` — its direct `#tts-provider-select` write
     also sets `axis_values["tts-provider-select"]`. Provider is an axis like
     any other.
   - the pane's `Select.Changed` / speed `Input.Changed` handlers — before
     delegating to the catalog mixin, mirror the user's edit into `axis_values`.
2. **Marker refresh is display-only and in-place.** `SpeechAxisRow` gains
   `update_values(values)` (and the pane passes its dicts by reference or
   re-hands them): re-paint each axis's label text, `speech-chip-override`
   class, and tooltip from `is_override`. No recompose — recompose would
   rebuild the Selects mid-`_apply_controls` and instances do not survive it —
   and no events emitted, so it is safe to call while
   `_applying_catalog_controls` is True.
3. **The guard is reused, never duplicated.** Any new model-driven widget write
   happens inside the existing `_applying_catalog_controls` window. Without it,
   applying a preset fires `Select.Changed`, which reaches
   `_end_profile_preset` (`speech_catalog_mixin.py` ~:1024) and the preset
   detaches itself.
4. **Presets never touch defaults.** Applying a preset writes `axis_values`
   only. Override markers therefore light up relative to the persisted
   defaults — correct: a preset **is** a session-scoped selection. The only
   defaults writers remain `_save_axes_as_default` and the constructor seed.
5. **Defaults are seeded from preferences at construction.** The host
   (`STTS_Window`) builds `axis_defaults` from the persisted
   `TTSPreferencesSnapshot` read path (`TTS/preferences.py`) when constructing
   the pane, for both the plain view and the preset-opened view. Missing
   preference → axis absent from `defaults` → `is_override` returns False (the
   row's documented first-run rule).
6. **Language stays out of scope.** `#tts-language-select` appears nowhere in
   the shared catalog path (kokoro-only in legacy). It participates in the
   model like any axis when something writes it; no new wiring is invented for
   it here.

**Takeover:** with contracts 1–5 in place, the reconciliation the deciding
test's docstring gates on is done. `STTS_Window` then mounts
`SpeechPlaygroundPane` for the playground view (compose ~:4311 and the
`watch_current_view` mount ~:4391, passing `profile_preset` through), and
`test_the_playground_view_mounts_a_playground` flips to assert the pane
specifically. Retiring the legacy widget's code is a separate task and not part
of this ruling.

## Why Option A over the alternatives

- **B (row owns):** the row is rebuilt by any recompose and instances do not
  survive; state in a leaf widget dies with it. It also cannot see preset
  application or catalog loads without new events.
- **C (widgets own, derive markers by querying):** markers become a polling or
  N-handlers problem, and "what is the default?" still needs a home; it leaves
  the pane/row constructor dicts as dead parameters.
- A matches the codebase's established pattern: mixin state on the host,
  widgets as projection (`_applied_model_id` et al. already live on the host).
