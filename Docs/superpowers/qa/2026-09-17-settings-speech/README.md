# Speech & TTS Settings — TASK-32756

Base: `53cad29c5e5ff38dab665cbb3d36c017e8cece2d`, draft PR #2704 against
`dev`. This review covers existing global configuration, keyboard access,
reflow and guarded navigation. Provider/runtime ownership is unchanged.

ADR required: no. Existing ADR-039 (global configuration versus Speech Lab),
ADR-012 (credentials), ADR-150 (tokens) and ADR-161 (component patterns) apply.

## Confirmed defects and repair

- The Voice selector was three rows high inside a one-row action strip.
  Its row now grows to fit the control and stacks with Browse when narrow.
  `voice-picker-red.txt` retains the initial compositor failure.
- At 190×55, the horizontal form exposed a second problem hidden by the
  already-stacked 170-column shell. Field stacks requested the entire
  102-column row beside a 24-column label, leaving only 78 visible columns.
  Browse was absent from the compositor and Speed extended beyond its clip.
  Horizontal field stacks now consume the remaining width; the Voice select
  also leaves room for its adjacent Browse action. The existing stacked rule
  still provides full-width controls. `wide-geometry-red.json`,
  `wide-controls-before.svg` and `wide-browse-red.txt` record the defect.
- The 60-column unsaved-changes dialog clipped **Save and continue**.
  Its three actions now stack at full width. `leave-actions-before.svg` and
  `leave-actions-red.txt` retain visual and automated failures.

The production change is one CSS class and 37 source TCSS lines. The builder
puts the ten modal lines in the boot bundle and the field rules in the lazy
Settings sheet. No token value changes. Boot CSS grows by 219 bytes to
620,281, below the unchanged 634,050 ceiling and above the 600,000 floor.

## Test-fixture repairs

The initial affected cases failed with `raw_source_selection_changed` before
their assertions. Twelve panel functions and ten model functions now use the
existing interpreter-lifetime private-profile helper; model wrappers carry the
async marker required by that helper. Original behavior assertions remain.
The click test also assumed the whole Speech form fitted into 130 terminal
rows. It now uses production CSS at 190×55, scrolls the real target into view
without moving input focus, and checks the actual hit-tested click result.
Baseline excerpts and the click failure are retained in this directory.

Independent review requested actual compact Browse activation, which exposed
the horizontal overflow. The added keyboard test asserts the real
`stts` / `playground` / selected provider / `refresh-voices` message at the
harness boundary. Follow-up reviews accepted the field and modal repairs;
see `review.txt`.

## Final verification

**68 distinct targeted cases pass**; overlapping intermediate runs are not
added to that total:

- `panel.txt`: 13 existing mounted behavior cases for draft recovery, provider
  switching, local guidance, custom IDs, disclosure/action visibility and
  text-entry-safe shortcuts/clicks.
- `model.txt`: 17 ownership/validation cases, including selection-only saves
  without adapter payloads, credential exclusion, identifier/range constraints
  and non-secret restore semantics.
- `journeys.txt`: four production-CSS dark/light × 190×55/80×24 keyboard
  journeys through all global defaults, provider/endpoint controls, visible
  action labels, invalid Speed recovery, custom ID staging, same-control resize,
  and Cancel/Discard navigation; two 190→80 Browse activation cases verify the
  provider-specific Speech Lab handoff.
- `governance.txt`: 32 token, component, generated CSS and boot-budget cases.
- `preflight.txt`: all seven derived-artifact guards pass.
- `static.json` compares Ruff diagnostics and formatter hunks against the exact
  base; no new scoped debt. `static-fatal.txt` passes all five affected Python
  files. Inherited whole-file lint/format debt is preserved.

Final native run `/private/tmp/tldw-32756-native-005`, PID 49639, used real
`TldwCli.run`, LinuxDriver and attached tmux rendering streams with private
HOME/config/data and an acquired instance lock. All four dark/light ×
190×55/80×24 journeys passed. Each performed a real custom-voice Save and a
separate real Speed Save-and-continue, with exact current/legacy TOML deltas;
validation, Revert, Cancel and Discard left saved values unchanged.

All 16 SVG captures and matching terminal text were inspected. The final
`native-result.json` fingerprints match the runner and production sources.
`lifecycle.json` records normal exit zero, app.run return, exact PID absence
before terminal closure, 12 healthy private databases, zero conversations and
messages, no app ERROR records, empty faulthandler output and unchanged default
config/UI-state/policy fingerprints. The owned terminal is closed.
`capture-manifest.json` records original/stored digests; only trailing whitespace
was removed.

## Earlier native attempts

- 001 performed a local save but the runner omitted the established legacy
  `tts_settings` projection from its expected delta. The next profile seeded
  both current and legacy values; the verifier checks both explicitly.
- 002 completed one cell before normal quit interrupted the run for the
  newly reproduced wide-layout problem. Its pending autopilot then timed out
  waiting for Save and exited 1. A later TERM attempt found the exact PID
  already absent; no process was killed. This attempt is unqualified.
- 003 reproduced horizontal Speed clipping in the real app and exited 1.
- 004 completed the wide cell, then inspected compact Save while ordinary
  success toasts covered it. The final runner waits for those transient
  notices to clear before inspecting the underlying focused control. This
  attempt also revealed the clipped third leave-dialog action visually.

`earlier-native-*.json` retains these outcomes. A preparation assertion for
005 initially looked for the modal selector in the lazy sheet; inspection
confirmed the builder correctly emits it in the boot bundle. Final source
and runner fingerprints verify the files actually used by the native run.

## Limits

The native profile uses synthetic OpenAI default selections and performs only
local configuration operations. Mounted Browse activation ends at the captured
navigation message; native checks do not follow it into discovery. No provider
authentication, speech synthesis/playback, model downloads, managed audio.cpp
process operation, credential-store mutation or realtime session is qualified.
Local-provider guidance is checked with explicit dependency fixtures. The full
test suite was not run. Appearance/Theme/Splash and the other Settings reviews
remain separate work in the completion ledger.
